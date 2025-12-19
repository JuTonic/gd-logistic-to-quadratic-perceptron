import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ====================== determinism ====================== #
SEED = 0
os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
    ":4096:8"  # needed for deterministic matmul on CUDA
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# ====================== data: CIFAR10 classes {0,1} ====================== #
transform = transforms.ToTensor()
trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

targets = np.asarray(trainset.targets)
idx0 = np.where(targets == 0)[0]
idx1 = np.where(targets == 1)[0]

os.makedirs("results", exist_ok=True)
idx_path = "results/cifar10_cls01_idx_seed0.npy"

if os.path.exists(idx_path):
    indices = np.load(idx_path)
else:
    rng = np.random.default_rng(SEED)
    idx0_sub = rng.choice(idx0, size=2500, replace=False)
    idx1_sub = rng.choice(idx1, size=2500, replace=False)
    indices = np.concatenate([idx0_sub, idx1_sub])
    np.save(idx_path, indices)

subset = Subset(trainset, indices)

loader = DataLoader(
    subset,
    batch_size=len(subset),
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
)

imgs, labels = next(iter(loader))
imgs = imgs.to(device=device, non_blocking=True, dtype=dtype)
labels = labels.to(device=device, non_blocking=True)

# ToTensor already scales to [0,1]. Keep SCALE=1.0 to match that.
SCALE = 1.0
imgs = imgs * SCALE

n = imgs.shape[0]
X = imgs.reshape(n, -1).contiguous()  # (n,d)
d = X.shape[1]

# y in {-1,+1}
y = torch.where(
    labels == 0,
    -torch.ones_like(labels, dtype=dtype),
    torch.ones_like(labels, dtype=dtype),
).to(device)

# "P" operator for k=2 circular convolution: one-step circular shift
X_shift = torch.roll(X, shifts=1, dims=1).contiguous()

X_T = X.t().contiguous()
X_shift_T = X_shift.t().contiguous()

print(f"Subset n={n}, d={d}, device={device}, dtype={dtype}, SCALE={SCALE}")

# ====================== m_cv model + GD (parallel over step sizes) ====================== #
# Paper Fig 1(b): gamma in {2^-6, ..., 2^0}
step_sizes = [2.0 ** (-p) for p in range(9, 3, -1)]  # 2^-6 .. 2^0
etas = torch.tensor(step_sizes, device=device, dtype=dtype).view(-1, 1)  # (m,1)
m = len(step_sizes)

T = 500_000
save_every = 1
print_every = 10_000

# IMPORTANT: nonzero init; otherwise bilinear model can get stuck at all-zeros
init_scale = 1e-2
C = init_scale * torch.randn(m, 2, device=device, dtype=dtype)  # kernels (c0,c1)
V = init_scale * torch.randn(m, d, device=device, dtype=dtype)  # weights


@torch.no_grad()
def mcv_logits(X, X_shift, V, C):
    # A_j = <x, v_j>, B_j = <Px, v_j>, logits = c0*A + c1*B
    A = X @ V.t()  # (n,m)
    B = X_shift @ V.t()  # (n,m)
    c0 = C[:, 0].view(1, -1)  # (1,m)
    c1 = C[:, 1].view(1, -1)
    return A * c0 + B * c1, A, B


@torch.no_grad()
def batched_accuracy_from_logits(logits, y):
    preds = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
    return (preds == y.view(-1, 1)).float().mean(dim=0)  # (m,)


@torch.no_grad()
def mcv_batched_gd_step(X, X_T, X_shift, X_shift_T, y, V, C, etas):
    # forward
    logits, A, B = mcv_logits(X, X_shift, V, C)  # (n,m), (n,m), (n,m)

    # dL/dlogits = -(y * sigmoid(-y*logits)) / n
    ycol = y.view(-1, 1)
    dl = -(ycol * torch.sigmoid(-ycol * logits)) / X.shape[0]  # (n,m)

    # grads for C: gC0 = sum dl*A, gC1 = sum dl*B  (already includes /n)
    gC0 = (dl * A).sum(dim=0)  # (m,)
    gC1 = (dl * B).sum(dim=0)  # (m,)
    gC = torch.stack([gC0, gC1], dim=1)  # (m,2)

    # grads for V:
    # gV^T = X^T (dl*c0) + X_shift^T (dl*c1)
    c0 = C[:, 0].view(1, -1)  # (1,m)
    c1 = C[:, 1].view(1, -1)
    term1 = X_T @ (dl * c0)  # (d,m)
    term2 = X_shift_T @ (dl * c1)  # (d,m)
    gV = (term1 + term2).t().contiguous()  # (m,d)

    # updates (one eta per row)
    V.add_(gV * (-etas))  # V -= eta * gV
    C.add_(gC * (-etas))  # C -= eta * gC


# ====================== train + log ====================== #
iter_list = []
acc_lists = {eta: [] for eta in step_sizes}

for t in range(T + 1):
    if t % save_every == 0:
        logits, _, _ = mcv_logits(X, X_shift, V, C)
        accs_t = batched_accuracy_from_logits(logits, y)
        accs = accs_t.detach().cpu().tolist()

        iter_list.append(t)
        for j, eta in enumerate(step_sizes):
            acc_lists[eta].append(accs[j])

        if t % print_every == 0:
            parts = [f"iter {t}/{T}"]
            parts += [f"eta={step_sizes[j]:g} acc={accs[j]:.4f}" for j in range(m)]
            print("  ".join(parts))

    if t < T:
        mcv_batched_gd_step(X, X_T, X_shift, X_shift_T, y, V, C, etas)

histories = {eta: (iter_list, acc_lists[eta]) for eta in step_sizes}
out_path = "results/mcv_histories.pt"
torch.save(histories, out_path)
print(f"Saved histories to {out_path}")

# ====================== plot ====================== #
plt.figure(figsize=(7, 4))
for eta in step_sizes:
    it, acc = histories[eta]
    plt.plot(it, acc, label=f"m_cv η={eta:g}")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.01)
plt.legend()
plt.tight_layout()
plt.show()
