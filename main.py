import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------- determinism ---------------------- #
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# cuDNN determinism settings (mostly relevant for convs, but keep consistent)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = (
    False  # important: benchmark can override determinism in practice [web:63][web:62]
)
torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# ---------------------- data ---------------------- #
# NOTE: ToTensor maps uint8 [0..255] -> float [0..1]
transform = transforms.ToTensor()
trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

targets = np.array(trainset.targets)
idx0 = np.where(targets == 0)[0]
idx1 = np.where(targets == 1)[0]

os.makedirs("results", exist_ok=True)
idx_path = "results/cifar10_cls01_idx_seed0.npy"

# Save indices so the subset is exactly repeatable across runs
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
    num_workers=0,  # keep deterministic
)
imgs, labels = next(iter(loader))

imgs = imgs.to(device=device, non_blocking=True, dtype=dtype)
labels = labels.to(device=device, non_blocking=True)

# Optional: if you want to test different scaling, toggle this.
# Using scale=255 changes the effective learning rates dramatically.
SCALE = 1.0
imgs = imgs * SCALE

n = imgs.shape[0]
X = imgs.reshape(n, -1).contiguous()  # (n,d)
d = X.shape[1]

# y in {-1, +1}
y = torch.where(
    labels == 0, torch.tensor(-1.0, device=device), torch.tensor(1.0, device=device)
).to(dtype)

X_T = X.t().contiguous()

# ---------------------- training ---------------------- #
step_sizes = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
etas = torch.tensor(step_sizes, device=device, dtype=dtype).view(-1, 1)  # (m,1)
m = etas.shape[0]

T = 400_000
eval_every = 1000

V = torch.zeros(m, d, device=device, dtype=dtype)  # one row per eta

iter_list = []
acc_lists = {eta: [] for eta in step_sizes}


@torch.no_grad()
def batched_accuracy(V, X, y):
    logits = X @ V.t()  # (n,m)
    preds = torch.sign(logits)
    preds[preds == 0] = 1  # explicit tie-breaking (avoid 0-labels)
    return (preds == y.view(-1, 1)).float().mean(dim=0)  # (m,)


@torch.no_grad()
def batched_gd_step(V, X, X_T, y, etas):
    logits = X @ V.t()  # (n,m)
    s = torch.sigmoid(-y.view(-1, 1) * logits)  # (n,m)
    tmp = y.view(-1, 1) * s  # (n,m)
    g = -(X_T @ tmp) / X.shape[0]  # (d,m)
    V.add_(g.t(), alpha=-1.0)  # V <- V - g^T
    V.add_(g.t() * 0.0)  # no-op; keep as placeholder to show no extra coupling
    V.add_(g.t() * (-etas + etas))  # no-op; ensure broadcasting sanity
    V.add_(g.t() * (-etas))  # V <- V - etas * g^T


print(f"Subset size n={n}, dimension d={d}, device={device}, SCALE={SCALE}")

for t in range(T + 1):
    if t % eval_every == 0:
        accs = batched_accuracy(V, X, y).detach().cpu().tolist()
        iter_list.append(t)

        parts = [f"iter {t}/{T}"]
        for j, eta in enumerate(step_sizes):
            acc = accs[j]
            acc_lists[eta].append(acc)
            parts.append(f"eta={eta:g} acc={acc:.4f}")
        print("  ".join(parts))

        # early stop if any eta reaches 100%
        if max(accs) >= 1.0:
            print("Reached 100% training accuracy for at least one step size.")
            # keep going if you want full curves; otherwise break
            # break

    if t < T:
        batched_gd_step(V, X, X_T, y, etas)

histories = {eta: (iter_list, acc_lists[eta]) for eta in step_sizes}
torch.save(histories, "results/linear_histories.pt")
print("Saved histories to results/linear_histories.pt")

plt.figure(figsize=(7, 4))
for eta in step_sizes:
    it, acc = histories[eta]
    plt.plot(it, acc, label=f"Linear model step {eta:g}")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.01)
plt.legend()
plt.tight_layout()
plt.show()
