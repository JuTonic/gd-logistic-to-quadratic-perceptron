#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ---------------------- determinism ---------------------- #
SEED = 0
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
    num_workers=0,
)
imgs, labels = next(iter(loader))

imgs = imgs.to(device=device, non_blocking=True, dtype=dtype)
labels = labels.to(device=device, non_blocking=True)

SCALE = 1.0
imgs = imgs * SCALE

n = imgs.shape[0]
X = imgs.reshape(n, -1).contiguous()  # (n,d)
d = X.shape[1]

# y in {-1, +1}
y = torch.where(
    labels == 0,
    torch.tensor(-1.0, device=device),
    torch.tensor(1.0, device=device),
).to(dtype)

# For k=2 convolution with circular shift P: (c*b) = c1*b + c2*(P b)
X_shift = X.roll(shifts=1, dims=1).contiguous()

X_T = X.t().contiguous()
X_shift_T = X_shift.t().contiguous()


# ---------------------- m_cv training ---------------------- #
# "Suitable" step sizes for k=2 (Conv2Linear) used in the paper's CIFAR-10 experiment:
# 2^0, 2^-1, ..., 2^-6
step_sizes = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]

# Optional: if you want to try slightly smaller too
# step_sizes += [0.0078125, 0.00390625]

etas = torch.tensor(step_sizes, device=device, dtype=dtype).view(-1, 1)  # (m,1)
m = etas.shape[0]

T = 400_000
save_every = 100
print_every = 10_000
EPS = 1e-12

# Parameters per eta:
# C: (m,2) for kernel c=[c1,c2], V: (m,d) for v
C = torch.ones(m, 2, device=device, dtype=dtype)  # start at [1,1]
V = torch.zeros(m, d, device=device, dtype=dtype)  # start at 0


@torch.no_grad()
def batched_accuracy_mcv(C, V, X, X_shift, y):
    a1 = X @ V.t()  # (n,m)
    a2 = X_shift @ V.t()  # (n,m)
    logits = a1 * C[:, 0].view(1, -1) + a2 * C[:, 1].view(1, -1)

    preds = torch.sign(logits)
    preds[preds == 0] = 1
    return (preds == y.view(-1, 1)).float().mean(dim=0)  # (m,)


@torch.no_grad()
def batched_gd_step_mcv(C, V, X, X_T, X_shift, X_shift_T, y, etas):
    a1 = X @ V.t()  # (n,m)
    a2 = X_shift @ V.t()  # (n,m)
    logits = a1 * C[:, 0].view(1, -1) + a2 * C[:, 1].view(1, -1)

    s = torch.sigmoid(-y.view(-1, 1) * logits)  # (n,m)
    ys = y.view(-1, 1) * s  # (n,m)
    n = X.shape[0]

    # grad wrt c1,c2
    g_c1 = -(ys * a1).mean(dim=0)  # (m,)
    g_c2 = -(ys * a2).mean(dim=0)  # (m,)

    # grad wrt v
    tmp1 = ys * C[:, 0].view(1, -1)  # (n,m)
    tmp2 = ys * C[:, 1].view(1, -1)  # (n,m)
    g_v = -((X_T @ tmp1) + (X_shift_T @ tmp2)).t() / n  # (m,d)

    # GD updates
    C[:, 0].add_(g_c1 * (-etas.view(-1)))
    C[:, 1].add_(g_c2 * (-etas.view(-1)))
    V.add_(g_v * (-etas))


print(f"m_cv (k=2) | n={n}, d={d}, device={device}, SCALE={SCALE}, etas={step_sizes}")

iter_list = []
acc_lists = {eta: [] for eta in step_sizes}

for t in range(T + 1):
    if t % save_every == 0:
        acc_t = batched_accuracy_mcv(C, V, X, X_shift, y)
        acc = acc_t.detach().cpu().tolist()

        iter_list.append(t)
        for j, eta in enumerate(step_sizes):
            acc_lists[eta].append(acc[j])

        if t % print_every == 0:
            parts = [f"iter {t}/{T}"]
            for j, eta in enumerate(step_sizes):
                parts.append(f"eta={eta:g} acc={acc[j]:.4f}")
            print("  ".join(parts))

        # early stop if any eta reaches 100%
        if float(acc_t.max().item()) >= 1.0 - EPS:
            best_j = int(torch.argmax(acc_t).item())
            print(f"Stopping: eta={step_sizes[best_j]:g} reached 1.0 acc at iter={t}.")
            break

    if t < T:
        batched_gd_step_mcv(C, V, X, X_T, X_shift, X_shift_T, y, etas)

histories = {eta: (iter_list, acc_lists[eta]) for eta in step_sizes}
torch.save(histories, "results/mcv_k2_histories.pt")
torch.save({"C": C.detach().cpu(), "V": V.detach().cpu()}, "results/mcv_k2_params.pt")
print(
    "Saved histories to results/mcv_k2_histories.pt and params to results/mcv_k2_params.pt"
)

plt.figure(figsize=(7, 4))
for eta in step_sizes:
    it, acc = histories[eta]
    plt.plot(it, acc, label=f"m_cv step {eta:g}")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.01)
plt.legend()
plt.tight_layout()
plt.show()
