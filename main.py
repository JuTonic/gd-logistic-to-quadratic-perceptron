import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------- setup & data ---------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

targets = np.array(trainset.targets)
idx0 = np.where(targets == 0)[0]
idx1 = np.where(targets == 1)[0]

rng = np.random.default_rng(0)
idx0_sub = rng.choice(idx0, size=2500, replace=False)
idx1_sub = rng.choice(idx1, size=2500, replace=False)
indices = np.concatenate([idx0_sub, idx1_sub])
subset = Subset(trainset, indices)

loader = DataLoader(
    subset,
    batch_size=len(subset),
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)
imgs, labels = next(iter(loader))
imgs = imgs.to(device, non_blocking=True).float()
labels = labels.to(device, non_blocking=True)

n = imgs.shape[0]
d = 3 * 32 * 32
X = imgs.view(n, d)

# y in {-1, +1}
y = torch.where(
    labels == 0,
    -torch.ones_like(labels, dtype=torch.float32),
    torch.ones_like(labels, dtype=torch.float32),
)
y = y.to(device)

X_T = X.t()  # (d, n)

# ---------------------- batched training over step sizes ---------------------- #

step_sizes = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
etas = torch.tensor(step_sizes, device=device, dtype=torch.float32).view(-1, 1)  # (m,1)
m = etas.shape[0]

T = 400_000
eval_every = 1000

# V has one row per step size
V = torch.zeros(m, d, device=device)

# history buffers
iter_list = []
acc_lists = {eta: [] for eta in step_sizes}


@torch.no_grad()
def batched_accuracy(V, X, y):
    # logits: (n,m)
    logits = X @ V.t()
    preds = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
    accs = (preds == y.view(-1, 1)).float().mean(dim=0)  # (m,)
    return accs


@torch.no_grad()
def batched_gd_step(V, X, X_T, y, etas):
    # logits: (n,m)
    logits = X @ V.t()
    # s = sigmoid(-y*logits): (n,m)
    s = torch.sigmoid(-y.view(-1, 1) * logits)
    # tmp = y*s: (n,m)
    tmp = y.view(-1, 1) * s
    # g: (d,m) = -(1/n) X^T tmp
    g = -(X_T @ tmp) / X.shape[0]
    # V = V - etas * g^T  (etas is (m,1), g^T is (m,d))
    V.add_(g.t() * (-etas))


print(f"Training {m} step sizes in parallel on {device} ...")

for t in range(T + 1):
    if t % eval_every == 0:
        accs = batched_accuracy(V, X, y).detach().cpu().tolist()
        iter_list.append(t)
        line = [f"iter {t}/{T}"]
        for j, eta in enumerate(step_sizes):
            acc = accs[j]
            acc_lists[eta].append(acc)
            line.append(f"eta={eta:g} acc={acc:.4f}")
        print("  ".join(line))

    if t < T:
        batched_gd_step(V, X, X_T, y, etas)

histories = {eta: (iter_list, acc_lists[eta]) for eta in step_sizes}

# ---------------------- plotting + saving ---------------------- #

os.makedirs("results", exist_ok=True)
history_path = "results/linear_histories.pt"
torch.save(histories, history_path)
print(f"Saved histories to {history_path}")

plt.figure(figsize=(7, 4))
for eta in step_sizes:
    it, acc = histories[eta]
    plt.plot(it, acc, label=f"Linear model step {eta:g}")

plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.01)
plt.legend()
plt.tight_layout()
plt.show()
