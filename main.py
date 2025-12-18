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

loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
imgs, labels = next(iter(loader))
imgs = imgs.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)

n = imgs.shape[0]
d = 3 * 32 * 32
X = imgs.view(n, d)

# y in {-1, +1}
y = torch.where(
    labels == 0, torch.tensor(-1.0, device=device), torch.tensor(1.0, device=device)
)

# precompute once for speed
X_T = X.t()  # (d, n)

# ---------------------- core ops ---------------------- #


@torch.no_grad()
def gd_step(v, X, X_T, y, eta):
    # logits = X @ v
    logits = X.mv(v)  # (n,)
    # s = 1 / (1 + exp(y * logits))
    s = torch.sigmoid(-y * logits)  # numerically stable
    # gradient: g = -(1/n) X^T (y * s)
    g = -(X_T.mv(y * s)) / X.shape[0]  # (d,)
    v.add_(g, alpha=-eta)  # in-place: v = v - eta * g


@torch.no_grad()
def accuracy(v, X, y):
    logits = X.mv(v)
    preds = torch.sign(logits)
    return (preds == y).float().mean().item()


# ---------------------- training loop ---------------------- #

step_sizes = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
T = 300000
eval_every = 1000

histories = {}

for eta in step_sizes:
    print(f"\n=== Step size {eta} ===")
    v = torch.zeros(d, device=device)
    acc_list = []
    iter_list = []

    for t in range(T + 1):
        # evaluate accuracy
        if t % eval_every == 0:
            with torch.no_grad():
                logits = X @ v
                preds = torch.sign(logits)
                correct = (preds == y).float().mean().item()
                acc_list.append(correct)
                iter_list.append(t)
                print(f"[eta={eta}] iter {t}/{T}  acc={correct:.4f}")

        # full-batch GD step
        g = grad_logistic(v, X, y)
        v = v - eta * g

    histories[eta] = (iter_list, acc_list)

# ---------------------- plotting ---------------------- #

os.makedirs("results", exist_ok=True)
history_path = "results/linear_histories.pt"

torch.save(histories, history_path)
print(f"Saved histories to {history_path}")

plt.figure(figsize=(7, 4))
for eta in step_sizes:
    it, acc = histories[eta]
    plt.plot(it, acc, label=f"Linear model step {eta}")

plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1.01)
plt.legend()
plt.tight_layout()
plt.show()
