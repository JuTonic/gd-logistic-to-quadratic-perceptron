import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 normalization as in many standard baselines (not required but reasonable)
transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# select classes 0 and 1
targets = np.array(trainset.targets)
idx0 = np.where(targets == 0)[0]
idx1 = np.where(targets == 1)[0]

# subsample 2500 from each class
rng = np.random.default_rng(0)
idx0_sub = rng.choice(idx0, size=2500, replace=False)
idx1_sub = rng.choice(idx1, size=2500, replace=False)
indices = np.concatenate([idx0_sub, idx1_sub])
subset = Subset(trainset, indices)

loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
imgs, labels = next(iter(loader))  # full batch
imgs = imgs.to(device)  # (n, 3, 32, 32)
labels = labels.to(device)

n = imgs.shape[0]
d = 3 * 32 * 32

# flatten inputs
X = imgs.view(n, d)

# map labels to {-1, +1}
y = labels.clone()
y[labels == 0] = -1
y[labels == 1] = 1
y = y.float()


def logistic_loss(v, X, y):
    # v: (d,), X: (n,d), y: (n,)
    logits = X @ v  # (n,)
    # logistic loss: (1/n) sum log(1 + exp(-y * logits))
    return torch.log1p(torch.exp(-y * logits)).mean()


def grad_logistic(v, X, y):
    logits = X @ v
    # gradient: -(1/n) sum y * x / (1 + exp(y * v^T x))
    s = 1.0 / (1.0 + torch.exp(y * logits))  # (n,)
    g = -(y * s).unsqueeze(1) * X  # (n,d)
    return g.mean(dim=0)  # (d,)


step_sizes = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
T = 5000  # iterations as in the figure
eval_every = 1000  # evaluate accuracy every k steps

histories = {}

for eta in step_sizes:
    v = torch.zeros(d, device=device)  # v_0 = 0
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

        # full-batch GD step
        g = grad_logistic(v, X, y)
        v = v - eta * g

    histories[eta] = (iter_list, acc_list)

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
