import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------- determinism ---------------------- #
SEED = 0
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
dtype = torch.float64  # helps stability

# ---------------------- data: CIFAR10 {0,1} ---------------------- #
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

SCALE = 1.0  # ToTensor already gives [0,1]
imgs = imgs * SCALE

n = imgs.shape[0]
X = imgs.reshape(n, -1).contiguous()  # (n,d)
d = X.shape[1]

y = torch.where(
    labels == 0,
    -torch.ones_like(labels, dtype=dtype),
    torch.ones_like(labels, dtype=dtype),
).to(device)

# P is one-step circular shift (k=2 case uses x and Px)
X_shift = torch.roll(X, shifts=1, dims=1).contiguous()
X_T = X.t().contiguous()
X_shift_T = X_shift.t().contiguous()

print(f"n={n}, d={d}, device={device}, dtype={dtype}")


# ---------------------- helpers ---------------------- #
@torch.no_grad()
def split_w(w):
    # w = [c1, c2, v] in R^{d+2}
    c1 = w[0]
    c2 = w[1]
    v = w[2:]
    return c1, c2, v


@torch.no_grad()
def scores_from_w(w):
    """
    score_i = (1/2) w^T A_i w = y_i * v^T( (c1 I + c2 P) b_i )
    This is the quantity whose sign determines correctness, and whose <=0 defines S_t in QPA.
    """
    c1, c2, v = split_w(w)
    Av = X.mv(v)  # <b_i, v>
    Bv = X_shift.mv(v)  # <P b_i, v>
    return y * (c1 * Av + c2 * Bv)  # (n,)


@torch.no_grad()
def acc_from_scores(scores):
    # correct if score > 0, tie broken as correct on score==0? we’ll treat >=0 as correct
    return (scores >= 0).float().mean().item()


@torch.no_grad()
def pred_sign(scores):
    # prediction for y*mcv: +1 if score>=0 else -1
    return torch.where(scores >= 0, torch.ones_like(scores), -torch.ones_like(scores))


# ---------------------- GD step in w-space (Eq. (10)) ---------------------- #
@torch.no_grad()
def gd_w_step(w, gamma):
    """
    Implements:
      w_{t+1} = w_t + (gamma/n) sum_i [ 1/(1+exp( 1/2 w^T A_i w )) ] A_i w
    using the structure of A_i (no explicit (d+2)x(d+2) matrices).
    """
    c1, c2, v = split_w(w)
    scores = scores_from_w(w)  # (n,)
    alpha = torch.sigmoid(-scores).clamp(0, 1)  # (n,)

    # a_i = y_i b_i ; Pa_i = y_i P b_i
    wy = alpha * y  # (n,)

    Av = X.mv(v)  # (n,)
    Bv = X_shift.mv(v)  # (n,)
    dot1 = (wy * Av).sum()  # sum alpha_i * y_i <b_i,v>
    dot2 = (wy * Bv).sum()  # sum alpha_i * y_i <P b_i,v>

    sum_a = X_T.mv(wy)  # sum alpha_i * y_i b_i      (d,)
    sum_Pa = X_shift_T.mv(wy)  # sum alpha_i * y_i P b_i    (d,)

    # A_i w summed with weights alpha:
    # top coords: dot1, dot2
    # bottom: c1 * sum_a + c2 * sum_Pa
    upd_c1 = (gamma / n) * dot1
    upd_c2 = (gamma / n) * dot2
    upd_v = (gamma / n) * (c1 * sum_a + c2 * sum_Pa)

    w_new = w.clone()
    w_new[0] += upd_c1
    w_new[1] += upd_c2
    w_new[2:] += upd_v
    return w_new


# ---------------------- Quadratic Perceptron Algorithm (Theorem 2.1) ---------------------- #
@torch.no_grad()
def qpa_step(theta, gamma, noise_std=0.0):
    """
    theta_{t+1} = theta_t + (gamma/n) sum_{i in S_t} A_i theta_t (+ optional tiny noise)
    S_t = { i : (1/2) theta^T A_i theta <= 0 }
    """
    c1, c2, v = split_w(theta)
    scores = scores_from_w(theta)  # (n,)
    mask = scores <= 0  # S_t
    m = mask.to(dtype=dtype)

    wy = m * y  # (n,)

    Av = X.mv(v)
    Bv = X_shift.mv(v)
    dot1 = (wy * Av).sum()
    dot2 = (wy * Bv).sum()

    sum_a = X_T.mv(wy)
    sum_Pa = X_shift_T.mv(wy)

    upd_c1 = (gamma / n) * dot1
    upd_c2 = (gamma / n) * dot2
    upd_v = (gamma / n) * (c1 * sum_a + c2 * sum_Pa)

    theta_new = theta.clone()
    theta_new[0] += upd_c1
    theta_new[1] += upd_c2
    theta_new[2:] += upd_v

    if noise_std > 0:
        theta_new += noise_std * torch.randn_like(theta_new)

    return theta_new


# ---------------------- run comparison ---------------------- #
gamma = 2.0 ** (-6)  # 0.015625 (safe)
T = 2000
print_every = 100

# random direction theta0, then huge-norm w0 = R * theta0
theta0 = torch.randn(d + 2, device=device, dtype=dtype)
theta0 = theta0 / theta0.norm()
R = 1e4  # make GD close to the "large norm" regime
w = R * theta0.clone()
theta = theta0.clone()

gd_acc_hist = []
qpa_acc_hist = []
cos_hist = []
mismatch_hist = []

for t in range(T + 1):
    # evaluate
    s_w = scores_from_w(w)
    s_th = scores_from_w(theta)

    acc_w = acc_from_scores(s_w)
    acc_th = acc_from_scores(s_th)

    # compare predictions (sign of score)
    pw = pred_sign(s_w)
    pth = pred_sign(s_th)
    mismatch = (pw != pth).float().mean().item()

    # compare directions
    w_dir = w / w.norm()
    th_dir = theta / theta.norm()
    cos = torch.dot(w_dir, th_dir).item()

    gd_acc_hist.append(acc_w)
    qpa_acc_hist.append(acc_th)
    cos_hist.append(cos)
    mismatch_hist.append(mismatch)

    if t % print_every == 0:
        print(
            f"t={t:4d}  acc(GD)={acc_w:.4f}  acc(QPA)={acc_th:.4f}  "
            f"cos(dir)={cos:.6f}  pred_mismatch={mismatch:.6f}"
        )

    if t < T:
        # GD step (w)
        w = gd_w_step(w, gamma)

        # QPA step (theta) — optionally add tiny noise if you ever get "stuck"
        theta = qpa_step(theta, gamma, noise_std=0.0)

# ---------------------- plots ---------------------- #
plt.figure(figsize=(7, 4))
plt.plot(gd_acc_hist, label="GD with m_cv (w-update)")
plt.plot(qpa_acc_hist, label="Quadratic Perceptron (theta)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.01)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(cos_hist, label="cosine(w/||w||, theta/||theta||)")
plt.plot(mismatch_hist, label="prediction mismatch fraction")
plt.xlabel("Iteration")
plt.legend()
plt.tight_layout()
plt.show()
