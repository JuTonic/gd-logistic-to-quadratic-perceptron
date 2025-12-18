import torch
import matplotlib.pyplot as plt

history_path = "results/linear_histories.pt"

# Load (works whether file was saved on GPU or CPU)
histories = torch.load(history_path, map_location="cpu")

plt.figure(figsize=(7, 4))

for eta, payload in sorted(histories.items(), key=lambda kv: float(kv[0])):
    # supports (iters, accs) or (iters, accs, losses)
    iters = payload[0]
    accs = payload[1]
    plt.plot(iters, accs, label=f"η={float(eta):g}")

plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.01)
plt.legend()
plt.tight_layout()
plt.show()
