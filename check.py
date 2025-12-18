import torch

history_path = "results/linear_histories.pt"
histories = torch.load(history_path, map_location="cpu")

hits = []
best = []

for eta, payload in histories.items():
    iters, accs = payload[0], payload[1]

    # exact hits
    for t, a in zip(iters, accs):
        if float(a) == 1.0:
            hits.append((float(eta), int(t), float(a)))
            break  # first time it hits 1.0

    # best (in case it never hits exactly 1.0)
    max_a = max(map(float, accs))
    t_max = iters[list(map(float, accs)).index(max_a)]
    best.append((float(eta), int(t_max), max_a))

hits.sort()
best.sort()

if hits:
    print("✅ Found accuracy == 1.0:")
    for eta, t, a in hits:
        print(f"  eta={eta:g} first hits 1.0 at iter {t}")
else:
    print("❌ No exact 1.0 found.")
    print("Best per step size (maybe close to 1.0):")
    for eta, t, a in best:
        print(f"  eta={eta:g} best={a:.6f} at iter {t}")

# Optional: also check "near 1.0" with toler
