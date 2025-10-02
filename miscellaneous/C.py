# simple_ij_sgd.py  ───────────────────────────────────────────────
"""
Pure, first‑principles Infinitesimal‑Jackknife on top of *serial* SGD.
  • Dataset   : configurable (linear by default)
  • Model     : tiny 1‑hidden‑layer MLP (tanh)  -- fits most toy curves
  • Influence : exact ± formula  (+1 for k, −1/n otherwise)
  • Output    :  prediction curve + IJ 95 % band
This matches the math in your original notes.
"""
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt

# ───────────────────  choose dataset  ───────────────────
DATASET   = "linear"           # "linear" | "quadratic" | "sine"
N         = 1000                # training points
NOISE_STD = 0.1

torch.manual_seed(0)
x = torch.linspace(-3, 3, N).unsqueeze(1)
if DATASET == "linear":
    y_true = 26*x + 35 
elif DATASET == "quadratic":
    y_true = 2*x**2 + .5*x + 1
elif DATASET == "sine":
    y_true = torch.sin(3*x)
else:
    raise ValueError("Unknown dataset")

y = y_true + NOISE_STD * torch.randn_like(x)

# ───────────────────  network (tiny)  ───────────────────
net = nn.Sequential(
    nn.Linear(1, 32), nn.Tanh(),
    nn.Linear(32, 1)
)

μ      = 2e-3
EPOCHS = 1000
loss_fn= nn.MSELoss(reduction='none')      # keep per‑sample loss

# influence matrix U  (n × p)
P = sum(p.numel() for p in net.parameters())
U = torch.zeros(N, P)

# helper to flatten grads
def flat_grad():
    return torch.cat([p.grad.reshape(-1) for p in net.parameters()])

# ───────────────────  SGD loop with IJ update  ──────────
for epoch in range(EPOCHS):
    for k in range(N):                         # pure serial SGD
        xk, yk = x[k:k+1], y[k:k+1]

        net.zero_grad()
        loss = loss_fn(net(xk), yk)            # scalar (since batch=1)
        loss.backward(create_graph=True)

        gk = flat_grad().detach()              # ∇_θ L_k
        # ----- SGD step -----
        with torch.no_grad():
            θ = flat_grad().detach() * 0       # dummy to get shape
            θ = torch.nn.utils.parameters_to_vector(net.parameters())
            θ = θ - μ * gk
            torch.nn.utils.vector_to_parameters(θ, net.parameters())
        # ----- IJ influence update  (+1 for k, −1/n for others) -----
        U -= μ * ((-1/N) * gk)                 # all rows get −μ/N g_k
        U[k] += μ * gk                         # row k gets an extra +μ g_k

# ───────────────────  compute IJ band on grid  ──────────
grid = torch.linspace(-3, 3, 400).unsqueeze(1)
net.eval()
with torch.no_grad():
    resid = (net(x) - y)
    sigma2 = resid.pow(2).mean().item()           # noise variance

y_hat, std = [], []
for x_star in grid:
    net.zero_grad()
    out = net(x_star.unsqueeze(0))
    out.backward()
    g_f = flat_grad().detach()
    var = sigma2 * ((U @ g_f)**2).sum().item()
    y_hat.append(out.item())
    std.append(np.sqrt(var))

y_hat, std = np.array(y_hat), np.array(std)

# ───────────────────  visualize  ────────────────────────
plt.figure(figsize=(10,4))
plt.scatter(x, y, s=10, alpha=.3, label='train pts')
if DATASET == "linear":
    plt.plot(grid, 2*grid+3, '--', c='gray', label='true line')
elif DATASET == "quadratic":
    plt.plot(grid, 2*grid**2 + .5*grid + 1, '--', c='gray', label='true quad')
else:
    plt.plot(grid, np.sin(3*grid), '--', c='gray', label='true sin')
plt.plot(grid, y_hat, c='tab:blue', lw=2, label='pred mean')
plt.fill_between(grid.squeeze(), y_hat-1.96*std, y_hat+1.96*std,
                 color='tab:blue', alpha=.25, label='IJ 95%')
plt.title(f"{DATASET.capitalize()} — IJ band with classic ± formula")
plt.xlabel("x"); plt.ylabel("y / ŷ"); plt.legend(); plt.tight_layout(); plt.show()
