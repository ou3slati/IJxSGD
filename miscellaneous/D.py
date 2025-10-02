import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION & REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

n_samples   = 800        # number of training points
noise_level = 0.1        # σ of Gaussian noise
μ0          = 1e-2       # initial SGD step size
decay       = 5e-3       # per‑epoch decay rate
n_epochs    = 200        # how many full passes
track_idx   = 42         # which sample to log grad‑norm
x_star      = 0.0        # test point for CI

# ──────────────────────────────────────────────────────────────────────────
# 1) SYNTHETIC LINEAR DATA:  y = 3x + 0.5 + ε
# ──────────────────────────────────────────────────────────────────────────
x = torch.randn(n_samples, 1)
y =  3.0 * x + 0.5 + noise_level * torch.randn_like(x)

# feature map φ(x)=x (we’ll prepend the intercept by hand in OLS)
def phi(xx): return xx

# ──────────────────────────────────────────────────────────────────────────
# 2) MODEL:  f(x)=w x + b via a single nn.Linear
# ──────────────────────────────────────────────────────────────────────────
model = nn.Linear(1, 1, bias=True)

# ──────────────────────────────────────────────────────────────────────────
# 3) HELPERS
#   – flatten_grad: stack all ∂L/∂θ into a (p,) vector
#   – ij_update : perform U ← U − μ [δ_{ik}−1/n] ∇θL_k
# ──────────────────────────────────────────────────────────────────────────
def flatten_grad(m):
    vs = [p.grad if p.grad is not None else torch.zeros_like(p)
          for p in m.parameters()]
    return torch.nn.utils.parameters_to_vector(vs)

def ij_update(U, k, g, mu):
    # build jackknife scale s_i = +1 at i=k, else -1/n
    n = U.shape[0]
    scale = torch.full((n,1), -1.0/n)
    scale[k] = 1.0
    # broadcast-multiply g (shape=(p,)) across rows → (n×p)
    return U - mu * scale * g

# ──────────────────────────────────────────────────────────────────────────
# 4) INITIALIZE IJ MATRIX
#    U has shape (n_samples × p_dim)
# ──────────────────────────────────────────────────────────────────────────
p_dim = sum(p.numel() for p in model.parameters())
U     = torch.zeros(n_samples, p_dim)

# ──────────────────────────────────────────────────────────────────────────
# 5) TRAINING LOOP: pure SGD + IJ accumulation + decay
# ──────────────────────────────────────────────────────────────────────────
mse_loss   = nn.MSELoss(reduction="mean")
mse_hist   = []
grad_log   = []
ci_ij_hist = []
ci_ols_hist= []

# Precompute design matrix for closed‑form OLS:
# Φ = [1  x], shape=(n×2)
Phi1 = torch.cat([torch.ones(n_samples,1), x], dim=1)

for ep in range(n_epochs):
    # decayed step‑size
    mu = μ0 / (1.0 + decay * ep)

    # one pass of pure‐SGD
    perm = torch.randperm(n_samples)
    for k in perm:
        xi, yi = x[k:k+1], y[k:k+1]

        # (a) forward + loss
        model.zero_grad()
        y_pred = model(phi(xi))
        loss   = mse_loss(y_pred, yi)
        loss.backward()

        # flatten gradient
        g = flatten_grad(model).detach()

        # (b) SGD update θ ← θ − μ g
        with torch.no_grad():
            θv = torch.nn.utils.parameters_to_vector(model.parameters())
            θv -= mu * g
            torch.nn.utils.vector_to_parameters(θv, model.parameters())

        # (c) IJ update
        U = ij_update(U, k, g, mu)

        # (d) log grad norm for sample #track_idx
        if k == track_idx:
            grad_log.append(g.norm().item())

    # end‐of‐epoch diagnostics

    # 1. training MSE
    with torch.no_grad():
        all_pred = model(phi(x))
        mse_hist.append(((all_pred - y)**2).mean().item())

        # noise variance σ² ≈ E[(y−f)²]
        sigma2 = ((all_pred - y)**2).mean().item()

    # 2. IJ‐based CI at x* (Eqs 4.1–4.2)
    model.zero_grad()
    y0 = model(phi(torch.tensor([[x_star]]))).squeeze()
    y0.backward()
    g0     = flatten_grad(model).detach()       # ∇θf(x*)
    Uproj  = U @ g0                             # (n,)
    var_ij = sigma2 * (Uproj**2).sum().item()
    ci_ij_hist.append(1.96 * np.sqrt(var_ij))

    # 3. closed‐form OLS CI at x*:
    #    β̂=(ΦᵀΦ)⁻¹Φᵀy,  Var[ŷ]=σ²[1+φ*(ΦᵀΦ)⁻¹φ*]
    with torch.no_grad():
        β_ols  = torch.linalg.lstsq(Phi1, y).solution   # shape=(2,1)
        phi_star = torch.tensor([[1.0, x_star]])
        covB     = sigma2 * torch.linalg.inv(Phi1.T @ Phi1)
        var_ols  = (phi_star @ covB @ phi_star.T).item() + sigma2
        ci_ols_hist.append(1.96 * np.sqrt(var_ols))

# ──────────────────────────────────────────────────────────────────────────
# 6) PLOT RESULTS
# ──────────────────────────────────────────────────────────────────────────
epochs = np.arange(1, n_epochs+1)

plt.figure(figsize=(6,4))
plt.plot(epochs, mse_hist,    label="Train MSE",           color="black")
plt.plot(epochs, ci_ij_hist,  label="IJ CI half‑width",    color="C0")
plt.plot(epochs, ci_ols_hist, "--", label="OLS CI half‑width", color="C2")
plt.xlabel("Epoch"); plt.ylabel("Value")
plt.title("Train MSE & 95 % CI half‑widths")
plt.legend(loc="upper right"); plt.tight_layout()

plt.figure(figsize=(6,3))
plt.plot(grad_log, color="gray", alpha=0.7)
plt.xlabel(f"Visit count of sample #{track_idx}")
plt.ylabel(r"$\|\nabla_\theta L_{%d}\|$" % track_idx)
plt.title("Gradient‐norm trace"); plt.tight_layout()

plt.show()

# ──────────────────────────────────────────────────────────────────────────
# 7) SUMMARY
# ──────────────────────────────────────────────────────────────────────────
print(f"Final Train MSE         : {mse_hist[-1]:.4f}")
print(f"Final IJ CI half‑width  : {ci_ij_hist[-1]:.4f}")
print(f"Final OLS CI half‑width : {ci_ols_hist[-1]:.4f}")
