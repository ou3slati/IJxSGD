# Updated IJ Uncertainty Script with Dynamic Singular-Value Plot

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 0) CONFIG & SEED
torch.manual_seed(0)

kind        = "linear"
n_samples   = 800
noise_level = 0.1
lr          = 1e-2
n_epochs    = 200
track_idx   = 42
x_star      = 0.0

# 1) Synthetic Data
if kind == "linear":
    x = torch.randn(n_samples,1)
    y = 3.0*x + 0.5 + noise_level*torch.randn_like(x)
elif kind == "quadratic":
    x = torch.linspace(-3,3,n_samples).unsqueeze(1)
    y = 2.0*x**2 + 0.5*x + 1.0 + noise_level*torch.randn_like(x)
else:
    raise ValueError

def phi(x):
    return x if kind=="linear" else torch.cat([x, x**2], dim=1)

# 2) Model
class LinearModel(nn.Module):
    def __init__(self): super().__init__(); self.lin = nn.Linear(1,1, bias=True)
    def forward(self, x): return self.lin(x)
class QuadraticModel(nn.Module):
    def __init__(self): super().__init__(); self.poly = nn.Linear(2,1, bias=True)
    def forward(self, x): return self.poly(x)

model = LinearModel() if kind=="linear" else QuadraticModel()

# 3) Utility
def flatten_grad(m):
    return torch.nn.utils.parameters_to_vector(
        [p.grad if p.grad is not None else torch.zeros_like(p) for p in m.parameters()])

# 4) IJ Influence
p_dim = sum(p.numel() for p in model.parameters())
U     = torch.zeros(n_samples, p_dim)
def ij_update(U, k, g, μ):
    scale = torch.full((n_samples,1), -1.0/n_samples)
    scale[k] += 1.0
    return U - μ * scale * g

# 5) Training + Logs
criterion      = nn.MSELoss(reduction="mean")
ci_half, ols_half, mse_history, grad_log = [], [], [], []
u_norm_log, sigma_ratio_log = [], []
Phi   = phi(x)
Phi1  = torch.cat([torch.ones(n_samples,1), Phi], dim=1)

for ep in range(n_epochs):
    for k in torch.randperm(n_samples):
        xi, yi = x[k:k+1], y[k:k+1]
        model.zero_grad()
        loss = criterion(model(phi(xi)), yi)
        loss.backward()
        g = flatten_grad(model).detach()

        # SGD step
        with torch.no_grad():
            θ = torch.nn.utils.parameters_to_vector(model.parameters())
            θ -= lr * g
            torch.nn.utils.vector_to_parameters(θ, model.parameters())

        # IJ update
        U = ij_update(U, k, g, lr)

        if k == track_idx:
            grad_log.append(g.norm().item())
            u_norm_log.append(U[k].norm().item())

    # End‑epoch diagnostics
    with torch.no_grad():
        y_all = model(phi(x))
        mse_history.append(((y_all - y)**2).mean().item())
    resid = (y_all - y).squeeze()
    sigma2 = resid.pow(2).mean().item()
    sigma_ratio_log.append(sigma2 / (noise_level**2))

    # IJ CI
    model.zero_grad()
    y0 = model(phi(torch.tensor([[x_star]])))
    y0.backward()
    g0 = flatten_grad(model).detach()
    var_ij = sigma2 * ( (U @ g0)**2 ).sum().item()
    ci_half.append(1.96 * np.sqrt(var_ij))

    # OLS CI
    with torch.no_grad():
        β   = torch.linalg.lstsq(Phi1, y).solution
        φ1  = torch.tensor([[1.0, x_star]]) if kind=="linear" else torch.tensor([[1.0, x_star, x_star**2]])
        Cov = sigma2 * torch.linalg.inv(Phi1.T @ Phi1)
        var_o = (φ1 @ Cov @ φ1.T).item() + sigma2
        ols_half.append(1.96 * np.sqrt(var_o))

# 6) SVD Diagnostics
with torch.no_grad():
    _, S, _ = torch.linalg.svd(U)
    top_svs = S.cpu().numpy()
    n_svs = len(top_svs)

# 7) Visualization

# (a) IJ vs OLS CI & MSE
plt.figure(figsize=(6,4))
plt.plot(ci_half, label="IJ CI half-width")
plt.plot(ols_half, '--', label="OLS CI half-width")
plt.plot(mse_history, label="Train MSE")
plt.xlabel("Epoch"); plt.ylabel("Value")
plt.title(f"IJ vs OLS CI & MSE ({kind})")
plt.legend(); plt.tight_layout()

# (b) Gradient norm trace
plt.figure(figsize=(6,3))
plt.plot(grad_log, alpha=0.7)
plt.xlabel(f"Visit count (sample #{track_idx})")
plt.ylabel("Gradient norm")
plt.title("Gradient norm trace")
plt.tight_layout()

# (c) U-vector norm trace
plt.figure(figsize=(6,3))
plt.plot(u_norm_log, alpha=0.7)
plt.xlabel(f"Visit count (sample #{track_idx})")
plt.ylabel("U-vector norm")
plt.title("Influence U-norm trace")
plt.tight_layout()

# (d) Noise variance ratio
"""plt.figure(figsize=(6,3))
plt.plot(sigma_ratio_log)
plt.axhline(1.0, linestyle='--', label="True σ² ratio")
plt.xlabel("Epoch"); plt.ylabel("Estimated σ² / True σ²")
plt.title("Noise variance estimation")
plt.legend(); plt.tight_layout()"""

# (e) Top singular values
"""plt.figure(figsize=(6,4))
plt.bar(np.arange(1, n_svs+1), top_svs)
plt.xlabel("Singular index"); plt.ylabel("Singular value")
plt.title("Top singular values of U")
plt.tight_layout()"""

plt.show()

# 8) Summary
print("Top singular values of U:", top_svs)
print(f"Final IJ CI half-width: {ci_half[-1]:.4f}")
print(f"Final OLS CI half-width: {ols_half[-1]:.4f}")
print(f"Mean Train MSE (last 10 epochs): {np.mean(mse_history[-10:]):.4f}")


# --- Summary of Results (for discussion with professor) ---

# Fig 1: IJ CI bands grow linearly while OLS CI stays flat — even though MSE stabilizes.
# ⇒ Suggests model is learning correctly, but IJ variance is inflating due to accumulation.

# Fig 2: Gradient norms for sample #42 are small and stable.
# ⇒ Confirms SGD is behaving normally — not driving instability directly.

# Fig 3: U-vector norm for sample #42 grows steadily without cancellation.
# ⇒ Implies IJ is over-accumulating influence with each revisit; no decay or correction.

# Diagnosis: IJ CI growth is not from model instability, but from unbounded accumulation in U.
# Proposal: Add decay, reset, or delayed start to U updates to prevent runaway variance.