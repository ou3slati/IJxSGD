import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION & RANDOM SEED
# ──────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)  # → make results reproducible

# Choose “linear” (y=3x+0.5+ε) or “quadratic” (y=2x²+0.5x+1+ε)
kind        = "linear"  
n_samples   = 800      # Number of training points
noise_level = 0.1      # Gaussian noise σ on y
lr          = 1e-2     # SGD step size μ
n_epochs    = 10      # How many passes over the data
track_idx   = 42       # Which sample’s gradient‖·‖ to log
x_star      = 0.0      # Test input where we track CI width

# ──────────────────────────────────────────────────────────────────────────
# 1) SYNTHETIC DATA — build (x_i, y_i)
#    y = f_true(x) + ε,   ε ∼ N(0, σ²)
# ──────────────────────────────────────────────────────────────────────────
if kind == "linear":
    # linear ground truth: y = 3x + 0.5 + noise
    x = torch.randn(n_samples,1)
    y = 3.0*x + 0.5 + noise_level*torch.randn_like(x)
elif kind == "quadratic":
    # quadratic ground truth: y = 2x² + 0.5x + 1 + noise
    x = torch.linspace(-3,3,n_samples).unsqueeze(1)
    y = 2.0*x**2 + 0.5*x + 1.0 + noise_level*torch.randn_like(x)
else:
    raise ValueError("Unknown kind")

# Feature map φ(x): identity for linear, [x, x²] for quadratic
def phi(x):
    if kind=="linear":
        return x
    else:
        return torch.cat([x, x**2], dim=1)

# ──────────────────────────────────────────────────────────────────────────
# 2) MODEL DEFINITION
#    We fit either
#      • LinearModel: θ0 + θ1·x
#      • QuadraticModel: θ0 + θ1·x + θ2·x²
# ──────────────────────────────────────────────────────────────────────────
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(1→1) learns weight W and bias b
        self.lin = nn.Linear(1,1, bias=True)
    def forward(self, x):
        return self.lin(x)

class QuadraticModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(2→1) learns [θ1,θ2] and θ0
        self.poly = nn.Linear(2,1, bias=True)
    def forward(self, x):
        return self.poly(x)

# instantiate model
model = LinearModel() if kind=="linear" else QuadraticModel()

# ──────────────────────────────────────────────────────────────────────────
# 3) UTILITY: FLATTEN GRADIENTS
#    After loss.backward(), each param.grad holds ∂L/∂param.
#    We need one long vector ∇_θ L to plug into Eq 3.2.
# ──────────────────────────────────────────────────────────────────────────
def flatten_grad(model):
    grads = []
    for p in model.parameters():
        grads.append(p.grad if p.grad is not None else torch.zeros_like(p))
    # concatenate into a single vector of length p_dim
    return torch.nn.utils.parameters_to_vector(grads)

# ──────────────────────────────────────────────────────────────────────────
# 4) INFLUENCE MATRIX INITIALIZATION
#    U is our IJ accumulator: U_i ≈ ∂θ/∂w_i at w=1 (Eq 3.1).
#    It has shape (n_samples × p_dim).
# ──────────────────────────────────────────────────────────────────────────
p_dim = sum(p.numel() for p in model.parameters())
U     = torch.zeros(n_samples, p_dim)

def ij_update(U, k, grad_vec, μ):
    """
    Eq (3.2):
      U_i ← U_i - μ [1(i=k) - 1/n] ∇_θ L_k
    Builds a scale vector of +1 @ k, -1/n elsewhere, then applies it.
    """
    # +1 for the visited sample k, -1/n for all others
    scale = torch.full((n_samples,1), -1.0/n_samples)
    scale[k] += 1.0
    # broadcast-multiply then subtract
    return U - μ * scale * grad_vec

# ──────────────────────────────────────────────────────────────────────────
# 5) TRAINING LOOP
#    Each epoch:
#      • Shuffle samples
#      • For each k:
#          – Compute loss L_k = (f(x_k)-y_k)²
#          – loss.backward() → ∇_θ L_k
#          – SGD step (Eq 2.3): θ←θ - μ ∇_θ L_k
#          – IJ update on U      (Eq 3.2)
#          – Log ∥grad∥ for sample #track_idx
#    At epoch end:
#      • Compute train MSE
#      • Estimate σ² from residuals
#      • CI half‑width @ x* via IJ  (Eqs 4.1–4.2)
#      • CI half‑width @ x* via OLS (closed form)
# ──────────────────────────────────────────────────────────────────────────
criterion   = nn.MSELoss(reduction="mean")
ci_half     = []
ols_half    = []
mse_history = []
grad_log    = []

# Precompute design Φ₁ = [1, x, x²...] for OLS
Phi   = phi(x)
Phi1  = torch.cat([torch.ones(n_samples,1), Phi], dim=1)  # add intercept

for ep in range(n_epochs):
    epoch_loss = 0.0

    # 5.1) SGD + IJ passes
    for k in torch.randperm(n_samples):
        xi, yi = x[k:k+1], y[k:k+1]

        # forward + loss
        model.zero_grad()
        y_pred = model(phi(xi))
        loss   = criterion(y_pred, yi)
        epoch_loss += loss.item()

        # backward → ∇_θ L_k
        loss.backward()
        g = flatten_grad(model).detach()  # detach so U stays graph‑free

        # (a) SGD update   θ ← θ - μ g
        with torch.no_grad():
            θ_vec = torch.nn.utils.parameters_to_vector(model.parameters())
            θ_vec -= lr * g
            torch.nn.utils.vector_to_parameters(θ_vec, model.parameters())

        # (b) IJ update   U ← ij_update(U, k, g, μ)
        U = ij_update(U, k, g, lr)

        # (c) log gradient norm for the tracked sample
        if k == track_idx:
            grad_log.append(g.norm().item())

    # 5.2) end‑of‑epoch diagnostics

    # (i) training MSE on all pairs
    with torch.no_grad():
        y_all = model(phi(x))
        mse_history.append( ((y_all - y)**2).mean().item() )

    # (ii) noise variance σ² ≈ mean residual²
    resid  = (y_all - y).squeeze()
    sigma2 = resid.pow(2).mean().item()

    # (iii) IJ CI @ x_star (Eq 4.1–4.2):
    #        U_i^{ŷ} = (∇_θ f(x*) )ᵀ U_i    var = σ² Σ U_i^{ŷ}²
    model.zero_grad()
    y0 = model(phi(torch.tensor([[x_star]])))
    y0.backward()
    g0     = flatten_grad(model).detach()
    U_proj = U @ g0               # shape (n_samples,)
    var_ij = sigma2 * (U_proj**2).sum().item()
    ci_half.append(1.96 * np.sqrt(var_ij))

    # (iv) OLS CI @ x_star (closed‑form)
    with torch.no_grad():
        β     = torch.linalg.lstsq(Phi1, y).solution   # shape ((d+1)×1)
        if kind=="linear":
            φ1 = torch.tensor([[1.0, x_star]])
        else:
            φ1 = torch.tensor([[1.0, x_star, x_star**2]])
        CovB  = sigma2 * torch.linalg.inv(Phi1.T @ Phi1)
        var_o = (φ1 @ CovB @ φ1.T).item() + sigma2
        ols_half.append(1.96 * np.sqrt(var_o))

# ──────────────────────────────────────────────────────────────────────────
# 6) VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────
# (a) compare IJ‑vs‑OLS half‑width & training MSE
plt.figure(figsize=(6,4))
plt.plot(ci_half,     label="IJ CI half‐width @ x*")
plt.plot(ols_half, '--', label="OLS CI half‐width @ x*")
plt.plot(mse_history,   label="Train MSE")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title(f"IJ vs OLS CI & MSE over training ({kind})")
plt.legend()
plt.tight_layout()

# (b) trace of ∥∇_θL_{track_idx}∥ over visits
plt.figure(figsize=(6,3))
plt.plot(grad_log, alpha=0.7)
plt.xlabel("Visit count for sample #{track_idx}")
plt.ylabel("‖∇θL_{track_idx}‖")
plt.title(f"Gradient norm trace (sample {track_idx})")
plt.tight_layout()

plt.show()

# ──────────────────────────────────────────────────────────────────────────
# 7) SUMMARY PRINT‑OUT
# ──────────────────────────────────────────────────────────────────────────
print(f"Final IJ CI @ x*={x_star}: {ci_half[-1]:.4f}")
print(f"Final OLS CI @ x*={x_star}: {ols_half[-1]:.4f}")
print(f"Mean train MSE (last 10 epochs): {np.mean(mse_history[-10:]):.4f}")
