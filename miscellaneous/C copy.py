import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
np.random.seed(0)

n        = 200      # number of training points
noise    = 0.1      # σ of Gaussian noise ε
lr       = 1e-2     # SGD learning rate μ
epochs   = 200      # number of epochs
x_star   = 0.5      # test input for CI

# ──────────────────────────────────────────────────────────────────────────
# 1) SYNTHETIC DATA: y = 3x + 0.5 + ε
# ──────────────────────────────────────────────────────────────────────────
x = torch.randn(n,1)
y = 3.0*x + 0.5 + noise * torch.randn_like(x)

# Design matrix Φ = [1, x]
Phi    = torch.cat([torch.ones(n,1), x], dim=1)  # shape (n×2)
p_dim  = Phi.shape[1]                            # = 2 parameters (intercept+slope)

# ──────────────────────────────────────────────────────────────────────────
# 2) MODEL & SGD TRAINING
#    f(x)=w x + b implemented as nn.Linear(1→1)
# ──────────────────────────────────────────────────────────────────────────
model    = nn.Linear(1,1, bias=True)
optimizer= torch.optim.SGD(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

mse_history = []
for ep in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss   = mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()

    # record train MSE
    with torch.no_grad():
        mse_history.append( ((model(x)-y)**2).mean().item() )

# extract θ̂ = [b; w]
with torch.no_grad():
    b_hat = model.bias.item()
    w_hat = model.weight.item()
θ_hat = torch.tensor([b_hat, w_hat]).view(2,1)  # shape (2×1)

# ──────────────────────────────────────────────────────────────────────────
# 3) CLOSED‑FORM OLS CI AT x* (Eq (4.2) without Hessian)
#    β_OLS = (ΦᵀΦ)⁻¹ Φᵀ y
#    σ² = RSS/(n−p)
#    Var[ŷ(x*)] = σ² [1 + φ(x*)ᵀ (ΦᵀΦ)⁻¹ φ(x*)]
# ──────────────────────────────────────────────────────────────────────────
PhiTPhi = Phi.T @ Phi
PhiT_y  = Phi.T @ y
β_ols   = torch.linalg.inv(PhiTPhi) @ PhiT_y      # (2×1)
resid   = (y - Phi @ β_ols).pow(2).sum() / (n-p_dim)
σ2_ols  = resid.item()

phi_star = torch.tensor([1.0, x_star]).view(2,1)  # φ(x*)
covB     = σ2_ols * torch.linalg.inv(PhiTPhi)
var_ols  = (phi_star.T @ covB @ phi_star).item() + σ2_ols
std_ols  = np.sqrt(var_ols)
ci_half_ols = 1.96 * std_ols

# ──────────────────────────────────────────────────────────────────────────
# 4) ONE‑SHOT IJ CI AT x* (correct Hessian‑inverse)
#    ∇θL_i = -2 (y_i - φ_iᵀθ̂) φ_i,  H = 2 ΦᵀΦ
#    U_i^θ = -H⁻¹ ∇θL_i  ⇒  U_i^ŷ = φ(x*)ᵀ U_i^θ
#    Var_IJ = σ² Σ_i (U_i^ŷ)²
# ──────────────────────────────────────────────────────────────────────────
resid_vec = (y - Phi @ θ_hat).squeeze()             # (n,)
G = 2.0 * resid_vec.unsqueeze(1) * Phi              # (n×2)  rows = -∇θL_i

H     = 2.0 * PhiTPhi                              # (2×2)
H_inv = torch.linalg.inv(H)
U     = G @ H_inv                                   # (n×2), rows = U_i^θᵀ

U_y   = (U @ phi_star).squeeze()                    # (n,)
var_ij= σ2_ols * (U_y**2).sum().item()
std_ij= np.sqrt(var_ij)
ci_half_ij = 1.96 * std_ij

print(f"OLS  95% CI half‑width @ x*={x_star}: {ci_half_ols:.4f}")
print(f"IJ   95% CI half‑width @ x*={x_star}: {ci_half_ij:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 5) PLOT: MSE vs. OLS/IJ CI (all flat) over epochs
# ──────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(6,4))
plt.plot(mse_history,    label="Train MSE")
plt.hlines(ci_half_ols, 0, epochs-1, linestyles="--", colors="green",
           label="OLS CI half‑width")
plt.hlines(ci_half_ij,  0, epochs-1, linestyles="-.", colors="blue",
           label="IJ CI half‑width")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Train MSE vs. 95% CI half‑width (OLS & IJ)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
