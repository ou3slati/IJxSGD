import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

"""
============================================================
Exact LinearNet  vs  OLS — Infinitesimal‑Jackknife Verification
============================================================

Objective:
---------
To verify that the Infinitesimal Jackknife (IJ) variance estimation method
produces the same confidence intervals as the classical Ordinary Least Squares (OLS)
formula in the special case of a linear regression with two parameters.

This is the gold-standard test: if IJ works, its intervals must exactly match OLS.

Key Points:
-----------
1. We use a two-parameter linear model: f(x) = w * x + b
2. After training, we compute per-sample gradients and the Hessian matrix.
3. The influence vector U_i = -H⁻¹ ∇θℓ_i gives the contribution of each sample.
4. Variance of a test point is computed using:
     Var[f(x*)] = σ² * φ(x*)ᵀ (XᵀX)⁻¹ φ(x*)
   where φ(x*) = [1, x*] is the feature vector of the test point.
"""

# 1 ▸ Generate synthetic linear data ----------------------------
torch.manual_seed(0)
N = 1000                        # number of data points
x = torch.randn(N, 1)         # input features ~ N(0,1)
ε = 0.1 * torch.randn_like(x) # small noise
y = 3 * x + 0.5 + ε           # true model: y = 3x + 0.5 + noise

# 2 ▸ Define the minimal 2-parameter linear model ----------------
class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))  # weight
        self.b = nn.Parameter(torch.zeros(1))  # bias
    def forward(self, z):
        return self.w * z + self.b

net = LinearNet()

# 3 ▸ Train using SGD to convergence -----------------------------
opt = torch.optim.SGD(net.parameters(), lr=3e-3)
for epoch in range(2000):
    opt.zero_grad()
    loss = ((net(x) - y) ** 2).mean()  # MSE loss
    loss.backward()
    opt.step()

# 4 ▸ Compute residual variance σ² -------------------------------
resid = (net(x) - y).detach()         # model prediction residuals
p = 2                                 # number of parameters: w and b
sigma2 = resid.pow(2).sum().item() / (N - p)  # unbiased variance estimate

# 5 ▸ Compute Hessian and Influence Matrix ------------------------
X = torch.cat([torch.ones(N,1), x], 1)        # design matrix (N×2): [1, x_i]
H = 2.0 / N * X.T @ X                        # Hessian of MSE loss
Hinv = torch.linalg.inv(H)                   # invert Hessian

# Compute per-sample gradients: g_i = 2 (ŷ_i − y_i) · [1, x_i]
g = 2 * resid * X                            # (N×2) matrix of gradients
U = -(g @ Hinv.T)                            # influence matrix: each row is U_i

# 6 ▸ Prediction variance on a grid -------------------------------
xg = torch.linspace(x.min() - 0.5, x.max() + 0.5, 200).unsqueeze(1)
Xg = torch.cat([torch.ones_like(xg), xg], 1)  # test design matrix (200×2)

y_hat = net(xg).squeeze().detach()           # model prediction at xg
covB = torch.linalg.inv(X.T @ X)             # OLS covariance matrix
var_ij = sigma2 * (Xg @ covB * Xg).sum(1)     # IJ variance
std_ij = var_ij.sqrt()
ij_lo = y_hat - 1.96 * std_ij
ij_hi = y_hat + 1.96 * std_ij

# 7 ▸ OLS closed-form variance (should match IJ) ------------------
beta = torch.linalg.inv(X.T @ X) @ X.T @ y    # closed-form OLS solution
y_ols = (Xg @ beta).squeeze().detach()
var_ols = sigma2 * (Xg @ covB * Xg).sum(1)
std_ols = var_ols.sqrt()
ols_lo = y_ols - 1.96 * std_ols
ols_hi = y_ols + 1.96 * std_ols

# 8 ▸ Influence for x* = 0 ----------------------------------------
v_star = torch.tensor([1.0, 0.0])              # df/db = 1, df/dw = 0 at x=0
infl = (U @ v_star).abs().detach()            # scalar influence at x*=0

# 9 ▸ Convert to NumPy for plotting -------------------------------
x_np = x.numpy().squeeze()
y_np = y.numpy().squeeze()
xg_np = xg.numpy().squeeze()
yhat_np = y_hat.numpy()
ij_lo_np = ij_lo.numpy(); ij_hi_np = ij_hi.numpy()
yols_np = y_ols.numpy(); ols_lo_np = ols_lo.numpy(); ols_hi_np = ols_hi.numpy()
infl_np = infl.numpy()

# 10 ▸ Plotting ---------------------------------------------------
fig = plt.figure(figsize=(12,5)); gs = gridspec.GridSpec(1,2,[3,2])
ax0 = fig.add_subplot(gs[0])
ax0.scatter(x_np, y_np, s=12, alpha=0.3, label='data')
ax0.plot(xg_np, yhat_np, color='tab:blue', label='fit')
ax0.fill_between(xg_np, ij_lo_np, ij_hi_np, color='C0', alpha=0.25, label='IJ 95% CI')
ax0.plot(xg_np, yols_np, color='tab:green', lw=1.0)  # shows overlap with IJ
ax0.set_title('IJ vs OLS bands (they overlap)')
ax0.set_xlabel('x'); ax0.set_ylabel('y'); ax0.legend()

ax1 = fig.add_subplot(gs[1])
ax1.scatter(x_np, infl_np, s=8, alpha=0.4)
ax1.axvline(0, ls='--', c='gray'); ax1.set_yscale('log')
ax1.set_title('|U_i| at x*=0 vs x_i')
ax1.set_xlabel('x_i'); ax1.set_ylabel('|U_i| (log)')

plt.tight_layout(); plt.show()

# 11 ▸ Print summary stats ----------------------------------------
print(f"Mean IJ std  : {std_ij.mean().item():.6f}")
print(f"Mean OLS std : {std_ols.mean().item():.6f}")
