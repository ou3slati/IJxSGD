import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

"""
Phase‑1 Extended: Scalable IJ for Quadratic Regression
----------------------------------------------------
Project scope:
  • We aim to quantify predictive uncertainty in deep learning via the Infinitesimal Jackknife (IJ).
  • Here, we validate IJ on a convex quadratic regression (toy non-linearity) with large n (10k).
  • We compare IJ-derived 95% confidence bands to classical closed-form OLS bands on the same features.
  • We visualize per-point influences U_i on a test prediction yˆ(x*), illustrating sensitivity of predictions.

Key equations:
  Eq (1):  R(θ) = Σ_i w_i L(y_i, f(x_i, θ))
  Eq (2.3): θ ← θ - μ ∇_θ L_k   (SGD update)
  Eq (3.2): U_i ← U_i - μ [1(i=k) - 1/n] ∇_θ L_k   (IJ update)
  Eq (4.1): U_i^{ŷ} = (∇_θ f(x*, θ))ᵀ U_i
  Eq (4.2): Var_IJ[ŷ(x*)] ≈ σ² Σ_i (U_i^{ŷ})²

This code is organized in sections 1–7 with in-depth comments.
"""

# 1) Generate synthetic quadratic data (n=10k)
#    y = 2 x² + 0.5 x + 1 + ε,   ε ∼ N(0, 0.5²)
#    Using a large dataset tests IJ scalability (we accumulate an n×p matrix U).
torch.manual_seed(0)
N = 1000
x_train = torch.linspace(-3, 3, N).unsqueeze(1)              # feature x ∈ ℝ
noise    = 0.5 * torch.randn_like(x_train)                   # Gaussian noise
# True labels y, ensures residual variance ~0.25
y_train  = 2 * x_train**2 + 0.5 * x_train + 1 + noise
#make variance 1

# 2) Define model: one linear layer on handcrafted features [x, x²]
#    f(x; θ) = θ₀ + θ₁ x + θ₂ x², model is convex in parameters
class QuadModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear transformation on 2D polynomial features
        self.poly = nn.Linear(2, 1, bias=True)
    def forward(self, x):
        # Build feature vector φ(x) = [x, x²]
        phi = torch.cat([x, x**2], dim=1)   # shape (batch,2)
        return self.poly(phi)                # linear output

net = QuadModel()

# 3) Train with pure SGD and track influence vectors U (n×p)
#    We accumulate per-sample sensitivities U_i = ∂θ/∂w_i (Eq 3.2)
mu, epochs = 1e-4, 500
p = sum(param.numel() for param in net.parameters())         # p = 3 parameters
U = torch.zeros(N, p)                                        # initialize U at zero

# Utility: flatten parameter gradients ∇_θ L into a vector g
def flat_grad(model):
    grads = []
    for param in model.parameters():
        # if param.grad is None (no grad), use zeros
        grads.append(param.grad if param.grad is not None else torch.zeros_like(param))
    return torch.nn.utils.parameters_to_vector(grads)

# Run epochs of SGD (Eq 2.3) and IJ update (Eq 3.2)
for _ in range(epochs):
    for k in range(N):
        xk, yk = x_train[k:k+1], y_train[k:k+1]
        # ---- Forward & loss L_k (MSE) ----
        net.zero_grad()
        loss = nn.functional.mse_loss(net(xk), yk)  # w_k = 1
        # ---- Backpropagate to get ∇_θ L_k ----
        loss.backward(create_graph=True)             # keep graph for IJ
        g = flat_grad(net).detach()                 # detach so U updates don't build graph
        # ---- SGD step: θ ← θ - μ g (Eq 2.3) ----
        with torch.no_grad():
            theta_vec = torch.nn.utils.parameters_to_vector(net.parameters())
            theta_vec -= mu * g
            torch.nn.utils.vector_to_parameters(theta_vec, net.parameters())
        # ---- IJ update: U_i ← U_i - μ [1(i=k)-1/N] g (Eq 3.2) ----
        scale = torch.full((N,1), -1.0/N)
        scale[k] = 1.0
        U -= mu * scale * g

# 4) Compute predictions on a test grid and IJ-based variance bands
#    Eq 4.1: U_i^ŷ = (∇_θ f(x*) )ᵀ U_i
#    Eq 4.2: Var_IJ[ŷ] ≈ σ² Σ_i (U_i^ŷ)²
x_grid = torch.linspace(-3, 3, 200).unsqueeze(1)
# Estimate noise level σ² from training residuals
residuals = net(x_train) - y_train
#sigma2 = residuals.pow(2).mean().item()

y_pred, std_ij = [], []
for xs in x_grid:
    # compute prediction and gradient ∇_θ f(xs)
    net.zero_grad()
    y_star = net(xs.unsqueeze(0))
    y_star.backward()
    grad_f = flat_grad(net).detach()
    # project influence vectors into prediction space
    U_proj = U @ grad_f                      # shape (N,)
    # IJ variance (scaled by noise) -> std
    var_ij = (U_proj**2).sum().item()
    y_pred.append(y_star.item())
    std_ij.append(var_ij**0.5)

# Convert lists into tensors for plotting
y_pred = torch.tensor(y_pred)
std_ij  = torch.tensor(std_ij)
ci_lo, ci_hi = y_pred - 1.96 * std_ij, y_pred + 1.96 * std_ij

# 5) Closed-form OLS on same polynomial features for ground-truth CI
#    θ̂_OLS = (ΦᵀΦ)^{-1} Φᵀ y,   Var[ŷ] = σ²[1 + φ(x*)ᵀ (ΦᵀΦ)^{-1} φ(x*)]
Phi = torch.cat([x_train, x_train**2], dim=1)          # shape (N,2)
Phi1 = torch.cat([torch.ones(N,1), Phi], dim=1)        # add intercept → (N,3)
beta_ols = torch.linalg.inv(Phi1.T @ Phi1) @ Phi1.T @ y_train
# predict on grid
Phi_g = torch.cat([torch.ones_like(x_grid), x_grid, x_grid**2], dim=1)
y_ols = (Phi_g @ beta_ols).squeeze()
# residual variance
res_ols = y_train - Phi1 @ beta_ols
sigma2_ols = res_ols.pow(2).sum().item() / (N - 3)
# covariance of β̂
Cov_beta = sigma2_ols * torch.linalg.inv(Phi1.T @ Phi1)
# predictive variance for ŷ over grid
var_ols  = (Phi_g @ Cov_beta @ Phi_g.T).diag() + sigma2_ols
std_ols  = var_ols.sqrt()
ols_lo, ols_hi = y_ols - 1.96 * std_ols, y_ols + 1.96 * std_ols

# 6) Influence scatter at x* = 0 with binned trend
x_star = torch.tensor([[0.0]])
net.zero_grad(); net(x_star).backward()
infl = (U @ flat_grad(net)).abs().detach().squeeze()
# bin x_i to reveal mean/SD trend of influence vs distance
n_bins = 50
bins   = torch.linspace(-3, 3, n_bins + 1)
bin_idx = torch.bucketize(x_train.squeeze(), bins)
centers, mean_inf, std_inf = [], [], []
for b in range(1, n_bins + 1):
    mask = bin_idx == b
    if mask.any():
        centers.append(0.5 * (bins[b] + bins[b-1]))
        vals = infl[mask]
        mean_inf.append(vals.mean().item())
        std_inf.append(vals.std().item())
centers  = np.array(centers)
mean_inf = np.array(mean_inf)
std_inf  = np.array(std_inf)

# 7) Plot everything
fig = plt.figure(figsize=(12,5))
gs  = gridspec.GridSpec(1,2,[3,2])

# 7a: fit curves + uncertainty bands
ax0 = fig.add_subplot(gs[0])
# subsample 1000 points for scatter clarity
tmp = torch.randperm(N)[:1000]
ax0.scatter(x_train[tmp], y_train[tmp], s=8, alpha=0.3, label='data (1k)')
ax0.plot(x_grid, y_pred, color='tab:blue', label='IJ fit')
ax0.fill_between(x_grid.squeeze(), ci_lo, ci_hi, color='tab:blue', alpha=0.25, label='IJ 95% CI')
ax0.plot(x_grid, y_ols, color='tab:green', label='OLS fit')
ax0.fill_between(x_grid.squeeze(), ols_lo, ols_hi, color='tab:green', alpha=0.2, label='OLS 95% CI')
ax0.set_title('Quadratic regression ±95% CI (N=10k)')
ax0.set_xlabel('x'); ax0.set_ylabel('y'); ax0.legend()

# 7b: influence vs x_i scatter + binned trend
ax1 = fig.add_subplot(gs[1])
ax1.scatter(x_train, infl, s=6, alpha=0.4, label='|U_i| (all)')
ax1.plot(centers, mean_inf, color='red', label='binned mean')
ax1.fill_between(centers, mean_inf-std_inf, mean_inf+std_inf, color='red', alpha=0.2, label='±1 SD')
ax1.axvline(0, ls='--', c='gray', label='x*=0')
ax1.set_yscale('log')
ax1.set_title('Influence magnitude vs x_i at x*=0')
ax1.set_xlabel('x_i'); ax1.set_ylabel('|U_i| (log scale)'); ax1.legend()

plt.tight_layout()
plt.show()

# print summary statistics
print(f"Mean IJ std  : {std_ij.mean():.4f}")
print(f"Mean OLS std : {std_ols.mean():.4f}")
