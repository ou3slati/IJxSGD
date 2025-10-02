import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

"""
Phase-1 (extended) – scalable IJ verification on a quadratic problem
-------------------------------------------------------------------
• n = 10 000 training points with x ∈ [-3, 3]
• True function : y = 2 x² + 0.5 x + 1 + ε,  ε ~ N(0, 0.5²)
• Learner      : Linear layer on handcrafted features [x, x²]
• During SGD we accumulate Influence vectors U_i (10 000 × 3)
• Compare IJ 95 % band to closed-form OLS band (same features)
• Plots
  – left  : 1 k-point subsample + IJ / OLS ±95 % bands
  – right : scatter of |U_i^{ŷ}(x*=0)| vs x_i (+ binned mean ±std)
"""

# 1 ──────────────────────────────────  synthetic data
torch.manual_seed(0)
N = 10_000
x_train = torch.linspace(-3, 3, N).unsqueeze(1)
y_train = 2 * x_train**2 + 0.5 * x_train + 1 + 0.5 * torch.randn_like(x_train)

# 2 ──────────────────────────────────  quadratic model
class QuadModel(nn.Module):
    """f(x;θ) = θ0 + θ1 x + θ2 x² implemented as Linear([x,x²])."""
    def __init__(self):
        super().__init__()
        self.poly = nn.Linear(2, 1, bias=True)
    def forward(self, x):
        phi = torch.cat([x, x**2], dim=1)
        return self.poly(phi)

net = QuadModel()

# 3 ──────────────────────────────────  SGD + influence tracking
mu, epochs = 1e-3, 5
p = sum(p.numel() for p in net.parameters())
U = torch.zeros(N, p)

def flat_grad(model):
    return torch.nn.utils.parameters_to_vector([
        p.grad if p.grad is not None else torch.zeros_like(p)
        for p in model.parameters()])

for _ in range(epochs):
    for k in range(N):
        xk, yk = x_train[k:k+1], y_train[k:k+1]
        net.zero_grad(); loss = nn.functional.mse_loss(net(xk), yk)
        loss.backward(create_graph=True)
        g = flat_grad(net).detach()
        # SGD step
        with torch.no_grad():
            theta = torch.nn.utils.parameters_to_vector(net.parameters()) - mu * g
            torch.nn.utils.vector_to_parameters(theta, net.parameters())
        # IJ update (+1 for k, -1/N for others)
        scale = torch.full((N, 1), -1.0 / N); scale[k] = 1.0
        U -= mu * scale * g

# 4 ──────────────────────────────────  predictions + IJ band
x_grid = torch.linspace(-3, 3, 200).unsqueeze(1)
resid = net(x_train) - y_train
sigma2 = resid.pow(2).mean().item()

y_pred, std_ij = [], []
for xs in x_grid:
    net.zero_grad(); yp = net(xs.unsqueeze(0)); yp.backward()
    g_f = flat_grad(net).detach()
    var_ij = sigma2 * ((U @ g_f) ** 2).sum().item()
    y_pred.append(yp.item()); std_ij.append(var_ij ** 0.5)

y_pred = torch.tensor(y_pred); std_ij = torch.tensor(std_ij)
ci_lo, ci_hi = y_pred - 1.96 * std_ij, y_pred + 1.96 * std_ij

# 5 ──────────────────────────────────  closed-form OLS band
Phi  = torch.cat([x_train, x_train**2], 1)
Phi1 = torch.cat([torch.ones(N, 1), Phi], 1)
beta = torch.linalg.inv(Phi1.T @ Phi1) @ Phi1.T @ y_train
Phi_g = torch.cat([torch.ones_like(x_grid), x_grid, x_grid**2], 1)
y_ols = (Phi_g @ beta).squeeze()
res_ols = y_train - Phi1 @ beta
sigma2_ols = res_ols.pow(2).sum().item() / (N - 3)
CovB = sigma2_ols * torch.linalg.inv(Phi1.T @ Phi1)
var_ols = (Phi_g @ CovB @ Phi_g.T).diag() + sigma2_ols
std_ols = var_ols.sqrt(); ols_lo, ols_hi = y_ols - 1.96 * std_ols, y_ols + 1.96 * std_ols

# 6 ──────────────────────────────────  influence at x*=0 (full scatter)
x_star = torch.tensor([[0.0]])
net.zero_grad(); net(x_star).backward();
influence = (U @ flat_grad(net)).abs().detach().squeeze()

# Bin to reveal trend
n_bins = 50
bins = torch.linspace(-3, 3, n_bins + 1)
idx_bin = torch.bucketize(x_train.squeeze(), bins)
centers, mean_inf, std_inf = [], [], []
for b in range(1, n_bins + 1):
    mask = idx_bin == b
    if mask.any():
        centers.append(0.5 * (bins[b] + bins[b - 1]))
        vals = influence[mask]
        mean_inf.append(vals.mean().item())
        std_inf.append(vals.std().item())
centers = torch.tensor(centers); mean_inf = torch.tensor(mean_inf); std_inf = torch.tensor(std_inf)

# 7 ──────────────────────────────────  plots
fig = plt.figure(figsize=(12, 5)); gs = gridspec.GridSpec(1, 2, [3, 2])

# left: fit + CI
ax0 = fig.add_subplot(gs[0])
idx_sample = torch.randperm(N)[:1000]
ax0.scatter(x_train[idx_sample], y_train[idx_sample], s=8, alpha=0.3, label='data (1k)')
ax0.plot(x_grid, y_pred, color='tab:blue', label='IJ fit')
ax0.fill_between(x_grid.squeeze(), ci_lo, ci_hi, color='tab:blue', alpha=0.25, label='IJ 95%')
ax0.plot(x_grid, y_ols, color='tab:green', label='OLS fit')
ax0.fill_between(x_grid.squeeze(), ols_lo, ols_hi, color='tab:green', alpha=0.20, label='OLS 95%')
ax0.set_title('Quadratic fit ±95% CI'); ax0.set_xlabel('x'); ax0.set_ylabel('y'); ax0.legend()

# right: influence scatter + binned mean ± std
ax1 = fig.add_subplot(gs[1])
ax1.scatter(x_train, influence.numpy(), s=6, alpha=0.4, label='|U_i| (all)')
ax1.plot(centers, mean_inf, color='red', label='binned mean')
ax1.fill_between(centers, mean_inf - std_inf, mean_inf + std_inf, color='red', alpha=0.2, label='±1 std')
ax1.axvline(0, ls='--', c='gray'); ax1.set_yscale('log')
ax1.set_title('|U_i^{ŷ}(x*=0)| vs x_i'); ax1.set_xlabel('x_i'); ax1.set_ylabel('|U_i^{ŷ}| (log)'); ax1.legend()

plt.tight_layout(); plt.show()

print(f"Mean IJ std  : {std_ij.mean():.4f}")
print(f"Mean OLS std : {std_ols.mean():.4f}")