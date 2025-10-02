import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1) Synthetic Linear Dataset
#    y = 3x + 0.5 + Gaussian noise
# ──────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0)
n_samples = 200
x_train = torch.randn(n_samples, 1)
noise   = 0.1 * torch.randn_like(x_train)
y_train = 3 * x_train + 0.5 + noise

# ──────────────────────────────────────────────────────────────────────────────
# 2) Model: Pure linear (1 hidden neuron with identity activation)
# ──────────────────────────────────────────────────────────────────────────────
#    f(x) = w*x + b
model = nn.Sequential(nn.Linear(1,1, bias=True))

# Shortcut to get all parameters as a single vector
def get_theta_vec():
    return torch.nn.utils.parameters_to_vector(model.parameters())

# Flattened gradient ∇θ L
def flat_grad():
    grads = []
    for p in model.parameters():
        grads.append(p.grad if p.grad is not None else torch.zeros_like(p))
    return torch.nn.utils.parameters_to_vector(grads)

# ──────────────────────────────────────────────────────────────────────────────
# 3) OLS closed‑form for comparison
#    θ̂ = (XᵀX)⁻¹ Xᵀy,  CI: σ² x*ᵀ (XᵀX)⁻¹ x*  (±1.96·√Var)
# ──────────────────────────────────────────────────────────────────────────────
# Build design matrix [1, x]
X = torch.cat([torch.ones(n_samples,1), x_train], dim=1)
# Precompute inverse Gram
XtX_inv = torch.linalg.inv(X.T @ X)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Training + IJ accumulation & diagnostics
# ──────────────────────────────────────────────────────────────────────────────
lr        = 5e-3
n_epochs  = 200

p         = sum(p.numel() for p in model.parameters())   # should be 2
# Influence matrix U (n_samples × p)
U = torch.zeros(n_samples, p)

# For diagnostics:
ci_ij_hist   = []   # IJ CI half-width @ x*=0
ci_ols_hist  = []   # OLS CI half-width @ x*=0
mse_hist     = []   # training MSE
# To trace per-step gradients:
grad_history = []   # list of (epoch*n + k, k, grad_norm) tuples

for epoch in range(n_epochs):
    # pure SGD passes
    epoch_mse = 0.0
    for k in range(n_samples):
        xk, yk = x_train[k:k+1], y_train[k:k+1]

        # --- Forward + loss ---
        model.zero_grad()
        y_pred = model(xk)
        loss   = nn.functional.mse_loss(y_pred, yk)
        loss.backward(create_graph=True)

        # --- Extract gradient and record its norm ---
        g = flat_grad().detach()
        grad_history.append((epoch, k, g.norm().item()))

        # --- SGD parameter update (Eq 2.3) ---
        with torch.no_grad():
            θ = get_theta_vec()
            θ -= lr * g
            torch.nn.utils.vector_to_parameters(θ, model.parameters())

        # --- IJ update (Eq 3.2): U_i ← U_i − μ [1(i=k) − 1/n] ∇θL_k ---
        # Build scale vector: +1 at k, −1/n everywhere else
        scale = torch.full((n_samples,1), -1.0/n_samples)
        scale[k] = 1.0
        U -= lr * scale * g  # broadcast along p dims

        epoch_mse += loss.item()

    # average MSE over epoch
    mse_hist.append(epoch_mse / n_samples)

    # --- Compute IJ CI at x* = 0 (Eq 4.2) ---
    # 1) forward‐gradient ∇θ f(x*), here f(x)=w x + b so grad is [1, x*]
    x_star = torch.tensor([[0.0]])
    model.zero_grad()
    y_star = model(x_star)
    y_star.backward()
    grad_f = flat_grad().detach()     # shape (p,)

    # 2) projected influences U_y = U @ grad_f
    U_y = U @ grad_f                  # (n_samples,)

    # 3) noise estimate σ² from residuals
    with torch.no_grad():
        resid = model(x_train) - y_train
        sigma2 = resid.pow(2).mean().item()

    var_ij = sigma2 * (U_y**2).sum().item()
    ci_ij_hist.append(1.96 * np.sqrt(var_ij))

    # --- Compute OLS CI at x* = 0 ---
    # θ̂ = (b,w),  var(ŷ)=σ²·x*ᵀ(XᵀX)⁻¹ x* + σ²  (we include noise-term for prediction)
    beta_ols = XtX_inv @ (X.T @ y_train)    # (2×1)
    # residual variance
    res_ols  = y_train - X @ beta_ols
    sigma2_ols = (res_ols.pow(2).sum()/(n_samples-2)).item()
    # design vector for x*=0
    x0_vec = torch.tensor([1.0, 0.0]).unsqueeze(1)   # shape (2×1)
    pred_var = sigma2_ols * (x0_vec.T @ XtX_inv @ x0_vec).item() + sigma2_ols
    ci_ols_hist.append(1.96 * np.sqrt(pred_var))

# ──────────────────────────────────────────────────────────────────────────────
# 5) Plots
# ──────────────────────────────────────────────────────────────────────────────
epochs = np.arange(1, n_epochs+1)

plt.figure(figsize=(10,4))
plt.plot(epochs, ci_ij_hist,  label="IJ CI half‑width @ x*=0")
plt.plot(epochs, ci_ols_hist, label="OLS CI half‑width @ x*=0", linestyle="--")
plt.plot(epochs, mse_hist,    label="Train MSE", alpha=0.6)
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("IJ vs OLS CI and MSE Over Training")
plt.legend()
plt.tight_layout()
plt.show()

# Pick one data index to inspect its per‑step gradient norms
inspect_idx = 0
times, grads = [], []
for (ep, k, gn) in grad_history:
    if k == inspect_idx:
        times.append(ep + k/n_samples)
        grads.append(gn)

plt.figure(figsize=(8,3))
plt.scatter(times, grads, s=4, alpha=0.5)
plt.xlabel("Epoch (fractional)")
plt.ylabel(f"||∇θ L_k|| for k={inspect_idx}")
plt.title(f"Gradient Norms for Sample #{inspect_idx} Over Visits")
plt.tight_layout()
plt.show()
