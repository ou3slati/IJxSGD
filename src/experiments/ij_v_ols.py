import torch, numpy as np, matplotlib.pyplot as plt
from data.synthetic import make_dataset
from models import linear, quadratic, utilities as mu
from influence.tracker import InfluenceTracker
from influence.utils import flatten_grad
from viz.plotting import plot_bands

# ----- config -----
kind   = "linear"          # "linear" | "quadratic"
n      = 800
epochs = 200
lr     = 1e-2
# -------------------

# 1. data + model --------------------------------------------------
x, y = make_dataset(kind, n, noise=0.1)
model = (linear.build() if kind == "linear" else quadratic.build())
tracker = InfluenceTracker(n, model)

# 2. training with IJ tracking ------------------------------------
criterion = torch.nn.MSELoss()
for ep in range(epochs):
    for i in torch.randperm(n):
        xi, yi = x[i:i+1], y[i:i+1]
        xi_phi = mu.phi(xi, kind)

        model.zero_grad()
        loss = criterion(model(xi_phi), yi)
        loss.backward()

        g = flatten_grad(model).detach()
        # SGD param update
        with torch.no_grad():
            theta = torch.nn.utils.parameters_to_vector(model.parameters())
            torch.nn.utils.vector_to_parameters(theta - lr * g, model.parameters())
        # IJ update
        tracker.sgd_update(i, g, lr)

# 3. σ² estimate ---------------------------------------------------
resid = model(mu.phi(x, kind)) - y
p_dim = sum(p.numel() for p in model.parameters())
sigma2 = resid.pow(2).sum().item() / (n - p_dim)

# 4. IJ variance on grid ------------------------------------------
grid = torch.linspace(-3, 3, 400).unsqueeze(1)
grid_phi = mu.phi(grid, kind)
ij_mean, ij_std = [], []

for x_star in grid_phi:
    model.zero_grad()
    y_star = model(x_star.unsqueeze(0))
    y_star.backward()
    g_f = flatten_grad(model).detach()
    var = sigma2 * (tracker.U @ g_f).pow(2).sum().item()
    ij_mean.append(y_star.item()); ij_std.append(np.sqrt(var))

ij_mean, ij_std = np.array(ij_mean), np.array(ij_std)

# 5. analytic OLS band --------------------------------------------
Phi   = mu.phi(x, kind)
Phi1  = torch.cat([torch.ones(n,1), Phi], 1)
beta  = torch.linalg.lstsq(Phi1, y).solution
Phi_g = torch.cat([torch.ones_like(grid), grid_phi], 1)
ols_mean = (Phi_g @ beta).squeeze().numpy()
resid_ols = y - Phi1 @ beta
sigma2_ols = resid_ols.pow(2).sum().item() / (n - p_dim)
CovB = sigma2_ols * torch.linalg.inv(Phi1.T @ Phi1)
var_g = (Phi_g @ CovB @ Phi_g.T).diag() + sigma2_ols
ols_std = var_g.sqrt().numpy()

print("Mean |IJ std – OLS std| =", np.mean(np.abs(ij_std - ols_std)))

# 6. plot ----------------------------------------------------------
plt.figure(figsize=(9,4))
plt.scatter(x, y, s=10, alpha=.3, label="train")
plot_bands(grid, ij_mean, ij_std, "IJ", "tab:blue")
plot_bands(grid, ols_mean, ols_std, "OLS", "tab:green")
plt.title(f"{kind.capitalize()} — IJ vs OLS (n={n})")
plt.legend(); plt.tight_layout(); plt.show()
