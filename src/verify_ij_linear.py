import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate synthetic linear data
torch.manual_seed(0)
n = 400
x_train = torch.randn(n, 1)
y_train = 3 * x_train + 0.5 + 0.1 * torch.randn_like(x_train)
w = torch.ones(n)

# 2. Define a small neural network model
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))

net = TinyNet()

# 3. Setup training and influence tracking
μ = 5e-4
n_epochs = 600
p = sum(p.numel() for p in net.parameters())
U = torch.zeros(n, p)

def flat_grad(model):
    return torch.nn.utils.parameters_to_vector([
        p.grad if p.grad is not None else torch.zeros_like(p)
        for p in model.parameters()
    ])

# 4. Train with SGD and accumulate influence matrix
for epoch in range(n_epochs):
    for k in range(n):
        xk = x_train[k:k+1]
        yk = y_train[k:k+1]

        net.zero_grad()
        loss = w[k] * nn.functional.mse_loss(net(xk), yk)
        loss.backward(create_graph=True)

        gθ = flat_grad(net)

        with torch.no_grad():
            θ = torch.nn.utils.parameters_to_vector(net.parameters())
            θ -= μ * gθ
            torch.nn.utils.vector_to_parameters(θ, net.parameters())

        scale = torch.full((n, 1), -1.0 / n)
        scale[k] = 1.0
        U -= μ * scale * gθ

# 5. Predict on grid and compute IJ uncertainty
x_grid = torch.linspace(x_train.min() - 0.5, x_train.max() + 0.5, 200).unsqueeze(1)
y_hat_list, std_list = [], []

for x_star in x_grid:
    net.zero_grad()
    y_star = net(x_star.unsqueeze(0))
    y_star.backward()
    grad_f = flat_grad(net)
    U_y = U @ grad_f
    var_hat = (U_y ** 2).sum().item()

    y_hat_list.append(y_star.item())
    std_list.append(var_hat ** 0.5)

y_hat = torch.tensor(y_hat_list)
std_hat = torch.tensor(std_list)
ci_lower = y_hat - 1.96 * std_hat
ci_upper = y_hat + 1.96 * std_hat

# 6. Compute OLS regression line + CI
X = torch.cat([torch.ones_like(x_train), x_train], dim=1)
θ_ols = torch.linalg.inv(X.T @ X) @ X.T @ y_train
resid = y_train - X @ θ_ols
sigma2 = (resid ** 2).sum() / (n - 2)
X_grid = torch.cat([torch.ones_like(x_grid), x_grid], dim=1)
ols_preds = X_grid @ θ_ols
ols_std = (sigma2 * (X_grid @ torch.linalg.inv(X.T @ X) @ X_grid.T).diag()).sqrt()
ols_lower = ols_preds.squeeze() - 1.96 * ols_std
ols_upper = ols_preds.squeeze() + 1.96 * ols_std

# 7. Plotting both confidence bands
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2])

# Left plot: Regression + IJ vs OLS CI
ax0 = plt.subplot(gs[0])
ax0.scatter(x_train.numpy(), y_train.numpy(), alpha=0.3, label="train pts")
ax0.plot(x_grid.numpy(), y_hat.numpy(), color='blue', label="TinyNet f(x)")
ax0.fill_between(x_grid.squeeze().numpy(), ci_lower.numpy(), ci_upper.numpy(),
                 alpha=0.3, label="IJ 95%", color='blue')
ax0.plot(x_grid.numpy(), ols_preds.numpy(), color='green', label="OLS line")
ax0.fill_between(x_grid.squeeze().numpy(), ols_lower.numpy(), ols_upper.numpy(),
                 alpha=0.2, color='green', label="OLS 95%")
ax0.set_title("TinyNet vs. OLS  ±95 % CI")
ax0.set_xlabel("x")
ax0.set_ylabel("y")
ax0.legend()

# Right plot: Influence magnitude at x*=0
x_focus = torch.tensor([[0.0]])
net.zero_grad()
net(x_focus).backward()
grad_focus = flat_grad(net)
U_focus = U @ grad_focus
U_focus_abs = U_focus.abs().detach().numpy()
top3 = np.argsort(-U_focus_abs)[:3]

ax1 = plt.subplot(gs[1])
ax1.scatter(x_train.numpy(), U_focus_abs, label="all pts", alpha=0.4)
ax1.scatter(x_train[top3].numpy(), U_focus_abs[top3], color='red', label="top-3")
ax1.axvline(x=0.0, linestyle='--', color='gray', label='x*=0')
ax1.set_title("|U_i^{ŷ}|  influence at x*=0")
ax1.set_xlabel("x_i")
ax1.set_ylabel("|U_i^{ŷ}|")
ax1.legend()

plt.tight_layout()
plt.show()
