import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

"""
Tracks how IJ bands evolve during training on a non-linear regression problem: y = sin(3x) + ε
This implementation uses old-style IJ computation via the influence matrix U
"""

# 1 ▸ Generate training data: y = sin(3x) + noise
N = 300
x_train = torch.linspace(-2*np.pi, 2*np.pi, N).unsqueeze(1)
y_true = torch.sin(3 * x_train)
y_train = y_true + 0.1 * torch.randn_like(x_train)

# 2 ▸ Define the MLP
class SineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# 3 ▸ Set up training parameters
mu = 5e-3
epochs = 101
checkpoints = [0, 50, 100]  # epoch milestones to record IJ bands

# 4 ▸ Create grid for prediction and initialize storage
x_grid = torch.linspace(-2*np.pi, 2*np.pi, 300).unsqueeze(1)
y_sin = torch.sin(3 * x_grid).squeeze()
bands = {}

# 5 ▸ Helper to flatten gradients
def flat_grad(model):
    return torch.nn.utils.parameters_to_vector([
        p.grad if p.grad is not None else torch.zeros_like(p)
        for p in model.parameters()])

# 6 ▸ Begin training and record IJ bands at checkpoints
net = SineNet()
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    # Reset influence matrix and track weights
    net.train()
    p = sum(p.numel() for p in net.parameters())
    U = torch.zeros(N, p)

    for k in range(N):
        xk, yk = x_train[k:k+1], y_train[k:k+1]
        net.zero_grad()
        loss = loss_fn(net(xk), yk)
        loss.backward(create_graph=True)

        g = flat_grad(net).detach()
        scale = torch.full((N, 1), -1.0/N)
        scale[k] = 1.0
        U -= mu * scale * g

        # SGD update
        with torch.no_grad():
            theta = torch.nn.utils.parameters_to_vector(net.parameters())
            theta -= mu * g
            torch.nn.utils.vector_to_parameters(theta, net.parameters())

    if epoch in checkpoints:
        # Compute residual variance (σ²)
        with torch.no_grad():
            y_pred = net(x_train)
            sigma2 = ((y_pred - y_train)**2).mean().item()

        # Compute predictions and variance
        y_hat, std_ij = [], []
        for xs in x_grid:
            net.zero_grad()
            ys = net(xs.unsqueeze(0))
            ys.backward()
            g_f = flat_grad(net).detach()
            var = sigma2 * ((U @ g_f) ** 2).sum().item()
            y_hat.append(ys.item())
            std_ij.append(var ** 0.5)

        y_hat = torch.tensor(y_hat)
        std_ij = torch.tensor(std_ij)
        lo = y_hat - 1.96 * std_ij
        hi = y_hat + 1.96 * std_ij
        bands[epoch] = (y_hat.detach(), lo, hi)

# 7 ▸ Plot evolution of IJ bands
plt.figure(figsize=(12, 6))
plt.plot(x_grid.squeeze(), y_sin, linestyle='--', color='gray', label='true sin(3x)')
colors = plt.cm.plasma(np.linspace(0, 1, len(checkpoints)))

for i, ep in enumerate(checkpoints):
    y_hat, lo, hi = bands[ep]
    plt.plot(x_grid.squeeze(), y_hat, color=colors[i], label=f"epoch {ep}")
    plt.fill_between(x_grid.squeeze(), lo, hi, color=colors[i], alpha=0.15)

plt.title("IJ 95% Bands Shrink During Training")
plt.xlabel("x")
plt.ylabel("y / ŷ")
plt.legend()
plt.tight_layout()
plt.show()
