import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ------------------------------------------------------------
# 1. Larger synthetic dataset (n = 200)
# ------------------------------------------------------------
# We create a simple linear data-generating process with noise:
#   y = 3x + 0.5 + noise
# This synthetic dataset lets us validate the learned model and its uncertainty.
torch.manual_seed(0)  # For reproducibility
n = 200               # Number of training samples
x_train = torch.randn(n, 1)  # Inputs sampled from standard normal
y_train = 3 * x_train + 0.5 + 0.1 * torch.randn_like(x_train)  # Add Gaussian noise
w = torch.ones(n)  # Uniform weights (w_i = 1 for all i) corresponds to Eq-(1) with w_i=1

# ------------------------------------------------------------
# 2. Model — one hidden layer with tanh activation
# ------------------------------------------------------------
# We define a small neural network (TinyNet) by subclassing nn.Module.
# - nn.Linear implements a fully-connected layer: output = input * weight^T + bias.
# - torch.tanh applies a smooth nonlinearity, ensuring differentiability everywhere.

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First linear layer: maps 1-dimensional input to 16 hidden units
        self.fc1 = nn.Linear(in_features=1, out_features=16)
        # Second linear layer: maps hidden representation back to scalar output
        self.fc2 = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        # Forward pass (inference): compute activations and return prediction
        # 1. self.fc1(x) computes a linear transform of x
        # 2. torch.tanh(...) adds nonlinearity
        # 3. self.fc2(...) maps back to a scalar prediction
        return self.fc2(torch.tanh(self.fc1(x)))

# Instantiate the model
net = TinyNet()

# ------------------------------------------------------------
# 3. Hyperparameters & helpers
# ------------------------------------------------------------
μ = 5e-4           # Learning rate (step size) for SGD. Smaller for stability on larger data
n_epochs = 200    # Number of full passes over the data (epochs)

# Flatten initial parameters θ into a single vector (p-dimensional)
θ_init = torch.nn.utils.parameters_to_vector(net.parameters())
p = θ_init.numel()  # Total number of parameters in the model

# Influence matrix U (n × p): each row U[i] will track ∂θ/∂w_i over training (Eq 3.2)
U = torch.zeros(n, p)

# Helper function: extract the current gradient ∇_θ L_k as a flat vector
def flat_grad(model):
    # PyTorch stores gradients in parameter.grad after backward()
    # We gather them and flatten into one vector for easy arithmetic
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad)
        else:
            grads.append(torch.zeros_like(param))
    return torch.nn.utils.parameters_to_vector(grads)

# ------------------------------------------------------------
# 4. Training loop: SGD + influence tracking (Eq 2.3 & 3.2)
# ------------------------------------------------------------
# We perform pure SGD (batch size = 1) for clarity.
# At each sample k:
#  1) Forward pass: compute prediction f(x_k, θ)
#  2) Compute loss L_k = (f(x_k)-y_k)^2 (MSE) weighted by w_k
#  3) Backward pass: compute ∇_θ L_k via autograd
#  4) SGD step: θ ← θ - μ ∇_θ L_k  (Eq 2.3)
#  5) Influence update: U[i] ← U[i] - μ [1(i=k)-1/n] ∇_θ L_k  (Eq 3.2)

for epoch in range(n_epochs):
    for k in range(n):  # Loop over each training example
        xk = x_train[k:k+1]  # Shape (1,1)
        yk = y_train[k:k+1]  # Shape (1,1)
        wk = w[k]            # Scalar weight

        # --- Forward pass & loss computation ---
        net.zero_grad()  # Clear previous gradients in model
        y_pred = net(xk)  # f(x_k, θ)
        # Mean Squared Error: L_k = (y_pred - y_k)^2
        loss = wk * nn.functional.mse_loss(y_pred, yk)

        # --- Backward pass: compute gradients dL_k/dθ ---
        # create_graph=True retains graph so we can compute higher-order derivatives
        loss.backward(create_graph=True)
        gθ = flat_grad(net)  # ∇_θ L_k, a vector of length p

        # --- (4) SGD parameter update (Eq 2.3) ---
        # We disable gradient tracking for manual parameter manipulation
        with torch.no_grad():
            θ_vec = torch.nn.utils.parameters_to_vector(net.parameters())
            θ_vec_new = θ_vec - μ * gθ  # θ ← θ - μ ∇_θ L_k
            torch.nn.utils.vector_to_parameters(θ_vec_new, net.parameters())

        # --- (5) Influence update (Eq 3.2) ---
        # Build jackknife scaling: +1 for sample k, -1/n for all others
        scale = torch.full((n, 1), -1.0 / n)
        scale[k] = 1.0
        # Update influence matrix by subtracting scaled gradient
        U -= μ * scale * gθ

# ------------------------------------------------------------
# 5. Prediction on test grid + IJ uncertainty band (Eq 4.1 & 4.2)
# ------------------------------------------------------------
# We now use the trained θ and accumulated U to compute:
#  - y_hat(x) = f(x, θ)  (forward pass)
#  - ∇_θ f(x, θ)         (backward pass)
#  - U_i^ŷ = U_i · ∇_θ f(x) (Eq 4.1)
#  - Var_IJ(x) = Σ_i (U_i^ŷ)^2  (Eq 4.2)

# Create a dense grid of inputs for plotting
x_grid = torch.linspace(x_train.min() - 0.5, x_train.max() + 0.5, 200).unsqueeze(1)

y_hat_list, std_list = [], []
for x_star in x_grid:
    # Compute prediction and its θ-sensitivity
    net.zero_grad()
    y_star = net(x_star.unsqueeze(0))  # Forward pass
    y_star.backward()                   # Compute ∇_θ f(x_star)
    gradθ_f = flat_grad(net)           # Flatten ∇_θ f(x_star)

    # Push influence vectors forward: U_i^ŷ = U_i · ∇_θ f(x_star)
    U_y = U @ gradθ_f
    var_hat = (U_y ** 2).sum().item()  # IJ variance estimate

    y_hat_list.append(y_star.item())
    std_list.append(var_hat ** 0.5)

# Convert lists to tensors for plotting
y_hat = torch.tensor(y_hat_list)
std_hat = torch.tensor(std_list)
ci_upper = y_hat + 1.96 * std_hat  # 95% CI upper bound
ci_lower = y_hat - 1.96 * std_hat  # 95% CI lower bound

# ------------------------------------------------------------
# 6. Visualization
# ------------------------------------------------------------
# Left: data points, learned curve, and IJ confidence band
# Right: influence magnitudes for a specific x* (here x*=0)
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2])

# Left plot: regression line + band
ax0 = plt.subplot(gs[0])
ax0.scatter(x_train.numpy(), y_train.numpy(), alpha=0.4, label="Train points")
ax0.plot(x_grid.numpy(), y_hat.numpy(), color="blue", label="Predicted y")
ax0.fill_between(x_grid.squeeze().numpy(), ci_lower.numpy(), ci_upper.numpy(),
                 color="orange", alpha=0.3, label="IJ 95% CI")
ax0.set_title("Prediction curve with IJ confidence band")
ax0.set_xlabel("x (input)")
ax0.set_ylabel("y / ŷ (output)")
ax0.legend()

# Right plot: per-point influence on ŷ(x*=0)
x_focus = torch.tensor([[0.0]])
net.zero_grad()
y_focus = net(x_focus)
y_focus.backward()
U_focus = U @ flat_grad(net)  # Compute U_i^ŷ at x*=0

ax1 = plt.subplot(gs[1])
ax1.scatter(x_train.numpy(), U_focus.abs().detach().numpy(), s=2, alpha=0.4)
ax1.set_title("|Influence| on ŷ(x*=0)")
ax1.set_xlabel("x_i (training input)")
ax1.set_ylabel("|U_i^{ŷ}| (abs magnitude)")

plt.tight_layout()
plt.show()
