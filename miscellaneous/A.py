import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Helper functions (updated range)
# -------------------------------
def generate_data(n, x_min=-5, x_max=5, noise_std=0.4, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(x_min, x_max, n)
    y = 2.0 * x + 1.0 + rng.normal(0.0, noise_std, size=n)
    return x, y

def ols_band(x, y, x_grid):
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    theta_hat = XtX_inv @ X.T @ y
    res = y - X @ theta_hat
    sigma2 = (res @ res) / (n - 2)
    Xg = np.column_stack([np.ones_like(x_grid), x_grid])
    y_pred = Xg @ theta_hat
    se = np.sqrt(sigma2 * np.einsum("ij,jk,ik->i", Xg, XtX_inv, Xg))
    return y_pred, se

class OneNodeTanh:
    def __init__(self, rng):
        self.w1, self.b1, self.w2, self.b2 = rng.normal(size=4)

    def forward(self, x):
        self.z = np.tanh(self.w1 * x + self.b1)
        return self.w2 * self.z + self.b2

    def grads(self, x, y):
        y_pred = self.forward(x)
        dL = y_pred - y
        dw2 = dL * self.z
        db2 = dL
        dz = dL * self.w2 * (1 - self.z ** 2)
        dw1 = dz * x
        db1 = dz
        return np.array([dw1, db1, dw2, db2], dtype=float)

    def step(self, g, lr):
        self.w1, self.b1, self.w2, self.b2 = (
            p - lr * gp for p, gp in zip([self.w1, self.b1, self.w2, self.b2], g)
        )

def train_tanh_ij(x, y, eta=1e-2, steps=120000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    net = OneNodeTanh(rng)
    U = np.zeros((n, 4))
    for _ in range(steps):
        i = rng.integers(0, n)
        g_vec = net.grads(x[i], y[i])
        net.step(g_vec, eta)
        U += (eta / n) * g_vec
        U[i] -= eta * g_vec
    return net, U

def ij_band(net, U, x_grid, eta, n):
    preds, se = [], []
    for xv in x_grid:
        z = np.tanh(net.w1 * xv + net.b1)
        dz = 1 - z**2
        grad_y = np.array([net.w2 * dz * xv, net.w2 * dz, z, 1.0])
        Uy = U @ grad_y
        var = (eta**2 / n) * np.sum(Uy**2)
        preds.append(net.w2 * z + net.b2)
        se.append(np.sqrt(var))
    return np.array(preds), np.array(se)

# ----------------------------------
# Experiment across larger n values
# ----------------------------------
sample_sizes = [100, 400, 800, 1600]
eta = 1e-2
x_grid = np.linspace(-6, 6, 300)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.ravel()

for ax, n, seed in zip(axes, sample_sizes, [0, 1, 2, 3]):
    x, y = generate_data(n, x_min=-5, x_max=5, seed=seed)
    y_ols, se_ols = ols_band(x, y, x_grid)

    net, U = train_tanh_ij(x, y, eta=eta, steps=120000, seed=seed)
    y_ij, se_ij = ij_band(net, U, x_grid, eta, n)

    ax.scatter(x, y, s=8, alpha=0.4, label="data")
    ax.plot(x_grid, y_ols, color="tab:blue", label="OLS mean")
    ax.fill_between(x_grid, y_ols - 1.96 * se_ols, y_ols + 1.96 * se_ols,
                    color="tab:blue", alpha=0.25, label="OLS 95% CI")
    ax.plot(x_grid, y_ij, color="k", label="IJ mean")
    ax.fill_between(x_grid, y_ij - 1.96 * se_ij, y_ij + 1.96 * se_ij,
                    color="gray", alpha=0.3, label="IJ 95% CI")

    ax.set_title(f"n = {n}")
    ax.grid(True)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles[:5], labels[:5], loc="upper center", ncol=5)
fig.suptitle("OLS vs IJ Bands with Wider x‑Range and Larger Sample Sizes", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
