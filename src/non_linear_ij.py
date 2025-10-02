import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

"""
PHASE-2 synthetic demo — IJ on a clearly **non-linear sine wave**
────────────────────────────────────────────────────────────────
We extend the IJ pipeline from Phase-1 to a non-convex function
y = sin(3x) + ε,     ε ~ N(0, 0.1²)
Key goals:
  • Train a small MLP; no closed-form OLS exists, so IJ must stand
    on its own.
  • Visualize IJ 95 % confidence band vs. ground-truth sine curve
  • Check influence decay around x* = 0 as before.
"""

# ------------------------------------------------------------------
# 1) Generate nonlinear sine data (moderate n for speed)
# ------------------------------------------------------------------

torch.manual_seed(0)
N = 4000                                   # training points
x_train = torch.linspace(-2*np.pi, 2*np.pi, N).unsqueeze(1)
noise    = 0.1 * torch.randn_like(x_train)
y_train  = torch.sin(3 * x_train) + noise   # y = sin(3x) + ε

# ------------------------------------------------------------------
# 2) Define a tiny MLP (nonlinear) to fit sine
# ------------------------------------------------------------------
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

net = SineNet()

# ------------------------------------------------------------------
# 3) SGD + Influence tracking (U matrix, Eq 2.3 & 3.2)
# ------------------------------------------------------------------
mu, epochs = 2e-3, 200                       # more epochs for convergence
p = sum(p.numel() for p in net.parameters())
U = torch.zeros(N, p)                       # (n × p) influence matrix

# helper to flatten gradients
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
        with torch.no_grad():
            θ = torch.nn.utils.parameters_to_vector(net.parameters()) - mu * g
            torch.nn.utils.vector_to_parameters(θ, net.parameters())
        scale = torch.full((N,1), -1.0/N); scale[k] = 1.0
        U -= mu * scale * g

# ------------------------------------------------------------------
# 4) IJ variance on grid, using empirical σ² from residuals
# ------------------------------------------------------------------
resid  = net(x_train) - y_train
sigma2 = resid.pow(2).mean().item()

x_grid = torch.linspace(-2*np.pi, 2*np.pi, 300).unsqueeze(1)
y_hat, std_ij = [], []
for xs in x_grid:
    net.zero_grad(); ys = net(xs.unsqueeze(0)); ys.backward()
    g_f = flat_grad(net).detach()
    var = sigma2 * ((U @ g_f) ** 2).sum().item()
    y_hat.append(ys.item()); std_ij.append(var**0.5)

y_hat = torch.tensor(y_hat); std_ij = torch.tensor(std_ij)
ci_lo, ci_hi = y_hat - 1.96*std_ij, y_hat + 1.96*std_ij

# ------------------------------------------------------------------
# 5) Influence scatter at x* = 0 (with trend)
# ------------------------------------------------------------------
x_star = torch.tensor([[0.0]])
net.zero_grad(); net(x_star).backward();
infl = (U @ flat_grad(net)).abs().detach().squeeze()
# bin for trend
bins = torch.linspace(-2*np.pi, 2*np.pi, 60)
idx  = torch.bucketize(x_train.squeeze(), bins)
cent, m_inf, s_inf = [], [], []
for b in range(1, len(bins)):
    m = idx == b
    if m.any():
        cent.append(0.5*(bins[b]+bins[b-1]));
        vals = infl[m]; m_inf.append(vals.mean().item()); s_inf.append(vals.std().item())
cent = torch.tensor(cent); m_inf = torch.tensor(m_inf); s_inf = torch.tensor(s_inf)

# ------------------------------------------------------------------
# 6) Plot predictions + influence
# ------------------------------------------------------------------
fig = plt.figure(figsize=(12,5)); gs = gridspec.GridSpec(1,2,[3,2])

a0 = fig.add_subplot(gs[0])
# plot true function for reference
x_true = np.linspace(-2*np.pi, 2*np.pi, 1000)
a0.plot(x_true, np.sin(3*x_true), color='gray', linestyle='--', label='true sin(3x)')
# subsample scatter
sub = torch.randperm(N)[:1000]
a0.scatter(x_train[sub], y_train[sub], s=8, alpha=0.25, label='train (1k)')
a0.plot(x_grid, y_hat, color='tab:blue', label='MLP fit')
a0.fill_between(x_grid.squeeze(), ci_lo, ci_hi, color='tab:blue', alpha=0.25, label='IJ 95% CI')
a0.set_title('Sine function fit ±95% IJ band'); a0.set_xlabel('x'); a0.set_ylabel('y'); a0.legend()

a1 = fig.add_subplot(gs[1])
a1.scatter(x_train, infl, s=6, alpha=0.4, label='|U_i| (all)')
a1.plot(cent, m_inf, color='red', label='binned mean')
a1.fill_between(cent, m_inf-s_inf, m_inf+s_inf, color='red', alpha=0.2, label='±1 SD')
a1.axvline(0, ls='--', c='gray'); a1.set_yscale('log')
a1.set_title('Influence vs x_i at x*=0'); a1.set_xlabel('x_i'); a1.set_ylabel('|U_i| (log)'); a1.legend()

plt.tight_layout(); plt.show()

print(f"Mean IJ std on grid : {std_ij.mean():.4f}")