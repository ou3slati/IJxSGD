import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

"""
FAST‑DEMO  (N ≈ 800 points, 120 epochs)
=======================================
Same IJ‑band tracker as the full version, but with **smaller settings** so it
runs in <10 s on CPU:

    • N           =  800   (training samples)
    • HIDDEN      =   32   (per layer)
    • EPOCHS      =  120
    • SNAPSHOTS   = [0,30,60,90,120]

Everything else (influence‑matrix maths, unbiased σ², minibatching) is
identical to the previous script, just lighter.
"""

# ─────────────────────── 0. Hyper‑parameters ─────────────────────────
N             = 800
BATCH         = 64
EPOCHS        = 120
SNAP_EPOCHS   = [0, 30, 60, 90, 120]
LR            = 2e-3
HIDDEN        = 32
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────── 1. Synthetic data ──────────────────────────

torch.manual_seed(0)
X_train = torch.linspace(-2*np.pi, 2*np.pi, N).unsqueeze(1)
Y_true  = torch.sin(3*X_train)
Y_train = Y_true + 0.1*torch.randn_like(X_train)
train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
loader   = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True)

# query grid
Xq = torch.linspace(-2*np.pi, 2*np.pi, 200).unsqueeze(1)
Yq_true = torch.sin(3*Xq)

# ─────────────────────── 2. Model ───────────────────────────────────

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 1)
        )
    def forward(self, x):
        return self.net(x)

net   = MLP().to(DEVICE)
optim = torch.optim.Adam(net.parameters(), lr=LR)
loss_fn = nn.MSELoss(reduction='none')

P = sum(p.numel() for p in net.parameters())
U = torch.zeros(N, P, device=DEVICE)

flat = lambda params: torch.cat([p.grad.reshape(-1) for p in params])

# ─────────────────────── 3. Train + snapshot IJ bands ───────────────
X_train, Y_train, Xq = X_train.to(DEVICE), Y_train.to(DEVICE), Xq.to(DEVICE)

bands = {}
for epoch in range(EPOCHS+1):
    net.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optim.zero_grad()
        preds = net(xb)
        losses = (preds - yb).pow(2)
        losses.mean().backward(create_graph=True)

        # per‑sample gradients for the batch
        grads = []
        for l in losses:
            net.zero_grad(); l.backward(retain_graph=True)
            grads.append(flat(net.parameters()).detach())
        G = torch.stack(grads)                              # (B,P)

        # SGD update
        g_batch = G.mean(0)
        with torch.no_grad():
            θ = flat(net.parameters()) - LR * g_batch
            torch.nn.utils.vector_to_parameters(θ, net.parameters())

        # Influence update
        batch_idx = (X_train.view(-1,1) == xb).all(1).nonzero(as_tuple=True)[0]
        scale = (-LR / N) * g_batch
        for local, idx in enumerate(batch_idx):
            U[idx] += LR * G[local] + scale

    # ─ snapshot bands ─
    if epoch in SNAP_EPOCHS:
        net.eval()
        with torch.no_grad():
            resid  = (net(X_train) - Y_train)
            sigma2 = resid.pow(2).sum().item() / (N - P)

        yhat, std = [], []
        for xstar in Xq:
            net.zero_grad(); y_ = net(xstar.unsqueeze(0)); y_.backward()
            g_f = flat(net.parameters()).detach()
            var = sigma2 * ( (U @ g_f)**2 ).sum().item()
            yhat.append(y_.item()); std.append(np.sqrt(var))
        bands[epoch] = (np.array(yhat), np.array(std))

# ─────────────────────── 4. Plot IJ evolution ───────────────────────
col = plt.cm.viridis(np.linspace(0,1,len(SNAP_EPOCHS)))
plt.figure(figsize=(10,5))
plt.plot(Xq.cpu(), Yq_true.cpu(), '--', c='gray', lw=2, label='true sin(3x)')
for c,(ep,(m,s)) in zip(col, bands.items()):
    plt.plot(Xq.cpu(), m, c=c, lw=2, label=f'epoch {ep}')
    plt.fill_between(Xq.cpu().squeeze(), m-1.96*s, m+1.96*s, color=c, alpha=.18)
plt.title('Fast‑demo  ·  IJ 95% bands  (N=800)')
plt.xlabel('x'); plt.ylabel('y / ŷ');
plt.legend(fontsize=8, ncol=3); plt.tight_layout(); plt.show()
