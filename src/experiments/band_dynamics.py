"""
Track IJ confidence‑interval width at x*=0 over training epochs.
"""
import torch, matplotlib.pyplot as plt, numpy as np
from data.synthetic import make_dataset
from models.linear import build as build_model
from models.utilities import phi
from influence.tracker import InfluenceTracker
from influence.utils import flatten_grad

x, y = make_dataset("linear", 800, 0.1)
model = build_model()
tracker = InfluenceTracker(len(x), model)
lr, epochs = 1e-2, 200
criterion = torch.nn.MSELoss()

ci_width, losses = [], []
for ep in range(epochs):
    epoch_loss = 0.0
    for i in torch.randperm(len(x)):
        xi, yi = x[i:i+1], y[i:i+1]
        model.zero_grad()
        loss = criterion(model(phi(xi,"linear")), yi); epoch_loss += loss.item()
        loss.backward()
        g = flatten_grad(model).detach()
        with torch.no_grad():
            theta = torch.nn.utils.parameters_to_vector(model.parameters()) - lr * g
            torch.nn.utils.vector_to_parameters(theta, model.parameters())
        tracker.sgd_update(i, g, lr)
    # variance at x*=0
    model.zero_grad()
    out0 = model(phi(torch.tensor([[0.0]]),"linear"))
    out0.backward()
    g0 = flatten_grad(model).detach()
    resid = model(phi(x,"linear")) - y
    sigma2 = resid.pow(2).mean().item()
    var0 = sigma2 * (tracker.U @ g0).pow(2).sum().item()
    ci_width.append(1.96*np.sqrt(var0))
    losses.append(epoch_loss/len(x))

plt.figure(figsize=(6,4))
plt.plot(ci_width, label="CI half‑width @ x*=0")
plt.plot(losses, label="MSE per sample")
plt.xlabel("epoch"); plt.legend(); plt.tight_layout(); plt.show()
