"""
Log gradient norms for a specific point to verify cancellation over epochs.
"""
import torch, matplotlib.pyplot as plt
from data.synthetic import make_dataset
from models.linear import build as build_model
from models.utilities import phi
from influence.tracker import InfluenceTracker
from influence.utils import flatten_grad

x, y = make_dataset("linear", 400, 0.1)
model = build_model()
tracker = InfluenceTracker(len(x), model)
lr, epochs = 100, 400    # many passes for clearer plot
criterion = torch.nn.MSELoss()

target_idx = 42
log = []

for ep in range(epochs):
    for i in torch.randperm(len(x)):
        xi, yi = x[i:i+1], y[i:i+1]
        model.zero_grad()
        loss = criterion(model(phi(xi,"linear")), yi); loss.backward()
        g = flatten_grad(model).detach()
        # record grad norm when visiting target point
        if i == target_idx:
            log.append(g.norm().item())
        with torch.no_grad():
            theta = torch.nn.utils.parameters_to_vector(model.parameters()) - lr * g
            torch.nn.utils.vector_to_parameters(theta, model.parameters())
        tracker.sgd_update(i, g, lr)

plt.plot(log); plt.title(f"‖∇θL_{target_idx}‖ over steps"); plt.xlabel("visits")
plt.ylabel("gradient norm"); plt.tight_layout(); plt.show()
