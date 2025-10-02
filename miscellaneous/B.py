import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from tqdm import trange

# ─ reproducibility ────────────────────────────────────────────────
np.random.seed(0); tf.random.set_seed(0)

# ─ 1. synthetic data (x∈[-100,100]) ──────────────────────────────
N, R, SIGMA = 1000, 100, 10.0
x_train = np.random.uniform(-R, R, (N, 1)).astype("float32")
y_train = (3. * x_train[:, 0] + 5.
           + np.random.normal(0, SIGMA, N)).astype("float32")[:, None]
x_grid  = np.linspace(-R, R, 400, dtype="float32")[:, None]
G = x_grid.shape[0]

# ─ 2. 1‑hidden‐layer tanh net ────────────────────────────────────
inp = tf.keras.Input(shape=(1,))
h   = tf.keras.layers.Dense(12, activation="tanh")(inp)
out = tf.keras.layers.Dense(1)(h)
model = tf.keras.Model(inp, out)

θ_vars = model.trainable_variables
P      = sum(int(np.prod(v.shape)) for v in θ_vars)   # total params

# ─ 3. per‑example influence matrix U (N × P) ─────────────────────
U = tf.zeros((N, P), tf.float32)

LR, EPOCHS = 1e-3, 300
η2_div_n   = (LR**2) / N

loss_hist, ci_hist = [], []

def flatten(grads):
    """list of (shape_k) → 1‑D tensor of length P"""
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

# ─ 4. manual SGD + IJ updates ────────────────────────────────────
for epoch in trange(EPOCHS, desc="epochs"):
    for i in np.random.permutation(N):            # NumPy shuffle
        xi, yi = x_train[i:i+1], y_train[i:i+1]

        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(xi, training=True)
            loss   = tf.reduce_mean((y_pred - yi)**2)
        grads = tape.gradient(loss, θ_vars)
        g_vec = flatten(grads)

        # parameter step
        for v, g in zip(θ_vars, grads):
            v.assign_sub(LR * g)

        # influence update: up‑weight row i, down‑weight all
        U = tf.tensor_scatter_nd_add(U, [[i]], [-LR * g_vec])
        U += (LR / N) * g_vec                       # broadcast

    # ---- diagnostics each epoch ----
    with tf.GradientTape() as tape:
        y_grid = model(x_grid)                     # (G,1)
    J = tape.jacobian(y_grid, θ_vars)              # list: (G,…)
    # build (G, P) matrix
    Jflat = tf.concat([tf.reshape(j, (G, -1)) for j in J], axis=1)

    var_grid = η2_div_n * tf.reduce_sum(
        tf.matmul(Jflat, U, transpose_b=True)**2, axis=1
    )                                              # (G,)
    loss_hist.append(loss.numpy())
    ci_hist.append(1.96 * tf.sqrt(var_grid)[G//2].numpy())  # CI @ x=0

# ─ 5. final band plot ────────────────────────────────────────────
pred = model(x_grid).numpy().squeeze()
stderr = np.sqrt(var_grid.numpy())
upper, lower = pred + 1.96*stderr, pred - 1.96*stderr

plt.figure(figsize=(10,6))
plt.scatter(x_train, y_train, s=4, alpha=.15, label="data")
plt.plot(x_grid, pred, "k", lw=2, label="prediction")
plt.fill_between(x_grid.squeeze(), lower, upper, alpha=.25,
                 color="indianred", label="95 % IJ CI")
plt.title("1‑hidden tanh net • IJ confidence band (x∈[-100,100])")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.tight_layout()

# ─ 6. CI width & loss curves ────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(ci_hist, label="CI half‑width @ x=0"); ax1.set_ylabel("CI width")
ax2 = ax1.twinx(); ax2.plot(loss_hist, color="orange", label="loss")
ax2.set_ylabel("loss"); ax1.set_xlabel("epoch")
ax1.set_title("Training dynamics – IJ vs Loss")
fig.legend(loc="upper right"); fig.tight_layout()
plt.show()
