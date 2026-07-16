"""
CRIM-S Experiment 2: Boundary Contact Frequency Analysis
Run on Google Colab free tier (~8 min)
Produces: exp2.png
"""
import numpy as np
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt

# --- journal-ready sizing (sn-jnl: column 2.99in, text 6.30in) -----------
# The figure is drawn at the size it will actually be printed, so no
# shrink-down occurs and all lettering stays >= 8pt (Springer minimum).
import matplotlib as mpl
mpl.rcParams.update({
    'font.size'       : 8,
    'axes.labelsize'  : 9,
    'axes.titlesize'  : 9,
    'xtick.labelsize' : 8,
    'ytick.labelsize' : 8,
    'legend.fontsize' : 7.5,
    'lines.linewidth' : 1.2,
    'axes.linewidth'  : 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'savefig.dpi'     : 400,
    'figure.dpi'      : 110,
})
# ------------------------------------------------------------------------
import time

start = time.time()
np.random.seed(42)

def ring_laplacian(N, k=2):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(1, k+1):
            A[i, (i+j)%N] = 1
            A[i, (i-j)%N] = 1
    return np.diag(A.sum(1)) - A

def proj_simplex(X):
    u = np.sort(X, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    idx = np.arange(1, X.shape[1]+1)[None, :]
    rho = (X.shape[1]-1) - np.argmax((u * idx > cssv)[:, ::-1], axis=1)
    theta = cssv[np.arange(X.shape[0]), rho] / (rho + 1.0)
    return np.maximum(X - theta[:, None], 0)

N, S = 200, 3
dt = 0.01
T = 500
eta, gs, kR = 1.0, 0.1, 1.0
n_steps = int(T / dt)
rec = max(1, n_steps // 500)
n_runs = 20

fig, axes = plt.subplots(2, 1, figsize=(2.99, 4.30))

# ---- Panel (a): Boundary contact fraction vs lambda_2 ----
ax = axes[0]
k_values = [2, 3, 4, 6, 8, 10, 14]
colors_a = {0.0: '#d73027', 0.2: '#fc8d59', 0.5: '#4575b4'}

for alpha in [0.0, 0.2, 0.5]:
    lam2_v, bf_v = [], []
    for kk in k_values:
        L = ring_laplacian(N, k=kk)
        lam2 = eigvalsh(L)[1]
        lam2_v.append(lam2)
        bf_runs = []
        for run in range(n_runs):
            print(f"  Panel(a): alpha={alpha}, k={kk}, run {run+1}/{n_runs}", end='\r')
            np.random.seed(run * 1000 + kk * 7 + int(alpha * 10))
            psi = np.random.dirichlet(np.ones(S), size=N)
            bf_acc = []
            for step in range(n_steps):
                delta = psi - psi.mean(0, keepdims=True)
                psi_new = psi - eta * (L @ psi - alpha * kR * delta) * dt \
                          + eta * gs * np.sqrt(dt) * np.random.randn(N, S)
                # Record boundary contact BEFORE projection (second half only)
                if step % rec == 0 and step > n_steps // 2:
                    touching = np.any(psi_new < 0, axis=1) | np.any(psi_new > 1, axis=1)
                    bf_acc.append(touching.mean())
                psi = proj_simplex(psi_new)
            bf_runs.append(np.mean(bf_acc) if bf_acc else 0)
        bf_v.append(np.mean(bf_runs))
    ax.plot(lam2_v, bf_v, 'o-', color=colors_a[alpha],
            label=rf'$\alpha = {alpha}$', ms=4, lw=1.5)

ax.set_xlabel(r'Algebraic connectivity $\lambda_2$')
ax.set_ylabel('Boundary contact fraction')
ax.set_title('(a) Steady-state boundary contact')
ax.legend(framealpha=0.85, handlelength=1.4, handletextpad=0.5, borderpad=0.35, labelspacing=0.3)
ax.grid(True, alpha=0.3)
print()

# ---- Panel (b): Distribution of component values ----
ax = axes[1]
L_test = ring_laplacian(N, k=3)
configs = [
    (0.0, '#d73027', r'$\alpha = 0$'),
    (0.3, '#4575b4', r'$\alpha = 0.3$'),
    (0.6, '#1a9850', r'$\alpha = 0.6$'),
]
for alpha, color, label in configs:
    print(f"  Panel(b): alpha={alpha}")
    np.random.seed(999)
    psi = np.random.dirichlet(np.ones(S), size=N)
    for step in range(n_steps):
        delta = psi - psi.mean(0, keepdims=True)
        psi = psi - eta * (L_test @ psi - alpha * kR * delta) * dt \
              + eta * gs * np.sqrt(dt) * np.random.randn(N, S)
        psi = proj_simplex(psi)
    vals = psi.flatten()
    ax.hist(vals, bins=40, range=(0, 1), density=True, alpha=0.4,
            color=color, label=label, edgecolor=color, linewidth=0.8)

ax.axvline(1.0 / S, color='k', ls=':', lw=1, label=f'Interior ($1/S = {1/S:.2f}$)')
ax.set_xlabel(r'Component value $\psi_{ij}$')
ax.set_ylabel('Density')
ax.set_title('(b) Distribution of agent state components')
ax.legend(framealpha=0.85, handlelength=1.4, handletextpad=0.5, borderpad=0.35, labelspacing=0.3)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exp2.png', dpi=300)
plt.show()
print(f"\nDone in {time.time()-start:.0f}s. Saved exp2.png")
