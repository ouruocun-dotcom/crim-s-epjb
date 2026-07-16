"""
CRIM-S Experiment 1: Simplex-constrained vs Unconstrained Dynamics
Run on Google Colab free tier (~5 min)
Produces: exp1.png
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

# ---- Network ----
def ring_laplacian(N, k=2):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(1, k+1):
            A[i, (i+j)%N] = 1
            A[i, (i-j)%N] = 1
    return np.diag(A.sum(1)) - A

# ---- Vectorized simplex projection ----
def proj_simplex(X):
    u = np.sort(X, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    idx = np.arange(1, X.shape[1]+1)[None, :]
    rho = (X.shape[1]-1) - np.argmax((u * idx > cssv)[:, ::-1], axis=1)
    theta = cssv[np.arange(X.shape[0]), rho] / (rho + 1.0)
    return np.maximum(X - theta[:, None], 0)

def compute_phi(psi):
    delta = psi - psi.mean(0, keepdims=True)
    return np.sum(delta**2) / psi.shape[0]

# ---- Parameters (matching paper Table 1) ----
N, S = 200, 3
dt = 0.01
T = 500
eta, gs, kR = 1.0, 0.1, 1.0
n_steps = int(T / dt)
rec = max(1, n_steps // 1000)  # ~1000 data points
n_runs = 20

L = ring_laplacian(N, k=3)
lam2 = eigvalsh(L)[1]
print(f"N={N}, lambda_2={lam2:.4f}, n_steps={n_steps}")

# ---- Simulation function ----
def run_sim(L, N, S, alpha, kR, eta, gs, dt, n_steps, rec, use_simplex=True, seed=0):
    np.random.seed(seed)
    psi = np.random.dirichlet(np.ones(S), size=N)
    phi_h, t_h = [], []
    for step in range(n_steps):
        delta = psi - psi.mean(0, keepdims=True)
        psi = psi - eta * (L @ psi - alpha * kR * delta) * dt \
              + eta * gs * np.sqrt(dt) * np.random.randn(N, S)
        if use_simplex:
            psi = proj_simplex(psi)
        if step % rec == 0:
            phi_h.append(compute_phi(psi))
            t_h.append(step * dt)
    return np.array(t_h), np.array(phi_h)

# ---- Run experiments ----
fig, axes = plt.subplots(2, 1, figsize=(2.99, 4.30))

for idx, (alpha, title) in enumerate([
    (0.0, r'$\alpha = 0$'),
    (0.3, r'$\alpha = 0.3$  ($\alpha k_R > \lambda_2$)')
]):
    ax = axes[idx]
    for constrained, ls, color, label in [
        (True,  '-',  '#2166ac', 'Simplex-constrained'),
        (False, '--', '#b2182b', r'Unconstrained ($\mathbb{R}^S$)')
    ]:
        all_phi = []
        for run in range(n_runs):
            print(f"  alpha={alpha}, {'simplex' if constrained else 'free'}, run {run+1}/{n_runs}", end='\r')
            t, phi = run_sim(L, N, S, alpha, kR, eta, gs, dt, n_steps, rec,
                             use_simplex=constrained, seed=run*100+idx*10+int(constrained))
            all_phi.append(phi)
        mean_p = np.mean(all_phi, axis=0)
        std_p = np.std(all_phi, axis=0)
        ax.semilogy(t, np.maximum(mean_p, 1e-12), ls, color=color,
                    label=label, lw=1.5)
        ax.fill_between(t, np.maximum(mean_p - std_p, 1e-12),
                        np.maximum(mean_p + std_p, 1e-12),
                        color=color, alpha=0.12)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'Diversity $\Phi(t)$')
    ax.set_title(f'({chr(97+idx)})  {title}', loc='left')
    # the unconstrained trace sweeps ~100 decades: no interior corner is free
    ax.legend(loc='lower right', framealpha=0.9, handlelength=1.4,
              handletextpad=0.5, borderpad=0.35, labelspacing=0.3)
    ax.set_xlim(0, T)
    ax.grid(True, which='both', alpha=0.25, lw=0.5)
    print()

plt.subplots_adjust(left=0.20, right=0.95, top=0.93, bottom=0.11, hspace=0.55)
plt.savefig('exp1.png', dpi=300)
plt.show()
print(f"\nDone in {time.time()-start:.0f}s. Saved exp1.png")
