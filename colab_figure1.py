"""
CRIM-S Figure 1: Two contrasting regimes
k=4 ring (λ₂≈0.062), α=0 vs α=1 (αk_R > λ₂)
α=0: near-consensus (low Φ), α=1: vertex-concentrated (high Φ)
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

def ring_laplacian(N, k):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(1, k+1):
            A[i,(i+j)%N] = 1; A[i,(i-j)%N] = 1
    return np.diag(A.sum(1)) - A

def proj_simplex(X):
    u = np.sort(X, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    idx = np.arange(1, X.shape[1]+1)[None, :]
    rho = (X.shape[1]-1) - np.argmax((u*idx > cssv)[:, ::-1], axis=1)
    theta = cssv[np.arange(X.shape[0]), rho] / (rho+1.0)
    return np.maximum(X - theta[:, None], 0)

def compute_phi(psi):
    delta = psi - psi.mean(0, keepdims=True)
    return np.sum(delta**2) / psi.shape[0]

N, S = 200, 3
dt, T = 0.01, 200
eta, gs, kR = 1.0, 0.1, 1.0
n_steps = int(T / dt)
rec = max(1, n_steps // 1000)
n_runs = 20

L = ring_laplacian(N, k=4)
lam2 = eigvalsh(L)[1]
print(f"N={N}, k=4, lambda_2={lam2:.4f}")

fig, ax = plt.subplots(1, 1, figsize=(2.99, 2.10))

for alpha, color, label in [
    (0,   '#e08214', r'$\alpha = 0$'),
    (1.0, '#2166ac', r'$\alpha = 1$')
]:
    all_phi = []
    for run in range(n_runs):
        print(f"  alpha={alpha}, run {run+1}/{n_runs}", end='\r')
        np.random.seed(run * 100 + int(alpha*10))
        psi = np.random.dirichlet(np.ones(S), size=N)
        phi_h, t_h = [], []
        for step in range(n_steps):
            delta = psi - psi.mean(0, keepdims=True)
            psi = psi - eta*(L@psi - alpha*kR*delta)*dt \
                  + eta*gs*np.sqrt(dt)*np.random.randn(N, S)
            psi = proj_simplex(psi)
            if step % rec == 0:
                phi_h.append(compute_phi(psi))
                t_h.append(step * dt)
        all_phi.append(phi_h)
    mean_p = np.mean(all_phi, axis=0)
    std_p = np.std(all_phi, axis=0)
    t_h = np.array(t_h)
    ax.plot(t_h, mean_p, color=color, label=label, lw=1.5)
    ax.fill_between(t_h, mean_p-std_p, mean_p+std_p, color=color, alpha=0.15)
    print(f"  alpha={alpha}: steady Phi ~ {np.mean(mean_p[-100:]):.4f}    ")

ax.set_xlabel('Time')
ax.set_ylabel(r'Diversity $\Phi$')
ax.legend(loc='upper right', framealpha=0.85, handlelength=1.4, handletextpad=0.5, borderpad=0.35, labelspacing=0.3)
ax.set_xlim(0, T)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure1.png', dpi=300)
plt.show()
print(f"Done in {time.time()-start:.0f}s")
