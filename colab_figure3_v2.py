"""
Figure 3 (revised): Two panels only - (a) Three regimes, (b) Phase diagram
Removes the problematic Phi* vs lambda_2 panel (already shown in Figure 2).
Run on Colab free tier (~5 min)
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
dt, eta, gs, kR = 0.01, 1.0, 0.1, 1.0
alpha_base = 0.3

def run_steady(L, alpha, T_burn=200, T_avg=100):
    n_b, n_a = int(T_burn/dt), int(T_avg/dt)
    psi = np.random.dirichlet(np.ones(S), size=N)
    for s in range(n_b):
        d = psi - psi.mean(0, keepdims=True)
        psi = proj_simplex(psi - eta*(L@psi - alpha*kR*d)*dt + eta*gs*np.sqrt(dt)*np.random.randn(N,S))
    acc = []
    for s in range(n_a):
        d = psi - psi.mean(0, keepdims=True)
        psi = proj_simplex(psi - eta*(L@psi - alpha*kR*d)*dt + eta*gs*np.sqrt(dt)*np.random.randn(N,S))
        if s % 100 == 0: acc.append(compute_phi(psi))
    return np.mean(acc)

def run_traj(L, alpha, T=100, rec=50):
    n = int(T/dt)
    psi = np.random.dirichlet(np.ones(S), size=N)
    ph, th = [], []
    for s in range(n):
        d = psi - psi.mean(0, keepdims=True)
        psi = proj_simplex(psi - eta*(L@psi - alpha*kR*d)*dt + eta*gs*np.sqrt(dt)*np.random.randn(N,S))
        if s % rec == 0: ph.append(compute_phi(psi)); th.append(s*dt)
    return np.array(th), np.array(ph)

fig, axes = plt.subplots(2, 1, figsize=(2.99, 3.90))

# Panel (a): Three regimes
print("Panel (a): Three regimes")
ax = axes[0]
# k=16 -> lam2=1.46 (gap +1.16, deep in the stable regime)
# k=10 -> lam2=0.38 (gap +0.08, right at the boundary -> critical)
# k=4  -> lam2=0.03 (gap -0.27, unstable)
L6 = ring_laplacian(N, k=16); lam2_s = eigvalsh(L6)[1]
L3 = ring_laplacian(N, k=10); lam2_c = eigvalsh(L3)[1]
a_crit = lam2_c / kR
a_coll = alpha_base * 5

np.random.seed(77); t1, p1 = run_traj(L6, alpha_base)
np.random.seed(77); t2, p2 = run_traj(L3, a_crit)
np.random.seed(77); t3, p3 = run_traj(L3, a_coll)

ax.semilogy(t1, p1, color='#2166ac', lw=1.5, label=r'stable')
ax.semilogy(t2, p2, color='#e08214', lw=1.5, label=r'critical')
ax.semilogy(t3, p3, color='#d73027', lw=1.5, label=r'instability')
ax.set_xlabel('Time'); ax.set_ylabel(r'Diversity $\Phi(t)$')
ax.set_title('(a)', loc='left')
# On the log axis the three traces occupy the whole vertical range -- stable at
# the bottom (~5e-4), critical in the middle (~1e-2), instability at the top
# (~0.5). No interior corner is free, so the legend goes above the panel.
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
          handlelength=1.2, handletextpad=0.4, columnspacing=1.0, borderpad=0.2)
ax.grid(True, which='both', alpha=0.25, lw=0.5)
print("  Done.")

# Panel (b): Phase diagram
# The regime is classified from the DATA, not from the theory being tested:
# below threshold the population stays near consensus (Phi small); above it,
# agents separate onto simplex vertices (Phi large). The classifier is the
# measured Phi*, and we check where the transition falls.
print("Panel (b): Phase diagram")
ax = axes[1]

# lambda_2 must straddle the boundary for every alpha, so sweep the ring
# degree widely AND use denser graphs (a k=4 ring on N=200 has lambda_2 ~ 0.03;
# reaching lambda_2 ~ 2 needs k of order 20+).
# lambda_2 = 0.03 .. 2.77, straddling the boundary over alpha in [0, 2]
k_list   = [4, 8, 10, 12, 13, 14, 15, 16, 17, 18, 20]
lam2_of  = {}
for k in k_list:
    L = ring_laplacian(N, k=k)
    lam2_of[k] = (L, eigvalsh(L)[1])

a_range = np.linspace(0.05, 2.0, 10)
PHI_SPLIT = 0.05          # near-consensus vs vertex-concentrated

for a_test in a_range:
    for k in k_list:
        L, lam2 = lam2_of[k]
        np.random.seed(int(a_test * 100) + k)
        pf = run_steady(L, a_test)
        near_consensus = pf < PHI_SPLIT
        if near_consensus:
            ax.scatter(a_test, lam2, marker='o', facecolor='#1a9850',
                       edgecolor='#1a9850', s=13, lw=0.8, alpha=0.9, zorder=4)
        else:
            ax.scatter(a_test, lam2, marker='x', color='#d73027',
                       s=16, lw=1.0, alpha=0.9, zorder=4)
        print(f"  a={a_test:.2f} k={k:2d} lam2={lam2:5.2f} Phi*={pf:.4f}", end='\r')
print()

al = np.linspace(0, 2.2, 100)
ax.plot(al, al * kR, 'k--', lw=1.2, zorder=5, label=r'$\lambda_2 = \alpha k_R$')
ax.fill_between(al, al * kR, 3.0, alpha=0.07, color='#1a9850')
ax.fill_between(al, 0, al * kR, alpha=0.07, color='#d73027')
ax.scatter([], [], marker='o', facecolor='#1a9850', edgecolor='#1a9850',
           s=14, label='stable')
ax.scatter([], [], marker='x', color='#d73027', s=14, label='unstable')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\lambda_2$')
ax.set_title('(b)', loc='left')
ax.set_xlim(0, 2.05)
# the boundary lambda_2 = alpha*kR only reaches 2 over alpha in [0,2]; cropping
# just above it keeps both phases visible instead of a sliver of red
ax.set_ylim(0, max(v[1] for v in lam2_of.values()) * 1.06)
# green points fill the upper-left and red the lower-right, so no interior
# corner is free: put the legend outside, under the panel.
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3,
          frameon=False, handlelength=1.2, handletextpad=0.4,
          columnspacing=0.9, borderpad=0.2)
ax.grid(True, alpha=0.3, lw=0.5)

plt.subplots_adjust(left=0.20, right=0.97, top=0.90, bottom=0.20, hspace=0.75)
plt.savefig('Figure3.png', dpi=300)
plt.show()
print(f"\nDone in {time.time()-start:.0f}s")
