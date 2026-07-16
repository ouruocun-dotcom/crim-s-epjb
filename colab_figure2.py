"""
CRIM-S Figure 2 -- steady-state diversity vs. the Laplacian spectrum.

Runs a REAL simulation of the CRIM-S SDE on ring, Erdos-Renyi and
small-world networks, and tests the spectral prediction

    Phi* = c * [S eta gamma^2 sigma^2 / (2N)] * sum_{i>=2} 1/(lambda_i - alpha k_R)

(a) Phi* vs lambda_2. The single-mode form 1/(lambda_2 - alpha k_R) does NOT
    describe the data: many Laplacian modes contribute.
(b) Phi* vs the full spectral sum. All topologies collapse onto a straight
    line of slope c < 1. The deficit c < 1 is the variance suppression due to
    the simplex projection; the same run WITHOUT projection gives c ~ 1.

Runtime ~15-25 min on Colab free tier. Produces Figure2.png
"""
import numpy as np
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import time

# --- journal-ready sizing (sn-jnl: column 2.99in, text 6.30in) ---
mpl.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 6.8,
    'lines.linewidth': 1.2, 'axes.linewidth': 0.7,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'savefig.dpi': 300, 'figure.dpi': 110,
})

def _tidy_axis(ax, x=True, y=True, nx=4, ny=5):
    """Small-value tick labels like 0.0000/0.0005/0.0010 are far too wide for a
    2.99-inch column and collide. Pull out a common power of ten (shown once, in
    the corner) and cap the number of ticks."""
    for axis, use, n in ((ax.xaxis, x, nx), (ax.yaxis, y, ny)):
        if not use:
            continue
        # the data are already in units of 1e-3 (see K below), so the axis label
        # carries the exponent. Do NOT let a formatter add a second one.
        axis.set_major_locator(MaxNLocator(nbins=n, prune=None))



start = time.time()

# ---------------- parameters (match the main text) ----------------
N, S = 200, 3
eta, gs, kR = 1.0, 0.1, 1.0        # gs = gamma * sigma
alpha = 0.3
dt = 0.01
T_burn, T_avg = 300, 200
n_runs = 20

# ---------------- networks ----------------
def ring_laplacian(N, k):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(1, k+1):
            A[i, (i+j) % N] = 1
            A[i, (i-j) % N] = 1
    return np.diag(A.sum(1)) - A

def er_laplacian(N, p, seed):
    r = np.random.RandomState(seed)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            if r.rand() < p:
                A[i, j] = A[j, i] = 1
    for i in range(N-1):
        if A[i, i+1] == 0:
            A[i, i+1] = A[i+1, i] = 1
    return np.diag(A.sum(1)) - A

def sw_laplacian(N, k, p_rewire, seed):
    r = np.random.RandomState(seed)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(1, k+1):
            A[i, (i+j) % N] = 1
            A[i, (i-j) % N] = 1
    for i in range(N):
        for j in range(1, k+1):
            if r.rand() < p_rewire:
                old = (i+j) % N
                new = r.randint(N)
                while new == i or A[i, new] == 1:
                    new = r.randint(N)
                A[i, old] = A[old, i] = 0
                A[i, new] = A[new, i] = 1
    return np.diag(A.sum(1)) - A

# ---------------- CRIM-S dynamics ----------------
def proj_simplex(X):
    u = np.sort(X, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    idx = np.arange(1, X.shape[1]+1)[None, :]
    rho = (X.shape[1]-1) - np.argmax((u*idx > cssv)[:, ::-1], axis=1)
    theta = cssv[np.arange(X.shape[0]), rho] / (rho + 1.0)
    return np.maximum(X - theta[:, None], 0)

def compute_phi(psi):
    delta = psi - psi.mean(0, keepdims=True)
    return np.sum(delta**2) / psi.shape[0]

def steady_state(L, seed, constrained=True):
    rs = np.random.RandomState(seed)
    psi = rs.dirichlet(np.ones(S), size=N)
    for _ in range(int(T_burn/dt)):
        d = psi - psi.mean(0, keepdims=True)
        psi = psi - eta*(L @ psi - alpha*kR*d)*dt \
                  + eta*gs*np.sqrt(dt)*rs.randn(N, S)
        if constrained:
            psi = proj_simplex(psi)
    acc = []
    for s in range(int(T_avg/dt)):
        d = psi - psi.mean(0, keepdims=True)
        psi = psi - eta*(L @ psi - alpha*kR*d)*dt \
                  + eta*gs*np.sqrt(dt)*rs.randn(N, S)
        if constrained:
            psi = proj_simplex(psi)
        if s % 200 == 0:
            acc.append(compute_phi(psi))
    return np.mean(acc)

def spectral_sum(ev):
    """Full linear-theory prediction: every fluctuation mode contributes."""
    return S*eta*gs**2 / (2*N) * np.sum(1.0 / (ev[1:] - alpha*kR))

def single_mode(lam2):
    """Slowest mode only; valid when lambda_2 << lambda_3."""
    return S*(eta*gs)**2 / (2*N*eta*(lam2 - alpha*kR))

# ---------------- data ----------------
configs  = [('Ring', ring_laplacian(N, k)) for k in [4, 6, 8, 10, 14, 18]]
configs += [('ER',   er_laplacian(N, p, int(p*1000))) for p in [0.03, 0.04, 0.05, 0.07, 0.10]]
configs += [('SW',   sw_laplacian(N, k, 0.15, 6000+k)) for k in [4, 6, 8, 10, 14]]

data = {'Ring': [], 'ER': [], 'SW': []}
print("Running simulations...")
for name, L in configs:
    ev = eigvalsh(L)
    lam2 = ev[1]
    if lam2 <= alpha*kR + 0.15:
        continue
    runs = [steady_state(L, seed=1000*r + int(lam2*100)) for r in range(n_runs)]
    data[name].append(dict(lam2=lam2, lam3=ev[2],
                           phi=np.mean(runs), err=np.std(runs),
                           ssum=spectral_sum(ev), smode=single_mode(lam2)))
    print(f"  {name:<5} lam2={lam2:6.3f}  Phi*={np.mean(runs):.5f}  "
          f"spec-sum={spectral_sum(ev):.5f}")

print("\nControl (no simplex projection)...")
# the control MUST sit in the stable regime (lam2 > alpha*kR = 0.3),
# otherwise the unconstrained system diverges and c is meaningless.
L_ctl = ring_laplacian(N, 16)          # lam2 = 1.46
ev_ctl = eigvalsh(L_ctl)
assert ev_ctl[1] > alpha * kR, (
    f"control network is unstable (lam2={ev_ctl[1]:.3f} < alpha*kR={alpha*kR:.3f}); "
    "the unconstrained run would diverge and c would be meaningless")
phi_unc = np.mean([steady_state(L_ctl, seed=7*r, constrained=False) for r in range(n_runs)])
c_unc = phi_unc / spectral_sum(ev_ctl)
print(f"  lam2 = {ev_ctl[1]:.3f} > alpha*kR = {alpha*kR:.2f}  (stable regime)")
print(f"  unconstrained: c = {c_unc:.3f}   (linear theory predicts c = 1)")

# ---------------- fit ----------------
allpts = [d for v in data.values() for d in v]
phi   = np.array([d['phi']   for d in allpts])
ssum  = np.array([d['ssum']  for d in allpts])
smode = np.array([d['smode'] for d in allpts])

c   = np.sum(phi*ssum)  / np.sum(ssum**2)
R2  = 1 - np.sum((phi - c*ssum)**2)  / np.sum((phi - phi.mean())**2)
cm  = np.sum(phi*smode) / np.sum(smode**2)
R2m = 1 - np.sum((phi - cm*smode)**2) / np.sum((phi - phi.mean())**2)

print(f"\n  spectral sum : c = {c:.3f}   R^2 = {R2:.4f}")
print(f"  single mode  : c = {cm:.2f}    R^2 = {R2m:.4f}   (wrong shape)")
print(f"  projection suppresses Phi* by {100*(1 - c/c_unc):.0f}% "
      f"relative to the unconstrained system")

# ---------------- plot ----------------
style = {'Ring': ('o', '#2166ac'), 'ER': ('s', '#e08214'), 'SW': ('^', '#1a9850')}
K = 1e3   # plot everything in units of 1e-3: no offset text, no long ticks
fig, axes = plt.subplots(2, 1, figsize=(2.99, 4.30))

ax = axes[0]
for name in ['Ring', 'ER', 'SW']:
    if not data[name]:
        continue
    mk, cl = style[name]
    ax.errorbar([d['lam2'] for d in data[name]],
                [K*d['phi']  for d in data[name]],
                yerr=[K*d['err'] for d in data[name]],
                fmt=mk, color=cl, ms=3.5, capsize=1.5, lw=0,
                elinewidth=0.7, label=name)
lg = np.linspace(min(d['lam2'] for d in allpts), max(d['lam2'] for d in allpts), 200)
ax.plot(lg, K*single_mode(lg), 'k--', lw=1.0, alpha=0.6, label='single mode')
ax.set_xlabel(r'Algebraic connectivity $\lambda_2$')
ax.set_ylabel(r'$\Phi^*$  ($\times 10^{-3}$)')
ax.set_title('(a)', loc='left')
_tidy_axis(ax, x=False, y=True)
ax.legend(framealpha=0.85, handlelength=1.4, handletextpad=0.5,
          borderpad=0.35, labelspacing=0.3)
ax.grid(alpha=0.3, lw=0.5)
ax.set_ylim(bottom=0)

ax = axes[1]
for name in ['Ring', 'ER', 'SW']:
    if not data[name]:
        continue
    mk, cl = style[name]
    ax.errorbar([K*d['ssum'] for d in data[name]],
                [K*d['phi']  for d in data[name]],
                yerr=[K*d['err'] for d in data[name]],
                fmt=mk, color=cl, ms=3.5, capsize=1.5, lw=0,
                elinewidth=0.7, label=name)
xs = np.linspace(0, K*max(ssum)*1.05, 50)
ax.plot(xs, c*xs, 'k-',  lw=1.0, label=rf'slope $c={c:.2f}$')
ax.plot(xs, xs,   color='0.6', ls=':', lw=1.0, label=r'$c=1$ (unconstrained)')
ax.set_xlabel(r'Linear theory $\Phi^*_{\mathrm{lin}}$  ($\times 10^{-3}$)')
ax.set_ylabel(r'Measured $\Phi^*$  ($\times 10^{-3}$)')
ax.set_title('(b)', loc='left')
_tidy_axis(ax, x=True, y=True, nx=4, ny=5)
# the fitted line runs through the origin along the diagonal, so no interior
# corner is free: put the legend outside, under the panel.
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.45), ncol=2,
          frameon=False, handlelength=1.2, handletextpad=0.4,
          borderpad=0.2, labelspacing=0.25)
ax.grid(alpha=0.3, lw=0.5)
ax.set_xlim(left=0); ax.set_ylim(bottom=0)

plt.subplots_adjust(left=0.22, right=0.97, top=0.94, bottom=0.26, hspace=0.75)
plt.savefig('Figure2.png')
plt.show()
print(f"\nDone in {time.time()-start:.0f}s. Saved Figure2.png")
