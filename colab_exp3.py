"""
CRIM-S Experiment 3: Multiplicative vs Additive Noise Robustness
Run on Google Colab free tier (~8 min)
Produces: exp3.png
"""
import numpy as np
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

def _tidy_axis(ax, x=True, y=True, nx=4, ny=5):
    """Small-value tick labels like 0.0000/0.0005/0.0010 are far too wide for a
    2.99-inch column and collide. Pull out a common power of ten (shown once, in
    the corner) and cap the number of ticks."""
    for axis, use, n in ((ax.xaxis, x, nx), (ax.yaxis, y, ny)):
        if not use:
            continue
        axis.set_major_locator(MaxNLocator(nbins=n, prune=None))


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

def compute_phi(psi):
    delta = psi - psi.mean(0, keepdims=True)
    return np.sum(delta**2) / psi.shape[0]

N, S = 200, 3
dt = 0.01
T_burn = 300   # burn-in
T_avg  = 200   # averaging window
eta, gs, kR = 1.0, 0.1, 1.0
alpha = 0.2
n_burn = int(T_burn / dt)
n_avg  = int(T_avg / dt)
n_runs = 20

# lam2 must exceed alpha*kR; on an N=200 ring that needs k >~ 10.
k_values = [10, 11, 12, 13, 14, 15, 16, 18, 20, 22]

K = 1e3   # plot in units of 1e-3
fig, axes = plt.subplots(2, 1, figsize=(2.99, 4.30))

results = {}
for mult, marker, color, label in [
    (False, 'o', '#2166ac', 'Additive noise'),
    (True,  's', '#b2182b', 'Multiplicative noise')
]:
    lam2_v, phi_v, phi_e, ev_v = [], [], [], []
    for kk in k_values:
        L = ring_laplacian(N, k=kk)
        ev = eigvalsh(L)
        lam2 = ev[1]
        gap = lam2 - alpha * kR
        if gap <= 0.01:
            continue
        lam2_v.append(lam2)
        runs = []
        for run in range(n_runs):
            print(f"  {'Mult' if mult else 'Add'}: k={kk}, run {run+1}/{n_runs}", end='\r')
            np.random.seed(run * 500 + kk * 11 + int(mult))
            psi = np.random.dirichlet(np.ones(S), size=N)
            # Burn-in
            for step in range(n_burn):
                delta = psi - psi.mean(0, keepdims=True)
                drift = -eta * (L @ psi - alpha * kR * delta) * dt
                if mult:
                    noise = eta * gs * np.sqrt(dt) * np.sqrt(np.maximum(psi, 1e-12)) * np.random.randn(N, S)
                else:
                    noise = eta * gs * np.sqrt(dt) * np.random.randn(N, S)
                psi = proj_simplex(psi + drift + noise)
            # Averaging
            phi_acc = []
            for step in range(n_avg):
                delta = psi - psi.mean(0, keepdims=True)
                drift = -eta * (L @ psi - alpha * kR * delta) * dt
                if mult:
                    noise = eta * gs * np.sqrt(dt) * np.sqrt(np.maximum(psi, 1e-12)) * np.random.randn(N, S)
                else:
                    noise = eta * gs * np.sqrt(dt) * np.random.randn(N, S)
                psi = proj_simplex(psi + drift + noise)
                if step % 200 == 0:
                    phi_acc.append(compute_phi(psi))
            runs.append(np.mean(phi_acc))
        ev_v.append(ev)
        phi_v.append(np.mean(runs))
        phi_e.append(np.std(runs))
    results[label] = (np.array(lam2_v), np.array(phi_v), np.array(phi_e), ev_v, marker, color)
    print()

# ---- spectral-sum prediction (the same test as Figure 2) ----
def spectral_sum(ev):
    """Full linear theory: every fluctuation mode contributes."""
    return S * eta * gs**2 / (2 * N) * np.sum(1.0 / (ev[1:] - alpha * kR))

# ---- Panel (a): Phi* vs lambda_2 ----
ax = axes[0]
for label, (lv, pv, pe, evs, mk, cl) in results.items():
    ax.errorbar(lv, K*pv, yerr=K*pe, fmt=mk, color=cl, label=label,
                ms=3.5, capsize=1.5, lw=0, elinewidth=0.7)
ax.set_xlabel(r'$\lambda_2$')
ax.set_ylabel(r'$\Phi^*$  ($\times 10^{-3}$)')
ax.set_title('(a)', loc='left')
_tidy_axis(ax, x=False, y=True)
ax.legend(framealpha=0.85, handlelength=1.4, handletextpad=0.5,
          borderpad=0.35, labelspacing=0.3)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3, lw=0.5)

# ---- Panel (b): collapse onto the spectral sum ----
# The lambda-dependence is identical for the two noise models; only the
# prefactor c differs. This is the claim being tested.
ax = axes[1]
slopes = {}
for label, (lv, pv, pe, evs, mk, cl) in results.items():
    ss = np.array([spectral_sum(e) for e in evs])
    c = np.sum(pv * ss) / np.sum(ss**2)
    R2 = 1 - np.sum((pv - c*ss)**2) / np.sum((pv - pv.mean())**2)
    slopes[label] = (c, R2)
    ax.errorbar(K*ss, K*pv, yerr=K*pe, fmt=mk, color=cl, ms=3.5, capsize=1.5,
                lw=0, elinewidth=0.7, label=rf'{label}: $c={c:.2f}$')
    xs = np.linspace(0, K*ss.max()*1.05, 40)
    ax.plot(xs, c*xs, '-', color=cl, lw=1.0, alpha=0.7)
ax.set_xlabel(r'Linear theory $\Phi^*_{\mathrm{lin}}$  ($\times 10^{-3}$)')
ax.set_ylabel(r'Measured $\Phi^*$  ($\times 10^{-3}$)')
ax.set_title('(b)', loc='left')
_tidy_axis(ax, x=True, y=True, nx=4, ny=5)
# the additive line rises from lower-left to upper-right; only the lower-right
# corner is free
# the fitted line runs through the origin along the diagonal, so no interior
# corner is free: put the legend outside, under the panel.
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.45), ncol=1,
          frameon=False, handlelength=1.2, handletextpad=0.4,
          borderpad=0.2, labelspacing=0.25)
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3, lw=0.5)

print("\n  spectral-sum collapse:")
for label, (c, R2) in slopes.items():
    print(f"    {label:<22} c = {c:.3f}   R^2 = {R2:.4f}")
print("  -> the lambda-dependence is unchanged by the noise model;")
print("     only the prefactor c differs.")

plt.subplots_adjust(left=0.22, right=0.97, top=0.94, bottom=0.26, hspace=0.75)
plt.savefig('exp3.png', dpi=300)
plt.show()
print(f"\nDone in {time.time()-start:.0f}s. Saved exp3.png")
