"""
Circulation quantization wall vs gauge-potential remedy, full (alpha, R) plane
===============================================================================
Extends the worst-case quantization map (airfoil_circulation_errors.py) with
the gauge-potential encoding of Section gauge_reformulation as the remedy,
measured rather than assumed:

  (a) Phase encoding: the circulation Gamma = 4 pi R U sin(alpha) maps to the
      nearest flux quantum Gamma_q = 2 pi n, and the worst-case lift error
      |C_L - C_L^q|/C_L = |Gamma - Gamma_q|/Gamma forms the familiar wall
      over the aerodynamically relevant plane (analytical map, as before,
      with the angle of attack capped at 30 deg).

  (b) Gauge-potential encoding: the circulation enters as an Aharonov-Bohm
      flux Gamma = oint A . dl with A = grad Phi single-valued for any real
      Gamma.  The flux is MEASURED on the lattice with the paper's Peierls
      convention -- midpoint bond phases 1/2 (A_i + A_j) . (r_j - r_i)
      accumulated along a closed square loop enclosing the cylinder -- and
      compared against the exact Kutta value.  The loop follows lattice
      bonds of FIXED length h_loop = 3/512 (as on a production lattice,
      where the grid spacing does not scale with the airfoil), so the bond
      count M = 3 R / h_loop grows with R.  The measured lift error
      |C_L - C_L^gauge|/C_L = |Gamma_loop - Gamma|/Gamma is set by the
      loop resolution alone, (h_loop/R)^2: about 5e-5 at R = 0.1 down to
      2e-7 at R = 2, always four to six orders of magnitude below the wall
      of panel (a).  Two exact properties hold at any resolution: the
      uniform-flow part of the midpoint sum vanishes bitwise (Gamma = 0
      gives exactly zero), so the error carries no alpha dependence, and
      refining the loop lowers it as h_loop^2 (inset, NACA 0012 point).

Both panels span alpha in [0.5, 30] deg (moderate angles; the stationary
potential-flow construction is not exercised past the linear regime) and
R in [0.1, 2.0] (flux quantum number n = 2 R sin(alpha) up to 2).  Each
section is marked at the cylinder angle that carries its circulation, so a
cambered section sits at the alpha that reproduces its Gamma: the NACA 0012 at
its geometric 5 deg, the cambered NACA 2412 at 5 deg + 2.07 deg, and the
Joukowski airfoil, which carries camber-only lift at zero geometric angle,
Gamma = 4 pi a U sin(beta), at alpha ~ 2.6 deg.

Output (fig/): circulation_gauge_remedy_map.pdf
Run: python circulation_gauge_remedy_map.py
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# type 42: publishers reject type 3 (no scaling, no text extraction)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.colors import LogNorm
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401  -- puts every code/ sub-package on sys.path


U_INF = 1.0

# map resolution and loop protocol
N_ALPHA, N_R = 120, 120
ALPHA_LO, ALPHA_HI = 0.5, 30.0     # deg, moderate angles only
R_LO, R_HI = 0.1, 2.0
H_LOOP = 3.0 / 512                 # fixed bond length: R = 1 <-> 512 bonds/side

# ---------------------------------------------------------------- physics
def gauge_field(X, Y, R, alpha, Gamma):
    """A = grad Phi for uniform flow U at angle alpha past a cylinder of
    radius R with circulation Gamma.  Single-valued Cartesian field: the
    multi-valued Gamma th/(2 pi) term enters only through its gradient
    Gamma/(2 pi r) in the azimuthal component."""
    r2 = X * X + Y * Y
    r = np.sqrt(r2) + 1e-15
    th = np.arctan2(Y, X)
    vr = U_INF * (1.0 - R * R / r2) * np.cos(th - alpha)
    vt = -U_INF * (1.0 + R * R / r2) * np.sin(th - alpha) \
        + Gamma / (2.0 * np.pi * r)
    ct, st = X / r, Y / r
    return vr * ct - vt * st, vr * st + vt * ct

def kutta_gamma(R, alpha):
    return 4.0 * np.pi * R * U_INF * np.sin(alpha)

def peierls_loop_flux(R, alpha, Gamma, d, M):
    """Midpoint bond-phase sum 1/2 (A_i + A_j) . (r_j - r_i) along a closed
    square loop of half-width d > R with M bonds per side -- the lattice
    Aharonov-Bohm flux measurement.  Never references the multi-valued Phi."""
    s = np.linspace(-d, d, M + 1)
    seg = [
        (s, -d * np.ones_like(s)),          # bottom, left -> right
        (d * np.ones_like(s), s),           # right,  bottom -> top
        (s[::-1], d * np.ones_like(s)),     # top,    right -> left
        (-d * np.ones_like(s), s[::-1]),    # left,   top -> bottom
    ]
    px = np.concatenate([seg[k][0] for k in range(4)])
    py = np.concatenate([seg[k][1] for k in range(4)])
    Px, Py = px[:-1], py[:-1]
    Qx, Qy = px[1:], py[1:]
    Ax, Ay = gauge_field(Px, Py, R, alpha, Gamma)
    Bx, By = gauge_field(Qx, Qy, R, alpha, Gamma)
    return float(np.sum(0.5 * (Ax + Bx) * (Qx - Px)
                        + 0.5 * (Ay + By) * (Qy - Py)))

def quantization_wall(R, alpha):
    """Worst-case lift error of the phase encoding: |Gamma - 2 pi n|/Gamma."""
    Gamma = kutta_gamma(R, alpha)
    n = np.round(Gamma / (2.0 * np.pi))
    return np.minimum(1.0, np.abs(Gamma - 2.0 * np.pi * n) / np.maximum(Gamma, 1e-12))

# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(here, '..', 'fig')
    os.makedirs(fig_dir, exist_ok=True)

    d = np.load(os.path.join(here, '..', 'data', 'laurent_coeffs.npz'), allow_pickle=True)
    # (marker, colour, cylinder angle carrying the section's circulation, R).
    # The NACA 2412's ideal angle is -2.07 deg, so its circulation equals a
    # symmetric cylinder's at 5 + 2.07 deg; the symmetric 0012 stays at 5 deg.
    # Radii from the geometry module, which now scales every section to unit
    # chord: R_joukowski in the cached npz is the unscaled 1.1011.
    import airfoil_hb_quantum as A
    _g = {n: A.naca_geometry(n)['R'] for n in ('NACA 0012', 'NACA 2412')}
    airfoils = dict(NACA0012=('o', '#185FA5', 5.0, float(_g['NACA 0012'])),
                    NACA2412=('o', '#0F6E56', 5.0 + 2.0685, float(_g['NACA 2412'])))
    camber = (float(A.joukowski_geometry()['R']),)

    alphas_d = np.linspace(ALPHA_LO, ALPHA_HI, N_ALPHA)
    R_list = np.linspace(R_LO, R_HI, N_R)
    AA, RR = np.meshgrid(alphas_d, R_list, indexing='ij')

    # --- panel (a): analytical quantization wall ---
    W = quantization_wall(RR, np.radians(AA))

    # --- panel (b): measured Peierls loop flux of the gauge encoding ---
    # fixed bond length (production lattice): M = 3R/h_loop grows with R
    E = np.zeros_like(W)
    M_used = np.zeros(len(R_list), dtype=int)
    for j, R in enumerate(R_list):
        M_j = max(8, int(round(3.0 * R / H_LOOP)))
        M_used[j] = M_j
        for i, a_deg in enumerate(alphas_d):
            al = np.radians(a_deg)
            G = kutta_gamma(R, al)
            G_loop = peierls_loop_flux(R, al, G, d=1.5 * R, M=M_j)
            E[i, j] = abs(G_loop - G) / G
    print(f"  plane sweep: {N_ALPHA} x {N_R} = {N_ALPHA * N_R} loop "
          f"experiments (fixed h_loop = {H_LOOP:.5f}, M = "
          f"{M_used.min()}..{M_used.max()}/side)  [{time.time() - t0:.0f}s]")
    print(f"  gauge lift error:  min {E.min():.2e}  median "
          f"{np.median(E):.2e}  max {E.max():.2e}")

    # loop convergence audit at the NACA 0012 point (console + inset data)
    R0 = airfoils['NACA0012'][3]
    G0 = kutta_gamma(R0, np.radians(5.0))
    conv_M = (128, 256, 512, 1024, 2048)
    conv_e = []
    print("  loop refinement at NACA 0012 (alpha = 5 deg):")
    ref = None
    for M in conv_M:
        e = abs(peierls_loop_flux(R0, np.radians(5.0), G0, 1.5 * R0, M) - G0) / G0
        conv_e.append(e)
        ratio = f"  ratio {ref / e:5.1f}" if ref else ""
        print(f"    M = {M:4d}: rel err {e:.3e}{ratio}")
        ref = e

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))
    X, Y = np.meshgrid(alphas_d, R_list, indexing='ij')

    # flux-quantum curves Gamma/2pi = n:  R = n / (2 sin alpha)
    def n_curve(n):
        a_cur = np.radians(np.linspace(ALPHA_LO, ALPHA_HI, 400))
        R_cur = n / (2.0 * np.sin(a_cur))
        m = (R_cur >= R_LO) & (R_cur <= R_HI)
        return np.degrees(a_cur)[m], R_cur[m]

    ax = axes[0]
    # clip low errors so the flux-quantum valleys read as solid light bands
    mesh_a = ax.pcolormesh(X, Y, np.clip(W, 2e-2, 1.0), cmap='Reds',
                           shading='nearest', norm=LogNorm(2e-2, 1.0),
                           rasterized=True)
    cs = ax.contour(X, Y, W, levels=[0.05, 0.10, 0.20, 0.30, 0.50],
                    colors='k', linewidths=0.5, alpha=0.55)
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
    for n_q, lw in ((0.5, 0.6), (1.0, 0.9), (1.5, 0.6)):
        a_cur, R_cur = n_curve(n_q)
        ax.plot(a_cur, R_cur, ':', color='k', lw=lw, alpha=0.65)
    a_lab = 27.0
    # white with a dark stroke: these sit on dark red in (a) and on a bright
    # corner of viridis in (b), so neither plain white nor plain black reads
    st = [pe.withStroke(linewidth=2.2, foreground='k')]
    ax.annotate(r'$n=1/2$', (a_lab, 0.5 / (2 * np.sin(np.radians(a_lab)))),
                xytext=(2, 4), textcoords='offset points', fontsize=7,
                color='w', path_effects=st)
    ax.annotate(r'$n=1$', (a_lab, 1.0 / (2 * np.sin(np.radians(a_lab)))),
                xytext=(2, 4), textcoords='offset points', fontsize=7,
                color='w', path_effects=st)
    ax.annotate(r'$n=3/2$', (a_lab, 1.5 / (2 * np.sin(np.radians(a_lab)))),
                xytext=(4, 9), textcoords='offset points', fontsize=7,
                color='w', path_effects=st)
    cb_a = fig.colorbar(mesh_a, ax=ax, pad=0.02,
                        ticks=[0.02, 0.05, 0.1, 0.3, 1.0])
    cb_a.ax.set_yticklabels(['0.02', '0.05', '0.1', '0.3', '1'], fontsize=8)
    cb_a.set_label(r'$|C_L - C_L^q|/C_L$')
    ax.set_title('Phase encoding: worst-case $C_L$ error', fontsize=10.5)
    ax.set_xlabel(r'angle of attack $\alpha$ (deg)')
    ax.set_ylabel(r'cylinder radius $R$')

    ax = axes[1]
    mesh = ax.pcolormesh(X, Y, E, cmap='viridis', shading='nearest',
                         norm=LogNorm(vmin=1.5e-7, vmax=2e-4), rasterized=True)
    cb_b = fig.colorbar(mesh, ax=ax, pad=0.02,
                        ticks=[1e-6, 1e-5, 1e-4])
    cb_b.ax.set_yticklabels([r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$'],
                            fontsize=8)
    cb_b.set_label(r'$|C_L - C_L^{\,\mathrm{gauge}}|/C_L$ (measured)')
    for n_q, lw in ((0.5, 0.6), (1.0, 0.9), (1.5, 0.6)):
        a_cur, R_cur = n_curve(n_q)
        ax.plot(a_cur, R_cur, ':', color='w', lw=lw, alpha=0.8)
    ax.annotate(r'$n=1$', (29.6, 1.0 / (2 * np.sin(np.radians(29.6)))),
                xytext=(-2, -12), textcoords='offset points', fontsize=7,
                color='w', ha='right',
                path_effects=[pe.withStroke(linewidth=2.2, foreground='k')])
    ax.annotate('loop-resolution floor '
                + r'$\propto(h_\mathrm{loop}/R)^2$' + '\n'
                + r'$2\times10^{-7}$ (large $R$) $\to 8\times10^{-5}$ (small $R$)',
                (0.97, 0.06), xycoords='axes fraction', ha='right',
                fontsize=8, color='w')
    # resolution is what lowers the floor: loop-refinement inset (NACA point)
    # white text throughout -- the inset sits on the dark viridis field
    ins = ax.inset_axes([0.135, 0.500, 0.355, 0.410])
    ins.loglog(conv_M, conv_e, '-o', color='#185FA5', lw=1.1, ms=3,
               zorder=3)
    ins.loglog(conv_M, conv_e[0] * (np.array(conv_M) / conv_M[0]) ** -2,
               ':', color='#999999', lw=1.0, label=r'$\propto M^{-2}$')
    ins.set_xlabel('bonds per side $M$', fontsize=7, color='w')
    ins.set_ylabel('lift error', fontsize=7, color='w')
    ins.set_title('loop refinement (NACA 0012)', fontsize=7, pad=3,
                  color='w')
    ins.legend(fontsize=6, loc='lower left')
    ins.get_legend().get_frame().set_facecolor('white')
    ins.get_legend().get_frame().set_edgecolor('none')
    ins.get_legend().get_frame().set_alpha(0.92)
    ins.set_xticks([128, 512, 2048])
    ins.set_xticklabels(['128', '512', '2048'], fontsize=6.5, color='w')
    ins.set_yticks([1e-8, 1e-7, 1e-6, 1e-5])
    ins.tick_params(labelsize=6.5, colors='w', which='both')
    ins.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ins.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ins.grid(True, alpha=0.35, which='both', color='w')
    ax.set_title('Gauge-potential encoding: measured $C_L$ error',
                 fontsize=10.5)
    ax.set_xlabel(r'angle of attack $\alpha$ (deg)')
    ax.set_ylabel(r'cylinder radius $R$')

    # airfoil operating points on both panels: each section at the cylinder
    # angle that reproduces its own circulation, so the cambered 2412 sits at
    # the equivalent 7.07 deg; the Joukowski carries camber-only lift at
    # alpha = 0, shown at the equivalent-camber point alpha ~ 2.6 deg
    for ax in axes:
        for name, (mk, col, a_op, R_op) in airfoils.items():
            ax.plot(a_op, R_op, mk, color=col, ms=7, mec='k', mew=0.7, zorder=5)
        ax.plot(2.6, camber[0], 's', color='#993C1D', ms=6,
                mec='k', mew=0.7, zorder=5)
    for ax, lbl_col in zip(axes, ('w', 'k')):
        # the three operating points now sit within 0.4 deg and 0.004 in R of
        # each other (all sections carry unit chord), so the labels are stacked
        # upward instead of ringed around the markers
        _st = [pe.withStroke(linewidth=2.2, foreground='k')]
        _arr = dict(arrowstyle='-', lw=0.6, color='w', shrinkA=1, shrinkB=3)
        ax.annotate('NACA 0012', (5.0, airfoils['NACA0012'][3]),
                    xytext=(0, 34), textcoords='offset points', fontsize=8,
                    ha='center', color='w', path_effects=_st, arrowprops=_arr)
        ax.annotate('NACA 2412', (5.0 + 2.0685, airfoils['NACA2412'][3]),
                    xytext=(10, 48), textcoords='offset points', fontsize=8,
                    ha='left', color='w', path_effects=_st, arrowprops=_arr)
        ax.annotate('Joukowski', (2.6, camber[0]), xytext=(2, 14),
                    textcoords='offset points', fontsize=8, ha='left',
                    color='w', path_effects=_st, arrowprops=_arr)

    # pad keeps the shared y-label clear of the page edge after the panel split
    fig.tight_layout(pad=2.0)
    out = os.path.join(fig_dir, 'circulation_gauge_remedy_map.pdf')
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"  -> {os.path.relpath(out)}")
    print(f"  total [{time.time() - t0:.0f}s]")

if __name__ == '__main__':
    main()
