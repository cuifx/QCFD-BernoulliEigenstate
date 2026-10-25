"""Print the airfoil error-table numbers at full precision.

The manuscript table (tab:airfoil_errors) quotes two rows:

    Free-particle error    ~3e-3   ~2e-3   ~2e-3
    H_B exact floor        4.9e-4  3.4e-4  3.4e-4

Both use the metric rel_err_zeta() of code/figures/airfoil_hb_quantum.py,
i.e. the mass-weighted relative error of |v| over the annulus
1.05 R < r < 1.5 on the mapped grid, free-particle branch at T = 0.01.

This driver re-runs exactly that pipeline (same grid, same geometry, same
metric) and prints every digit that comes out, so the table can quote the
measured value instead of a one-digit estimate.  It writes nothing.
"""
import os, sys, time

import numpy as np

_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_d, "_paths.py")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths  # noqa: E402,F401

from airfoil_hb_quantum import (run_airfoil, joukowski_geometry,           # noqa: E402
                                naca_geometry, TS)


def main():
    t0 = time.time()
    geoms = [joukowski_geometry(), naca_geometry('NACA 0012'),
             naca_geometry('NACA 2412')]
    res = {}
    for g in geoms:
        r = run_airfoil(g)
        res[g['name']] = r
        print('%s  R=%.6f  alpha_deg=%.6f  gamma=%.6f'
              % (g['name'], g['R'], np.degrees(g['alpha']), g['gamma']))
        for T, ex, ef in zip(TS, r['err_hb_exact'], r['err_free']):
            print('   T=%-6.3f  H_B(exact)=%.8e   free=%.8e' % (T, ex, ef))
        print()

    print('--- what the manuscript table needs (T = 0.01) ---')
    k = TS.index(0.01)
    print('%-12s %14s %14s' % ('section', 'free (T=0.01)', 'H_B floor (T=0.01)'))
    for name, r in res.items():
        print('%-12s %14.6e %14.6e' % (name, r['err_free'][k],
                                       r['err_hb_exact'][k]))
    print()
    print('--- same numbers at the table row ordering ---')
    order = ['Joukowski', 'NACA 0012', 'NACA 2412']
    print('free      : ' + ' & '.join('%.6e' % res[n]['err_free'][k] for n in order))
    print('H_B floor : ' + ' & '.join('%.6e' % res[n]['err_hb_exact'][k] for n in order))
    print()
    print('H_B floor spread over all T (is the floor flat?):')
    for n in order:
        v = np.array(res[n]['err_hb_exact'])
        print('   %-12s min=%.6e max=%.6e  (max/min=%.4f)'
              % (n, v.min(), v.max(), v.max() / v.min()))
    print()
    print('free-particle growth over T:')
    for n in order:
        v = np.array(res[n]['err_free'])
        print('   %-12s ' % n + ' '.join('%.4e' % x for x in v))
    print('[%.0f s]' % (time.time() - t0))


if __name__ == '__main__':
    main()
