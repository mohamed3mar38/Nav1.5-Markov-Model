"""
calibrate_mexiletine.py
-----------------------
Least-squares grid search for mexiletine binding parameters.

Calibrates koff and kon for mexiletine by reproducing the
use-dependent block data of Wang et al. (2015), PLOS ONE 10(6):e0128653.

Target protocol (Wang et al. Methods, "Use/Frequency-dependent blockade"):
  - Cells held at -120 mV
  - Pulsed to -20 mV at 1, 5, and 10 Hz
  - Pulse width: 50 ms at 1 Hz and 5 Hz; 25 ms at 10 Hz
  - Drug concentration: 300 uM (0.3 mM)

Target values (Wang et al. Results):
  - 61.89% use-dependent block at 5 Hz
  - 76.03% use-dependent block at 10 Hz

Note: the Wang et al. protocol uses -120 mV holding potential between
pulses, which is sufficient for complete recovery from inactivation.
This means intrinsic channel rundown (ICR) is zero under this protocol,
so the naïve and corrected block metrics are equivalent for calibration.
The fitted parameters (koff, kon) therefore directly reflect drug binding
kinetics, independent of the ICR correction applied in the main AP-driven
simulations.

Note on concentration: the calibration is performed at 300 uM as reported
in the source study. For the main AP-driven simulations the drug is applied
at a lower therapeutic concentration (5 uM); this requires re-evaluation
of the block level under the AP protocol but does not invalidate the fitted
rate constants, which are concentration-independent.
"""

import json
import numpy as np
from scipy.linalg import expm

from step1_markov_core import steady_state
from step3_drug_model import build_generator, IDX


# ---------------------------------------------------------------------------
# Square-pulse block at a single frequency
# ---------------------------------------------------------------------------

def _block_at_freq(freq_hz, koff, kon, conc,
                   v_hold=-120.0, v_test=-20.0,
                   n_pulses=150):
    """
    Naïve use-dependent block (%) at given frequency under the Wang protocol.

    Returns (1 - peak[last] / peak[0]) x 100.
    """
    pulse_width = 25.0 if freq_hz >= 10.0 else 50.0
    period      = 1000.0 / freq_hz
    interpulse  = period - pulse_width

    A_test = build_generator(v_test, koff, kon, kon, conc)
    A_hold = build_generator(v_hold, koff, kon, kon, conc)
    Texp_test = expm(A_test * pulse_width)
    Texp_hold = expm(A_hold * interpulse)

    # Dense sub-stepping during the test pulse to capture activation peak
    n_sub    = 25
    Texp_sub = expm(A_test * (pulse_width / n_sub))

    u_hold, _ = steady_state(v_hold)
    u = np.zeros(18)
    u[0:9] = u_hold

    peaks = np.zeros(n_pulses)
    for p in range(n_pulses):
        u_pulse = u.copy()
        peak    = u_pulse[IDX['O']]
        for _ in range(n_sub):
            u_pulse = Texp_sub @ u_pulse
            if u_pulse[IDX['O']] > peak:
                peak = u_pulse[IDX['O']]
        peaks[p] = peak
        u = Texp_test @ u
        u = Texp_hold @ u

    return (1.0 - peaks[-1] / peaks[0]) * 100.0


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(conc=300.0, n_pulses=150):
    """
    Least-squares grid search over (koff, kon).

    Returns (best_fit_dict, full_results_list).
    """
    TARGET_5HZ  = 61.89  # %
    TARGET_10HZ = 76.03  # %

    koff_grid = [0.0010, 0.0015, 0.0020, 0.0025, 0.0030,
                 0.0035, 0.0040, 0.0050, 0.0060, 0.0080]
    kon_grid  = [0.00010, 0.00020, 0.00030, 0.00040, 0.00050,
                 0.00070, 0.00100]

    best    = None
    results = []

    for koff in koff_grid:
        for kon in kon_grid:
            b5  = _block_at_freq(5.0,  koff, kon, conc, n_pulses=n_pulses)
            b10 = _block_at_freq(10.0, koff, kon, conc, n_pulses=n_pulses)
            sse = (b5 - TARGET_5HZ) ** 2 + (b10 - TARGET_10HZ) ** 2
            row = {
                'koff': float(koff),
                'kon':  float(kon),
                'b5hz': round(float(b5),  2),
                'b10hz': round(float(b10), 2),
                'sse':  round(float(sse),  4),
            }
            results.append(row)
            if best is None or sse < best['sse']:
                best = row

    return best, results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Mexiletine least-squares calibration vs Wang et al. (2015)")
    print("  Target:  61.89% @5 Hz,  76.03% @10 Hz  (conc = 300 uM)\n")

    best, results = grid_search(conc=300.0, n_pulses=100)

    print(f"  Best fit:  koff={best['koff']:.4f} ms-1,  kon={best['kon']:.5f} uM-1 ms-1")
    print(f"  Predicted: {best['b5hz']:.2f}% @5 Hz,  {best['b10hz']:.2f}% @10 Hz")
    print(f"  Residuals: {best['b5hz']-61.89:+.2f} pp @5 Hz,  "
          f"{best['b10hz']-76.03:+.2f} pp @10 Hz")

    out = {
        'target_5hz':        61.89,
        'target_10hz':       76.03,
        'calibration_conc_uM': 300.0,
        'best_fit':          best,
        'residual_5hz_pp':   round(best['b5hz']  - 61.89, 2),
        'residual_10hz_pp':  round(best['b10hz'] - 76.03, 2),
        'full_grid':         results,
    }
    with open('mexiletine_calibration_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved mexiletine_calibration_results.json")
