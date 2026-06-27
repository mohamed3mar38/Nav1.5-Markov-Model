"""
Step 7: State-dependent unbinding (koff_open vs koff_inactivated) proof-of-concept.

Tests whether splitting koff into a fast open-state rate and a slow
inactivated-state rate can break the safety-efficacy tradeoff identified
with single-koff kinetics in step6/corrected_data.json.

Design choice (stated explicitly, not hidden):
  koff_vec is defined for ALL 9 unbound-mirrored states, not just the ones
  that can bind drug directly. Open (O) gets koff_open. Every other state
  (P,Q,R,S,T,U,V,W -- i.e. all closed/closed-inactivated AND the three
  fast/deep-inactivated states) gets koff_inact. This is a deliberate
  modeling choice: closed states are reached via the inactivation-recovery
  pathway, not the open-channel pathway, so they are grouped with the
  "inactivated" family for unbinding purposes.

  This design guarantees that when koff_open == koff_inact == koff, the
  model reduces EXACTLY to step6's single-koff model (regression-testable),
  and -- importantly -- it never assigns koff=0 to any bound state, which
  would trap drug permanently (the bug to avoid).

Everything else (gating_block, steady_state, voltage_at, apd_for_freq,
sampling/peak-detection protocol) is imported unchanged from step1/step5/step6.
"""
import numpy as np
from scipy.integrate import solve_ivp
from step1_markov_core import steady_state
from step5_ap_waveform import voltage_at, apd_for_freq
from step6_full_coupled import gating_block, IDX


def rhs_state_dependent(t, u, freq_hz, n_pulses, koff_open, koff_inact,
                         kon_open, kon_inact, conc, apd_override):
    V = voltage_at(t, freq_hz, n_pulses, apd=apd_override)
    Ag = gating_block(V)
    du = np.zeros(18)
    du[0:9] = Ag @ u[0:9]
    du[9:18] = Ag @ u[9:18]

    kon_vec = np.zeros(9)
    kon_vec[IDX['O']] = kon_open
    kon_vec[IDX['U']] = kon_inact
    kon_vec[IDX['V']] = kon_inact
    kon_vec[IDX['W']] = kon_inact

    # koff_vec: EVERY state gets a real (non-zero) unbinding rate.
    # O -> koff_open (fast). Everything else -> koff_inact (slow).
    koff_vec = np.full(9, koff_inact)
    koff_vec[IDX['O']] = koff_open

    for i in range(9):
        konc = kon_vec[i] * conc
        flux = konc * u[i] - koff_vec[i] * u[i + 9]
        du[i] -= flux
        du[i + 9] += flux
    return du


def run_pulse_train_sd(freq_hz, koff_open, koff_inact, kon_open, kon_inact, conc,
                        n_pulses=40, apd_override=None, t_eval_per_pulse=60):
    """Identical structure/sampling protocol to step6.run_pulse_train."""
    period = 1000.0 / freq_hz
    apd = apd_override if apd_override is not None else apd_for_freq(freq_hz)

    u0 = np.zeros(18)
    u_rest, _ = steady_state(-85.0)
    u0[0:9] = u_rest

    t_span = (0, n_pulses * period)
    sol = solve_ivp(rhs_state_dependent, t_span, u0, method='LSODA',
                     args=(freq_hz, n_pulses, koff_open, koff_inact, kon_open, kon_inact, conc, apd),
                     max_step=min(1.0, apd / 20), rtol=1e-6, atol=1e-9,
                     dense_output=True)

    t_list = []
    for p in range(n_pulses):
        t0 = p * period
        t_list.append(np.linspace(t0, t0 + 4.0, 200))
        t_list.append(np.linspace(t0 + 4.0, t0 + period, t_eval_per_pulse))
    t_eval = np.unique(np.concatenate(t_list))
    t_eval = t_eval[t_eval <= n_pulses * period]
    U = sol.sol(t_eval)

    O_unbound = U[0]
    peaks = np.zeros(n_pulses)
    for p in range(n_pulses):
        t0, t1 = p * period, p * period + min(apd, period)
        mask = (t_eval >= t0) & (t_eval <= t1)
        peaks[p] = O_unbound[mask].max() if mask.any() else 0.0
    return peaks, t_eval, U


def drug_specific_block_sd(freq, koff_open, koff_inact, kon, conc, n_pulses=20):
    """Drug-specific block (vs no-drug control at same freq), state-dependent koff."""
    from step6_full_coupled import run_pulse_train
    peaks_drug, _, _ = run_pulse_train_sd(freq, koff_open, koff_inact, kon, kon, conc, n_pulses=n_pulses)
    # no-drug control is identical regardless of koff choice (conc=0 means no binding at all)
    peaks_nodrug, _, _ = run_pulse_train(freq, koff_inact, kon, kon, 0.0, n_pulses=n_pulses)
    return (1 - peaks_drug[-1] / peaks_nodrug[-1]) * 100


def run_full_sweep_and_save(out_path='sd_sweep_final.json', n_pulses=20):
    """Regenerates: (1) regression test vs step6, (2) full koff_open x koff_inact
    sweep, (3) best-gap point, (4) frequency curve at the best point vs the
    single-koff baseline. Saves everything to JSON for figure-building."""
    import json
    from step6_full_coupled import run_pulse_train

    out = {}

    # --- 1. Regression test ---
    koff, kon_o, kon_i, conc = 0.01216, 0.0149, 0.0149, 1.0
    reg = []
    for freq in [2.0, 8.0]:
        p_orig, _, _ = run_pulse_train(freq, koff, kon_o, kon_i, conc, n_pulses=30)
        p_new, _, _ = run_pulse_train_sd(freq, koff, koff, kon_o, kon_i, conc, n_pulses=30)
        reg.append({'freq': freq, 'max_abs_diff': float(np.max(np.abs(p_orig - p_new)))})
    out['regression_test'] = reg

    # --- 2. koff_open x koff_inact sweep (conc=100uM, kon=0.01, near-saturating) ---
    sweep = []
    for koff_open in [0.02, 0.1, 1.0, 10.0, 50.0, 100.0]:
        for koff_inact in [0.0005, 0.001, 0.003, 0.005, 0.008, 0.012]:
            b2 = drug_specific_block_sd(2.0, koff_open, koff_inact, 0.01, 100.0, n_pulses=n_pulses)
            b8 = drug_specific_block_sd(8.0, koff_open, koff_inact, 0.01, 100.0, n_pulses=n_pulses)
            sweep.append({'koff_open': koff_open, 'koff_inact': koff_inact,
                           'b2': round(b2, 2), 'b8': round(b8, 2), 'gap': round(b8 - b2, 2),
                           'meets_strict': bool(b2 < 10 and b8 > 80)})
    out['sweep'] = sweep
    best = max(sweep, key=lambda r: r['gap'])
    out['best_gap_point'] = best
    out['any_meets_strict'] = any(r['meets_strict'] for r in sweep)

    # --- 3. Frequency curve at best point vs single-koff baseline ---
    freqs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    curve_sd, curve_single = [], []
    ko, ki = best['koff_open'], best['koff_inact']
    for f in freqs:
        b_sd = drug_specific_block_sd(f, ko, ki, 0.01, 100.0, n_pulses=n_pulses)
        b_single = drug_specific_block_sd(f, ki, ki, 0.01, 100.0, n_pulses=n_pulses)  # ko=ki -> single-koff
        curve_sd.append(round(b_sd, 2))
        curve_single.append(round(b_single, 2))
    out['frequency_curve'] = {'freqs': freqs, 'state_dependent': curve_sd, 'single_koff_baseline': curve_single}

    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    from step6_full_coupled import run_pulse_train

    print("=== REGRESSION TEST: koff_open == koff_inact should reproduce step6 exactly ===")
    koff, kon_open, kon_inact, conc = 0.01216, 0.0149, 0.0149, 1.0
    for freq in [2.0, 8.0]:
        peaks_orig, _, _ = run_pulse_train(freq, koff, kon_open, kon_inact, conc, n_pulses=30)
        peaks_new, _, _ = run_pulse_train_sd(freq, koff, koff, kon_open, kon_inact, conc, n_pulses=30)
        diff = np.max(np.abs(peaks_orig - peaks_new))
        print(f"freq={freq}Hz  max|diff| = {diff:.3e}  "
              f"(orig last peak={peaks_orig[-1]:.6f}, new last peak={peaks_new[-1]:.6f})")
