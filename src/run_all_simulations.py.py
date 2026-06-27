"""
generate_corrected_data_FULL.py
--------------------------------
Complete, reproducible data-generation pipeline.

Runs all simulations required to produce corrected_data.json, which
contains every numerical result cited in the paper:

  intrinsic_rundown   — ICR(f) at each key frequency (no drug)
  drug_specific       — corrected block for all four drugs
  naive               — naïve (uncorrected) block for comparison
  koff_sweep          — [koff, tau, B@2Hz, B@8Hz] across koff grid
  selectivity_ratio   — subset of koff_sweep with B8/B2 ratio
  sensitivity_kon     — block at optimal koff across 4x range of kon

Usage
-----
    python3 generate_corrected_data_FULL.py

Output
------
    corrected_data_REGENERATED.json   (should match corrected_data.json exactly)

Dependencies
------------
    step1_markov_core.py
    step3_drug_model.py
    step5_ap_waveform.py
    step6_full_coupled.py
"""

import json
import numpy as np
from step6_full_coupled import run_pulse_train


# ---------------------------------------------------------------------------
# Drug parameters (Table 1 of the paper)
# ---------------------------------------------------------------------------

DRUGS = {
    'Flecainide': dict(
        koff=0.01216, kon=0.0149, conc=1.0,
        source="Ramos & O'Leary (2004) J Physiol 560:37-49"),
    'Lidocaine': dict(
        koff=0.200, kon=0.00033, conc=2.0,
        source="Bennett et al. (1995) Circ Res 77:584-592"),
    'Ranolazine': dict(
        koff=0.006, kon=0.0011, conc=7.5,
        source="Nesterenko et al. (2011) Am J Physiol H1615"),
    'Mexiletine': dict(
        koff=0.0030, kon=0.0004, conc=5.0,
        source="Calibrated vs Wang et al. (2015) PLOS ONE"),
}

KEY_FREQS  = [1.0, 2.0, 4.0, 6.0, 8.0]
N_PULSES   = 30
KOFF_SWEEP = [0.0005, 0.001, 0.002, 0.003, 0.0035,
               0.004, 0.005, 0.008, 0.012, 0.02]
KOFF_RATIO_SUBSET = {0.001, 0.002, 0.003, 0.005, 0.008}
KON_SENSITIVITY   = [0.005, 0.010, 0.020]   # uM^-1 ms^-1
KOFF_OPTIMAL      = 0.0035                    # ms^-1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def naive_block(peaks):
    """1 - peak[last] / peak[first] (contaminated by ICR)."""
    return float((1.0 - peaks[-1] / peaks[0]) * 100.0)


def corrected_block(peaks_drug, peaks_nodrug):
    """1 - peak_drug[last] / peak_nodrug[last] (drug-specific)."""
    return float((1.0 - peaks_drug[-1] / peaks_nodrug[-1]) * 100.0)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    results = {
        'intrinsic_rundown': {},
        'drug_specific':     {},
        'naive':             {},
        'koff_sweep':        [],
        'selectivity_ratio': [],
        'sensitivity_kon':   {},
    }

    # ── Intrinsic channel rundown (no drug) ──────────────────────────────
    # Any koff/kon can be used since conc=0 means no drug binds.
    print("Step 1/4: Intrinsic channel rundown (no drug) ...")
    nodrug_cache = {}
    for f in KEY_FREQS:
        peaks, _, _ = run_pulse_train(f, 0.01216, 0.0149, 0.0149, 0.0, n_pulses=N_PULSES)
        nodrug_cache[f] = peaks
        icr = round(naive_block(peaks), 2)
        results['intrinsic_rundown'][str(f)] = icr
        print(f"  {f} Hz  ICR = {icr:.2f}%")

    # ── Naïve and corrected drug-specific block ───────────────────────────
    print("\nStep 2/4: Drug-specific block for all four drugs ...")
    for name, p in DRUGS.items():
        print(f"  {name} ...")
        results['drug_specific'][name] = {}
        results['naive'][name]         = {}
        for f in KEY_FREQS:
            peaks_drug, _, _ = run_pulse_train(
                f, p['koff'], p['kon'], p['kon'], p['conc'], n_pulses=N_PULSES)
            b_naive     = round(naive_block(peaks_drug), 2)
            b_corrected = round(corrected_block(peaks_drug, nodrug_cache[f]), 2)
            results['naive'][name][str(f)]         = b_naive
            results['drug_specific'][name][str(f)] = b_corrected
        print(f"    naive:     {results['naive'][name]}")
        print(f"    corrected: {results['drug_specific'][name]}")

    # ── koff sweep ───────────────────────────────────────────────────────
    # Near-saturating conditions: kon=0.01 uM^-1ms^-1, conc=100 uM.
    print("\nStep 3/4: koff sweep (kon=0.01, conc=100 uM) ...")
    for koff in KOFF_SWEEP:
        row = [koff, round(1.0 / koff, 1)]
        for f in [2.0, 8.0]:
            pd, _, _ = run_pulse_train(f, koff, 0.01, 0.01, 100.0, n_pulses=25)
            pn, _, _ = run_pulse_train(f, koff, 0.01, 0.01,   0.0, n_pulses=25)
            b = round(corrected_block(pd, pn), 2)
            row.append(b)
        results['koff_sweep'].append(row)
        print(f"  koff={koff:.4f} ms-1  tau={row[1]:.0f} ms  "
              f"B@2Hz={row[2]:.1f}%  B@8Hz={row[3]:.1f}%")

    # Build selectivity-ratio subset
    for row in results['koff_sweep']:
        koff, tau, b2, b8 = row
        if koff in KOFF_RATIO_SUBSET:
            ratio = round(b8 / b2, 3) if b2 > 0 else None
            results['selectivity_ratio'].append([koff, tau, b2, b8, ratio])

    # ── kon sensitivity analysis ─────────────────────────────────────────
    print(f"\nStep 4/4: kon sensitivity at koff={KOFF_OPTIMAL} ms-1, conc=100 uM ...")
    sens = []
    for kon in KON_SENSITIVITY:
        row = {'kon': kon}
        for f, label in [(2.0, 'b2'), (8.0, 'b8')]:
            pd, _, _ = run_pulse_train(f, KOFF_OPTIMAL, kon, kon, 100.0, n_pulses=25)
            pn, _, _ = run_pulse_train(f, KOFF_OPTIMAL, kon, kon,   0.0, n_pulses=25)
            row[label] = round(corrected_block(pd, pn), 2)
        sens.append(row)
        print(f"  kon={kon:.3f}  B@2Hz={row['b2']:.1f}%  B@8Hz={row['b8']:.1f}%")
    results['sensitivity_kon'][f'koff={KOFF_OPTIMAL}'] = sens

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = 'corrected_data_REGENERATED.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")
    return results


if __name__ == '__main__':
    main()
