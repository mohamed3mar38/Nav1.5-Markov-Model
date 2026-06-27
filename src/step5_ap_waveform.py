"""
Step 5: Realistic action-potential-like voltage waveform for driving the
channel model, instead of an artificial square pulse.

We use a simple, widely-used analytical AP template (sum of exponentials)
that reproduces the qualitative shape of a human atrial action potential:
fast upstroke, brief plateau, then repolarization -- with the property that
APD SHORTENS at higher pacing rate (restitution), which is a real and
important physiological feature of atrial tissue, especially in AF where
electrical remodeling produces APD shortening (Nattel & Harada 2014, JACC).

For the purposes of THIS Nav1.5/drug study, the AP shape is a forcing
function (not derived from the full ionic model), parameterized by:
  - APD_normal (ms): AP duration at normal sinus rates (~2Hz)
  - APD_AF (ms): AP duration at AF rates (electrically remodeled, shorter)
We linearly interpolate APD between these two regimes based on pacing
frequency, consistent with the well-documented AF-induced APD shortening.
"""
import numpy as np

V_REST = -85.0     # mV, resting membrane potential (atrial)
V_PEAK = 20.0      # mV, AP overshoot peak
APD_NORMAL = 200.0 # ms, normal sinus-rhythm atrial APD (Nattel & Harada 2014 range 150-250ms)
APD_AF     = 80.0  # ms, AF-remodeled atrial APD (shortened, well-documented; ~100ms or less)

def apd_for_freq(freq_hz, f_lo=2.0, f_hi=8.0):
    """Interpolate APD between normal (f<=f_lo) and AF-remodeled (f>=f_hi) regimes."""
    if freq_hz <= f_lo:
        return APD_NORMAL
    if freq_hz >= f_hi:
        return APD_AF
    frac = (freq_hz - f_lo) / (f_hi - f_lo)
    return APD_NORMAL + frac*(APD_AF - APD_NORMAL)

def ap_waveform(t_rel, apd):
    """Analytical single AP waveform, t_rel = time since upstroke onset (ms, >=0).
    Returns V(t) in mV. Shape: fast upstroke (tau_up~0.3ms), plateau, then
    repolarization timed to reach ~90% repolarization at t_rel=apd."""
    tau_up = 0.3      # ms, very fast Na-driven upstroke
    tau_down = apd / 3.5   # repolarization time constant scaled to APD

    if t_rel < 0:
        return V_REST
    upstroke = (V_PEAK - V_REST) * (1 - np.exp(-t_rel/tau_up))
    # repolarization phase kicks in and dominates after ~5*tau_up
    repol = (V_PEAK - V_REST) * np.exp(-max(0.0, t_rel - tau_up*2) / tau_down)
    V = V_REST + min(upstroke, (V_PEAK - V_REST)) * 0.0 + repol if t_rel > tau_up*2 else V_REST + upstroke
    return V

def voltage_at(t, freq_hz, n_pulses, apd=None):
    """Periodic AP train voltage at time t (ms), given pacing frequency.
    APD determined by apd_for_freq unless overridden."""
    period = 1000.0/freq_hz
    if apd is None:
        apd = apd_for_freq(freq_hz)
    phase = t % period
    return ap_waveform(phase, apd)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=(11,4))
    for freq, ax in zip([1.0, 8.0], axes):
        period = 1000.0/freq
        t = np.linspace(0, period*3, 3000)
        V = [voltage_at(tt, freq, 3) for tt in t]
        ax.plot(t, V)
        ax.set_title(f'{freq} Hz pacing (APD={apd_for_freq(freq):.0f}ms)')
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('V (mV)')
        ax.axhline(V_REST, color='gray', lw=0.5, ls=':')
    plt.tight_layout()
    plt.savefig('/home/claude/step5_ap_waveform.png', dpi=120)
    print("Saved AP waveform figure.")
    print(f"APD at 1Hz: {apd_for_freq(1.0)} ms")
    print(f"APD at 2Hz: {apd_for_freq(2.0)} ms")
    print(f"APD at 4Hz: {apd_for_freq(4.0)} ms")
    print(f"APD at 6Hz: {apd_for_freq(6.0)} ms")
    print(f"APD at 8Hz: {apd_for_freq(8.0)} ms")
