"""
Step 6: Full coupled model. AP waveform (step5) drives the 18-state
drug-extended Markov channel (step3) via continuous ODE integration
(not matrix-exponential, since V(t) varies continuously during the AP).
"""
import numpy as np
from scipy.integrate import solve_ivp
from step1_markov_core import rates, steady_state
from step5_ap_waveform import voltage_at, apd_for_freq

IDX = dict(O=0,P=1,Q=2,R=3,S=4,T=5,U=6,V=7,W=8)

def gating_block(V):
    r = rates(V)
    a11,a12,a13 = r['a11'], r['a12'], r['a13']
    b11,b12,b13 = r['b11'], r['b12'], r['b13']
    a2,b2,a3,b3 = r['a2'], r['b2'], r['a3'], r['b3']
    a4,b4,a5,b5 = r['a4'], r['b4'], r['a5'], r['b5']
    Ag = np.zeros((9,9))
    Ag[0,0] = -(b13+a2); Ag[0,1] = a13; Ag[0,6] = b2
    Ag[1,0] = b13; Ag[1,1] = -(b12+b3+a13); Ag[1,2] = a12; Ag[1,6] = a3
    Ag[2,1] = b12; Ag[2,2] = -(b11+b3+a12); Ag[2,3] = a11; Ag[2,5] = a3
    Ag[3,2] = b11; Ag[3,3] = -(b3+a11); Ag[3,4] = a3
    Ag[4,3] = b3; Ag[4,4] = -(a11+a3); Ag[4,5] = b11
    Ag[5,2] = b3; Ag[5,4] = a11; Ag[5,5] = -(a3+b11+a12); Ag[5,6] = b12
    Ag[6,0] = a2; Ag[6,1] = b3; Ag[6,5] = a12; Ag[6,6] = -(b12+a3+b2+a4); Ag[6,7] = b4
    Ag[7,6] = a4; Ag[7,7] = -(b4+a5); Ag[7,8] = b5
    Ag[8,7] = a5; Ag[8,8] = -b5
    return Ag

def rhs_full(t, u, freq_hz, n_pulses, koff, kon_open, kon_inact, conc, apd_override):
    V = voltage_at(t, freq_hz, n_pulses, apd=apd_override)
    Ag = gating_block(V)
    du = np.zeros(18)
    du[0:9]  = Ag @ u[0:9]
    du[9:18] = Ag @ u[9:18]

    kon_vec = np.zeros(9)
    kon_vec[IDX['O']] = kon_open
    kon_vec[IDX['U']] = kon_inact
    kon_vec[IDX['V']] = kon_inact
    kon_vec[IDX['W']] = kon_inact

    for i in range(9):
        konc = kon_vec[i]*conc
        flux = konc*u[i] - koff*u[i+9]
        du[i]   -= flux
        du[i+9] += flux
    return du

def run_pulse_train(freq_hz, koff, kon_open, kon_inact, conc, n_pulses=40,
                     apd_override=None, t_eval_per_pulse=60):
    """Simulate a pulse train and return peak open-probability per pulse.

    Peak detection uses DENSE sampling concentrated in the first ~3ms of each
    pulse (where the fast Na+ activation transient peaks), since coarse
    uniform sampling across the whole AP can miss/alias the sharp peak and
    produce spurious non-monotonic "peaks" (especially at low frequency
    where most of the pulse period is the long flat plateau/resting phase).
    """
    period = 1000.0/freq_hz
    apd = apd_override if apd_override is not None else apd_for_freq(freq_hz)

    u0 = np.zeros(18)
    u_rest, _ = steady_state(-85.0)
    u0[0:9] = u_rest

    t_span = (0, n_pulses*period)
    sol = solve_ivp(rhs_full, t_span, u0, method='LSODA',
                     args=(freq_hz, n_pulses, koff, kon_open, kon_inact, conc, apd),
                     max_step=min(1.0, apd/20), rtol=1e-6, atol=1e-9,
                     dense_output=True)

    # Dense, non-uniform sampling: heavy resolution in first 4ms of each pulse
    # (captures the activation peak), sparser resolution for the rest (for
    # plotting/visual purposes only).
    t_list = []
    for p in range(n_pulses):
        t0 = p*period
        t_list.append(np.linspace(t0, t0+4.0, 200))            # dense upstroke window
        t_list.append(np.linspace(t0+4.0, t0+period, t_eval_per_pulse))  # rest of period
    t_eval = np.unique(np.concatenate(t_list))
    t_eval = t_eval[t_eval <= n_pulses*period]
    U = sol.sol(t_eval)

    O_unbound = U[0]
    peaks = np.zeros(n_pulses)
    for p in range(n_pulses):
        t0, t1 = p*period, p*period + min(apd, period)
        mask = (t_eval >= t0) & (t_eval <= t1)
        peaks[p] = O_unbound[mask].max() if mask.any() else 0.0
    return peaks, t_eval, U

if __name__ == "__main__":
    koff = 0.01216
    kon_open = 0.0149
    kon_inact = 0.0149
    conc = 1.0  # uM

    print("=== Full AP-driven model: flecainide-like, 1Hz vs 8Hz ===\n")
    for freq in [1.0, 8.0]:
        peaks, t_eval, U = run_pulse_train(freq, koff, kon_open, kon_inact, conc, n_pulses=30)
        block_pct = (1-peaks[-1]/peaks[0])*100
        print(f"Freq={freq} Hz (APD={apd_for_freq(freq):.0f}ms): block@pulse30 = {block_pct:.2f}%  (p1={peaks[0]:.5f}, p30={peaks[-1]:.5f})")
