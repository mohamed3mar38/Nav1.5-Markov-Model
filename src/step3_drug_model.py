"""
Step 3: Add drug binding to the validated 9-state Clancy-Rudy core.

We extend the model following the Modulated Receptor framework (Hille, 1977),
now implemented exactly (not approximated) on top of the full Markov gating
scheme rather than the oversimplified HH dDB/dt approximation used in the
original (buggy) script.

Drug-bound states mirror the 9 unbound states: each channel state X has a
drug-bound counterpart X_D. Binding occurs predominantly from OPEN and
INACTIVATED states (consistent with the hydrophilic-pathway access
requirement of Hille 2001, Ch. 16), with separate kon for open-state and
inactivated-state access, as is standard in the literature (e.g. Moreno
et al. 2011/2013 for ranolazine; Ramos & O'Leary 2004 for flecainide).

This gives an 18-state Markov chain:
  Unbound: O, P, Q, R, S, T, U, V, W   (indices 0-8)
  Bound:   Od,Pd,Qd,Rd,Sd,Td,Ud,Vd,Wd  (indices 9-17)

Drug-channel transitions (state X <-> X_D):
  d(X_D)/dt += kon_X * [D] * X - koff * X_D
  dX/dt     -= kon_X * [D] * X - koff * X_D

where kon_X = kon_open for X in {O}, kon_inact for X in {U,V,W} (IF,IM1,IM2),
and kon_X = 0 for closed states C1,C2,C3,IC2,IC3 (drug cannot access pore
through hydrophilic pathway when activation gate is shut), consistent with
the "guarded receptor" aspect of the Modulated Receptor Model.

All gating transitions among the 9 unbound states are mirrored exactly among
the 9 drug-bound states (the channel still gates while drug-bound, it just
cannot conduct current -- this is the standard Modulated Receptor assumption
that drug binding does not alter voltage-sensor movement, only pore
conductance).
"""
import numpy as np
from scipy.integrate import solve_ivp
from step1_markov_core import rates

N_UNBOUND = 9
IDX = dict(O=0, P=1, Q=2, R=3, S=4, T=5, U=6, V=7, W=8)

def build_generator(V, koff, kon_open, kon_inact, conc):
    """Build the 18x18 generator matrix for the drug-extended Markov model.
    State vector order: [O,P,Q,R,S,T,U,V,W, Od,Pd,Qd,Rd,Sd,Td,Ud,Vd,Wd]
    """
    r = rates(V)
    a11,a12,a13 = r['a11'], r['a12'], r['a13']
    b11,b12,b13 = r['b11'], r['b12'], r['b13']
    a2,b2,a3,b3 = r['a2'], r['b2'], r['a3'], r['b3']
    a4,b4,a5,b5 = r['a4'], r['b4'], r['a5'], r['b5']

    # Build the 9x9 gating block (same structure for bound and unbound)
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

    A = np.zeros((18,18))
    A[0:9, 0:9]   = Ag   # unbound gating
    A[9:18, 9:18] = Ag   # bound gating (same kinetics while drug-bound)

    # Drug binding/unbinding kon depends on the state (open=O index0, inactivated=U,V,W idx 6,7,8)
    kon_vec = np.zeros(9)
    kon_vec[IDX['O']] = kon_open
    kon_vec[IDX['U']] = kon_inact
    kon_vec[IDX['V']] = kon_inact
    kon_vec[IDX['W']] = kon_inact
    # closed states (P,Q,R,S,T) = 0 (guarded receptor: no access when gate shut)

    for i in range(9):
        kon_i = kon_vec[i] * conc
        if kon_i > 0:
            A[i+9, i]   += kon_i     # unbound -> bound
            A[i, i]     -= kon_i
        A[i, i+9]   += koff       # bound -> unbound
        A[i+9, i+9] -= koff

    return A

def simulate_protocol(times_V, koff, kon_open, kon_inact, conc, u0=None, dt_max=0.01):
    """times_V: list of (t_start, t_end, V) segments (square pulses).
    Returns (t_eval, U) trajectory across the whole protocol via exact
    matrix-exponential stepping per segment (since A is piecewise-constant)."""
    from scipy.linalg import expm

    if u0 is None:
        u0 = np.zeros(18)
        u0[3] = 1.0  # start all channels in C3 (resting, unbound)

    t_all = []
    U_all = []
    u = u0.copy()
    for (t0, t1, V) in times_V:
        A = build_generator(V, koff, kon_open, kon_inact, conc)
        n_steps = max(1, int(np.ceil((t1-t0)/dt_max)))
        ts = np.linspace(t0, t1, n_steps+1)
        # Precompute matrix exponential for this segment's dt
        dt = ts[1]-ts[0] if len(ts) > 1 else (t1-t0)
        if dt <= 0:
            continue
        Texp = expm(A*dt)
        seg_U = np.zeros((18, len(ts)))
        seg_U[:,0] = u
        for k in range(1, len(ts)):
            u = Texp @ u
            seg_U[:,k] = u
        t_all.append(ts)
        U_all.append(seg_U)
        u = seg_U[:,-1]

    t_full = np.concatenate(t_all)
    U_full = np.concatenate(U_all, axis=1)
    return t_full, U_full

if __name__ == "__main__":
    # Quick sanity check: zero drug concentration should reduce to pure
    # 9-state gating identical to step1/step2.
    proto = [(0,2,-95.0), (2,7,-20.0), (7,20,-95.0)]
    t, U = simulate_protocol(proto, koff=0.012, kon_open=0.15, kon_inact=0.15, conc=0.0)
    O_t = U[0]
    print("conc=0 sanity check (should match step2 transient):")
    print(f"  Peak O = {O_t.max():.5f} at t={t[np.argmax(O_t)]:.3f} ms")
    print(f"  Sum of all 18 states at t=10: {U[:, np.argmin(np.abs(t-10))].sum():.6f}")
