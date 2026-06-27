"""
Step 1: Build and validate the Clancy & Rudy (2002) 9-state Markov model
for Nav1.5, WITHOUT drug binding, to confirm the core channel kinetics
behave physiologically (proper activation/inactivation, I-V curve shape).

States: O, C1, C2, C3, IC3, IC2, IF, IM1, IM2
(using Stary & Biktashev 2015 notation: O,P,Q,R,S,T,U,V,W)

Reference: Clancy CE, Rudy Y. (2002). Na+ channel mutation that causes
both Brugada and long-QT syndrome phenotypes: a simulation study of
mechanism. Circulation 105(10):1208-1213.

Transition rates as documented in Stary & Biktashev (2015, arXiv:1411.6204),
who reproduced the original Clancy-Rudy code with author permission.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Transition rate functions (V in mV, rates in ms^-1)
# State order: [O, C1, C2, C3, IC3, IC2, IF, IM1, IM2]
#              [0,  1,  2,  3,  4,   5,   6,  7,   8]
# Using paper notation: O=O, C1=P, C2=Q, C3=R, IC3=S, IC2=T, IF=U, IM1=V, IM2=W
# ---------------------------------------------------------------

def rates(V):
    a11 = 3.802 / (0.1027*np.exp(-V/17.0) + 0.20*np.exp(-V/150.0))   # alpha_RQ = alpha_ST
    a12 = 3.802 / (0.1027*np.exp(-V/15.0) + 0.23*np.exp(-V/150.0))   # alpha_QP = alpha_TU
    a13 = 3.802 / (0.1027*np.exp(-V/12.0) + 0.25*np.exp(-V/150.0))   # alpha_PO

    b11 = 0.1917*np.exp(-V/20.3)                                     # alpha_QR = alpha_TS
    b12 = 0.20*np.exp(-(V-5.0)/20.3)                                 # alpha_PQ = alpha_UT
    b13 = 0.22*np.exp(-(V-10.0)/20.3)                                # alpha_OP

    a3 = 3.7933e-7*np.exp(-V/7.7)                                    # alpha_UP=alpha_TQ=alpha_SR
    b3 = 8.4e-3 + 2e-5*V                                             # alpha_PU=alpha_QT=alpha_RS

    a2 = 9.178*np.exp(V/29.68)                                       # alpha_OU
    b2 = (a13*a2*a3)/(b13*b3)                                        # alpha_UO

    a4 = a2/100.0                                                    # alpha_UV
    b4 = a3                                                          # alpha_VU

    a5 = a2/(9.5e4)                                                  # alpha_VW
    b5 = a3/50.0                                                     # alpha_WV

    return dict(a11=a11, a12=a12, a13=a13, b11=b11, b12=b12, b13=b13,
                a2=a2, b2=b2, a3=a3, b3=b3, a4=a4, b4=b4, a5=a5, b5=b5)

def dudt(t, u, V):
    """Reference ODE system, eq. (153)-(161) of Stary & Biktashev (2015),
    reproducing Clancy & Rudy (2002). State order: O,P,Q,R,S,T,U,V,W.

    Transition-rate dictionary mapping (Table II of the same reference):
      a_RQ=a_ST=a11   a_QP=a_TU=a12   a_PO=a13
      a_QR=a_TS=b11   a_PQ=a_UT=b12   a_OP=b13
      a_UP=a_TQ=a_SR=a3     a_PU=a_QT=a_RS=b3
      a_OU=a2   a_UO=b2   a_UV=a4   a_VU=b4   a_VW=a5   a_WV=b5
    """
    O, P, Q, R, S, T, U, Vv, W = u
    r = rates(V)
    a11,a12,a13 = r['a11'], r['a12'], r['a13']
    b11,b12,b13 = r['b11'], r['b12'], r['b13']
    a2,b2,a3,b3 = r['a2'], r['b2'], r['a3'], r['b3']
    a4,b4,a5,b5 = r['a4'], r['b4'], r['a5'], r['b5']

    dO = a13*P + b2*U - (b13 + a2)*O
    dP = a12*Q + a3*U + b13*O - (b12 + b3 + a13)*P
    dQ = a11*R + a3*T + b12*P - (b11 + b3 + a12)*Q
    dR = a3*S + b11*Q - (b3 + a11)*R
    dS = b11*T + b3*R - (a11 + a3)*S
    dT = b3*Q + a11*S + b12*U - (a3 + b11 + a12)*T
    dU = a12*T + b3*P + b4*Vv + a2*O - (b12 + a3 + b2 + a4)*U
    dV = a4*U + b5*W - (b4 + a5)*Vv
    dW = a5*Vv - b5*W

    return [dO, dP, dQ, dR, dS, dT, dU, dV, dW]

def steady_state(V):
    """Find steady-state occupancies via the null space of the generator
    matrix A(V) (exact linear algebra; robust for this stiff CTMC)."""
    r = rates(V)
    a11,a12,a13 = r['a11'], r['a12'], r['a13']
    b11,b12,b13 = r['b11'], r['b12'], r['b13']
    a2,b2,a3,b3 = r['a2'], r['b2'], r['a3'], r['b3']
    a4,b4,a5,b5 = r['a4'], r['b4'], r['a5'], r['b5']

    A = np.zeros((9,9))
    # row i = d(state_i)/dt coefficients, columns = source state
    A[0,0] = -(b13+a2); A[0,1] = a13; A[0,6] = b2                        # dO
    A[1,0] = b13; A[1,1] = -(b12+b3+a13); A[1,2] = a12; A[1,6] = a3      # dP
    A[2,1] = b12; A[2,2] = -(b11+b3+a12); A[2,3] = a11; A[2,5] = a3      # dQ
    A[3,2] = b11; A[3,3] = -(b3+a11); A[3,4] = a3                        # dR
    A[4,3] = b3; A[4,4] = -(a11+a3); A[4,5] = b11                        # dS  (FIXED)
    A[5,2] = b3; A[5,4] = a11; A[5,5] = -(a3+b11+a12); A[5,6] = b12      # dT
    A[6,0] = a2; A[6,1] = b3; A[6,5] = a12; A[6,6] = -(b12+a3+b2+a4); A[6,7] = b4  # dU
    A[7,6] = a4; A[7,7] = -(b4+a5); A[7,8] = b5                          # dV
    A[8,7] = a5; A[8,8] = -b5                                            # dW

    M = A.copy()
    M[-1,:] = 1.0
    rhs = np.zeros(9); rhs[-1] = 1.0
    u = np.linalg.solve(M, rhs)
    return u, A

if __name__ == "__main__":
    # Test 1: at rest (V=-95mV), most channels should be in deep closed states (R,S)
    u_rest, A_rest = steady_state(-95.0)
    labels = ['O','C1','C2','C3','IC3','IC2','IF','IM1','IM2']
    print("Steady state at V=-95 mV (resting):")
    for l, v in zip(labels, u_rest):
        print(f"  {l:5s}: {v:.6f}")
    print(f"  Sum: {u_rest.sum():.6f}")
    print(f"  max|A u| (should be ~0): {np.max(np.abs(A_rest @ u_rest)):.3e}")

    # Test 2: at depolarized V=-20mV -> channels should inactivate (IF, IM1 high)
    u_depol, A_depol = steady_state(-20.0)
    print("\nSteady state at V=-20 mV (depolarized, inactivated):")
    for l, v in zip(labels, u_depol):
        print(f"  {l:5s}: {v:.6f}")
    print(f"  Sum: {u_depol.sum():.6f}")
    print(f"  max|A u| (should be ~0): {np.max(np.abs(A_depol @ u_depol)):.3e}")
