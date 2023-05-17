# Aspiration Rescaling (AR) algorithm for discounted MDPs

import time as _time
import numpy as np
from scipy.optimize import linprog

from .mdp import MDP
from . import util as _util


_MSG_STOP_MAX_ITER = "Iterating stopped due to maximum number of iterations " \
    "condition."


def _printVerbosity(iteration, variation):
    if isinstance(variation, float):
        print("{:>10}{:>12f}".format(iteration, variation))
    elif isinstance(variation, int):
        print("{:>10}{:>12d}".format(iteration, variation))
    else:
        print("{:>10}{:>12}".format(iteration, variation))

np.set_printoptions(suppress=True, formatter={'all': lambda x: f"{x:5.2f}"})

class ARTree(MDP):
    """Aspiration Rescaling (AR) algorithm for discounted tree-shaped MDPs"""

    isTerminal = None
    """(1d array of bools) Whether each state is terminal"""
    ERsquared = None
    """(1d array) Expected squared reward for each state"""
    s0 = None
    """Initial state"""
    aleph0 = None
    """Initial aspiration level"""

    stochasticPolicy = None
    """(2d array) Computed satisficing policy (*not* the 'optimal' policy!). 
    Index (s,a) is the probability of taking action a in state s."""
    V = None
    """(1d array) Value function for computed satisficing policy"""
    Q = None
    """(2d array) Q table for computed satisficing policy"""
    aleph = None
    """(1d array) Aspiration levels for computed satisficing policy, by state"""


    def __init__(self, 
                 isTerminal=None, # whether each state is terminal
                 reward=None, # expected reward for each state
                 ERsquared=None, # expected squared reward for each state. If None, use R**2
                 s0=None, aleph0=None, mode="minW", 
                 **kwargs):
        """Initialize an AR MDP for a specific initial state and aspiration level."""

        assert s0 is not None, "Initial state s0 must be specified"
        assert aleph0 is not None, "Initial aspiration level aleph0 must be specified"
        assert isTerminal is not None, "isTerminal (1d array of bools) must be specified"

        MDP.__init__(self, reward=reward, **kwargs)

        if ERsquared is None:
            ERsquared = tuple(Ra**2 for Ra in reward)

        # TODO: verify that the MDP is a tree?

        self.s0 = s0
        self.isTerminal = isTerminal
        self.aleph0 = aleph0
        self.mode = mode
        self.ERsquared = ERsquared

        if self.discount < 1:
            # compute a bound for the number of iterations and update the
            # stored value of self.max_iter
#            self._boundIter(self.epsilon)
            self.max_iter = 1000
            # computation of threshold of variation for V for an epsilon-
            # optimal policy
            self.thresh = self.epsilon * (1 - self.discount) / self.discount
        else:  # discount == 1
            # threshold of variation for V for an epsilon-optimal policy
            self.thresh = self.epsilon


    def run(self):
        self.setVerbose()
        self._startRun()

        # shortcuts:
        P = self.P
        ER = self.R  # expected reward conditional on state, action, next state
        ERsquared = self.ERsquared  # expected squared reward conditional on state, action, next state
        gamma = self.discount
        s0 = self.s0
        ones = np.ones(self.A)

        # initialize:
        V = np.zeros(self.S)
        W = np.zeros(self.S)
        Q = np.zeros((self.S, self.A))
        G = np.zeros((self.S, self.A))
        aleph = np.zeros(self.S)
        aleph[s0] = self.aleph0
        stochasticPolicy = np.zeros((self.S, self.A))

        # main loop over versions of V, W, Q, G, and aleph:
        converged = False
        while not converged:
            if self.verbose: print(f"new iteration")
            # compute quantities derived from current Q:
            Vbottom = Q.min(axis=1)
            Vtop = Q.max(axis=1)
            Vspan = Vtop - Vbottom
            BIG = 2 * Vspan.max()
            # one forward pass through the tree:
            nStatesVisited = 0
            stack = {s0}
            Vnext = V.copy()
            Wnext = W.copy()
            Qnext = Q.copy()
            Gnext = G.copy()
            # inner loop over states, in order of occurrence in the tree:
            while len(stack) > 0 and nStatesVisited < self.S:
                s = stack.pop()
                nStatesVisited += 1
                if not self.isTerminal[s]:
                    # compute local policy based on current Q and aleph:
                    Q_s = Q[s,:]
                    G_s = G[s,:]
                    aleph_s = aleph[s]
                    # set target V based on aleph and bounds:
                    Vtarget_s = max(min(aleph_s, Vtop[s]), Vbottom[s])
                    sP_s = np.zeros(self.A)
                    fallback = False
                    # compute stochastic policy depending on mode:
                    if self.mode == "minW":
                        # find a minimum-expected-G mix of actions with correct expected Q:
                        res = linprog(G_s, A_eq=[Q_s,ones], b_eq=[Vtarget_s,1], bounds=(0,1), 
                                      options={'cholesky':False, 'sym_pos':False, 'lstsq':True, 'presolve':True})
                        assert res.success, f"linprog failed: {res.message}"
                        if res.success:
                            sP_s = res.x
                        else:
                            fallback = True
                    if self.mode == "closest" or fallback:
                        # find a suitable mix of two actions just below and above the effective aleph level:
                        aLow = (Q_s - BIG*(Q_s > Vtarget_s)).argmax()
                        aHigh = (Q_s + BIG*(Q_s < Vtarget_s)).argmin()
                        if aLow == aHigh:
                            sP_s[aLow] = 1
                        else:
                            pHigh = (Vtarget_s - Q_s[aLow]) / (Q_s[aHigh] - Q_s[aLow])
                            pLow = 1 - pHigh
                            sP_s[aLow] = pLow
                            sP_s[aHigh] = pHigh
                    stochasticPolicy[s,:] = sP_s
                    if self.verbose: 
                        print(f"  s {s:2d} aleph {aleph_s:5.2f} Q",Q_s,f"Vbot {Vbottom[s]:5.2f} Vtop {Vtop[s]:5.2f} Vtarget {Vtarget_s:5.2f} sP",sP_s)
                    for a in range(self.A):
                        P_sa = P[a][s]
                        successors = np.where(P_sa > 0)[0]  # list of possible successor states
                        stack.update(successors)
                        ER_sa = ER[a][s]
                        ERsquared_sa = ERsquared[a][s]
                        # compute expectation of reward plus gamma times Vbottom or Vtop over all possible successor states:
                        Qbottom_sa = P_sa.dot(ER_sa + gamma*Vbottom)
                        Qtop_sa = P_sa.dot(ER_sa + gamma*Vtop)
                        # update Q[s,a] and G[s,a]:
                        Qnext[s,a] = P_sa.dot(ER_sa + gamma*V)
                        Gnext[s,a] = P_sa.dot(ERsquared_sa + 2*gamma*ER_sa*V + gamma**2*W)
                        # set alephs of all successor states,
                        # using aspiration rescaling if the current aspiration currently appears feasible,
                        # and retaining the current aleph otherwise:
                        Q_sa_or_aleph_s = Q_s[a] if Vtarget_s == aleph_s and sP_s[a] > 0 else aleph_s
                        # compute relative aspiration level and use it to set rescaled absolute aspiration levels for all successor states:
                        l = 0.5 if Qtop_sa == Qbottom_sa else min(max(0,(Q_sa_or_aleph_s - Qbottom_sa) / (Qtop_sa - Qbottom_sa)),1)
                        aleph[successors] = (1-l) * Vbottom[successors] + l * Vtop[successors]
                        if self.verbose:
                            print(f"    a {a:2d} R {ER_sa:5.2f} Qbot {Qbottom_sa:5.2f} Qtop {Qtop_sa:5.2f} q {Q_sa_or_aleph_s:5.2f} l {l:5.2f} al",aleph[successors])
                    # recalculate V and W for this state based on the updated Q and G:
                    Vnext[s] = (V[s] + sP_s.dot(Qnext[s,:])) / 2
                    Wnext[s] = (W[s] + sP_s.dot(Gnext[s,:])) / 2
                    if self.verbose: 
                        print(f"               new Q",Qnext[s,:],f"V {Vnext[s]:5.2f}")

            print("aleph", aleph)
            print("Vnext", Vnext)

            assert len(stack) == 0, "MDP appears not to be a tree"

            variation = _util.getSpan(Qnext - Q) + _util.getSpan(Vnext - V) + _util.getSpan(Gnext - G) +  _util.getSpan(Wnext - W)

            if self.verbose:
                _printVerbosity(self.iter, variation)

            if variation < self.epsilon:
                if self.verbose:
                    print("Iterating stopped, Q table changes less than epsilon.")
                converged = True
            elif self.iter == self.max_iter:
                if self.verbose:
                    print(_MSG_STOP_MAX_ITER)
                converged = True

            V = Vnext
            W = Wnext
            Q = Qnext
            G = Gnext

        self.V = V
        self.Q = Q
        self.aleph = aleph
        self.stochasticPolicy = stochasticPolicy

        self.policy = np.zeros(self.S)
        self._endRun()
        self.policy = None  # policy is stochastic, so we don't store a deterministic policy!

        assert abs(V[0] - self.aleph0) < self.epsilon, "V[0] != aleph0"

"""
POSSIBLE CYCLIC BEHAVIOUR:

...

aleph [ 0.00  0.76 -0.59 -0.39  0.84  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]
Vnext [-0.08  0.76 -0.59 -0.39  0.84  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]
         0    3.109636
new iteration
  s  0 aleph  0.00 Q [-0.08  0.31] Vbot -0.08 Vtop  0.31 Vtarget  0.00 sP [ 0.79  0.21]
    a  0 R -0.17 Qbot -0.43 Qtop  0.26 q -0.08 l  0.50 al [ 0.66 -0.65]
    a  1 R -0.69 Qbot -0.16 Qtop  0.78 q  0.31 l  0.50 al [ 0.19  1.11]
               new Q [ 0.00  0.00] V  0.00
  s  1 aleph  0.66 Q [ 1.07  0.25] Vbot  0.25 Vtop  1.07 Vtarget  0.66 sP [ 0.50  0.50]
    a  0 R  1.07 Qbot  1.07 Qtop  1.07 q  1.07 l  0.50 al [ 0.00  0.00]
    a  1 R  0.25 Qbot  0.25 Qtop  0.25 q  0.25 l  0.50 al [ 0.00  0.00]
               new Q [ 1.07  0.25] V  0.66
  s  2 aleph -0.65 Q [-0.91 -0.39] Vbot -0.91 Vtop -0.39 Vtarget -0.65 sP [ 0.50  0.50]
    a  0 R -0.91 Qbot -0.91 Qtop -0.91 q -0.91 l  0.50 al [ 0.00  0.00]
    a  1 R -0.39 Qbot -0.39 Qtop -0.39 q -0.39 l  0.50 al [ 0.00  0.00]
               new Q [-0.91 -0.39] V -0.65
  s  3 aleph  0.19 Q [ 1.08 -0.70] Vbot -0.70 Vtop  1.08 Vtarget  0.19 sP [ 0.50  0.50]
    a  0 R  1.08 Qbot  1.08 Qtop  1.08 q  1.08 l  0.50 al [ 0.00  0.00]
    a  1 R -0.70 Qbot -0.70 Qtop -0.70 q -0.70 l  0.50 al [ 0.00  0.00]
               new Q [ 1.08 -0.70] V  0.19
  s  4 aleph  1.11 Q [ 0.70  1.53] Vbot  0.70 Vtop  1.53 Vtarget  1.11 sP [ 0.50  0.50]
    a  0 R  0.70 Qbot  0.70 Qtop  0.70 q  0.70 l  0.50 al [ 0.00  0.00]
    a  1 R  1.53 Qbot  1.53 Qtop  1.53 q  1.53 l  0.50 al [ 0.00  0.00]
               new Q [ 0.70  1.53] V  1.11
aleph [ 0.00  0.66 -0.65  0.19  1.11  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]
Vnext [ 0.00  0.66 -0.65  0.19  1.11  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]
         0    3.109636
new iteration
  s  0 aleph  0.00 Q [ 0.00  0.00] Vbot  0.00 Vtop  0.00 Vtarget  0.00 sP [ 1.00  0.00]
    a  0 R -0.17 Qbot -0.43 Qtop  0.26 q  0.00 l  0.62 al [ 0.76 -0.59]
    a  1 R -0.69 Qbot -0.16 Qtop  0.78 q  0.00 l  0.17 al [-0.39  0.84]
               new Q [-0.08  0.31] V -0.08
  s  1 aleph  0.76 Q [ 1.07  0.25] Vbot  0.25 Vtop  1.07 Vtarget  0.76 sP [ 0.62  0.38]
    a  0 R  1.07 Qbot  1.07 Qtop  1.07 q  1.07 l  0.50 al [ 0.00  0.00]
    a  1 R  0.25 Qbot  0.25 Qtop  0.25 q  0.25 l  0.50 al [ 0.00  0.00]
               new Q [ 1.07  0.25] V  0.76
  s  2 aleph -0.59 Q [-0.91 -0.39] Vbot -0.91 Vtop -0.39 Vtarget -0.59 sP [ 0.38  0.62]
    a  0 R -0.91 Qbot -0.91 Qtop -0.91 q -0.91 l  0.50 al [ 0.00  0.00]
    a  1 R -0.39 Qbot -0.39 Qtop -0.39 q -0.39 l  0.50 al [ 0.00  0.00]
               new Q [-0.91 -0.39] V -0.59
  s  3 aleph -0.39 Q [ 1.08 -0.70] Vbot -0.70 Vtop  1.08 Vtarget -0.39 sP [ 0.17  0.83]
    a  0 R  1.08 Qbot  1.08 Qtop  1.08 q  1.08 l  0.50 al [ 0.00  0.00]
    a  1 R -0.70 Qbot -0.70 Qtop -0.70 q -0.70 l  0.50 al [ 0.00  0.00]
               new Q [ 1.08 -0.70] V -0.39
  s  4 aleph  0.84 Q [ 0.70  1.53] Vbot  0.70 Vtop  1.53 Vtarget  0.84 sP [ 0.83  0.17]
    a  0 R  0.70 Qbot  0.70 Qtop  0.70 q  0.70 l  0.50 al [ 0.00  0.00]
    a  1 R  1.53 Qbot  1.53 Qtop  1.53 q  1.53 l  0.50 al [ 0.00  0.00]
               new Q [ 0.70  1.53] V  0.84
aleph [ 0.00  0.76 -0.59 -0.39  0.84  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]
Vnext [-0.08  0.76 -0.59 -0.39  0.84  0.00  0.00  0.00  0.00  0.00  0.00  0.00
  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00]

"""


"""
POSSIBLE CONVERGENT BUT INCONSISTENT BEHAVIOUR DUE TO TOO LOW ALEPH0:

...


Vnext [ 0.41  0.81 -0.63  0.00  0.00  0.00  0.00]
         0    2.092540
new iteration
  s  0 aleph  0.10 Q [ 2.77  0.41] Vbot  0.41 Vtop  2.77 Vtarget  0.41 sP [ 0.00  1.00]
    a  0 R  1.96 Qbot  2.77 Qtop  2.81 q  0.10 l -65.04 al [-1.86]
    a  1 R  1.03 Qbot  0.41 Qtop  1.03 q  0.10 l -0.48 al [-0.93]
               new Q [ 2.77  0.41] V  0.41
  s  1 aleph -1.86 Q [ 0.81  0.85] Vbot  0.81 Vtop  0.85 Vtarget  0.81 sP [ 1.00  0.00]
    a  0 R  0.81 Qbot  0.81 Qtop  0.81 q -1.86 l  0.50 al [ 0.00]
    a  1 R  0.85 Qbot  0.85 Qtop  0.85 q -1.86 l  0.50 al [ 0.00]
               new Q [ 0.81  0.85] V  0.81
  s  2 aleph -0.93 Q [-0.63 -0.00] Vbot -0.63 Vtop -0.00 Vtarget -0.63 sP [ 1.00  0.00]
    a  0 R -0.63 Qbot -0.63 Qtop -0.63 q -0.93 l  0.50 al [ 0.00]
    a  1 R -0.00 Qbot -0.00 Qtop -0.00 q -0.93 l  0.50 al [ 0.00]
               new Q [-0.63 -0.00] V -0.63
V     [ 0.41  0.81 -0.63  0.00  0.00  0.00  0.00]
aleph [ 0.10 -1.86 -0.93  0.00  0.00  0.00  0.00]
Vnext [ 0.41  0.81 -0.63  0.00  0.00  0.00  0.00]
"""