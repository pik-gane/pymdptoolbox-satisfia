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


class ARTree(MDP):
    """Aspiration Rescaling (AR) algorithm for discounted tree-shaped MDPs"""

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
                 ERsquared=None, # expected squared reward for each state. If None, use R**2
                 s0=None, aleph0=None, mode="minW", 
                 **kwargs):
        """Initialize an AR MDP for a specific initial state and aspiration level."""

        assert s0 is not None, "Initial state s0 must be specified"
        assert aleph0 is not None, "Initial aspiration level aleph0 must be specified"
        assert isTerminal is not None, "isTerminal (1d array of bools) must be specified"

        MDP.__init__(self, **kwargs)

        if ERsquared is None:
            ERsquared = self.R**2

        # TODO: verify that the MDP is a tree?

        self.s0 = s0
        self.isTerminal = isTerminal
        self.aleph0 = aleph0
        self.mode = mode

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
        self._startRun()

        # initialize Q somehow
        # repeatedly pass forward from s0, updating Q and aleph
        # in each state, compute the stochastic policy, 
        # update the aleph of all possible successor states
        # and update the Q values of all possible predecessor states
        # finally compute and store the stochastic policy

        P = self.P
        R = self.R
        ERsquared_sa = self.ERsquared
        gamma = self.discount
        s0 = self.s0
        ones = np.ones(self.A)

        V = np.zeros(self.S)
        W = np.zeros(self.S)
        Q = np.zeros((self.S, self.A))
        G = np.zeros((self.S, self.A))
        aleph = np.zeros(self.S)
        aleph[s0] = self.aleph0
        stochasticPolicy = np.zeros((self.S, self.A))

        converged = False
        while not converged:
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
                    sP = np.zeros(self.A)
                    fallback = False
                    if self.mode == "minW":
                        # find a minimum-expected-G mix of actions with correct expected Q:
                        res = linprog(G_s, A_eq=[Q_s,ones], b_eq=[Vtarget_s,1], bounds=(0,1), 
                                      options={'cholesky':False, 'sym_pos':False})
                        if res.success:
                            sP = res.x
                        else:
                            fallback = True
                    if self.mode == "closest" or fallback:
                        # find a suitable mix of two actions just below and above the effective aleph level:
                        aLow = (Q_s - BIG*(Q_s > Vtarget_s)).argmax()
                        aHigh = (Q_s + BIG*(Q_s < Vtarget_s)).argmin()
                        if aLow == aHigh:
                            sP[aLow] = 1
                        else:
                            pLow = (Vtarget_s - Q_s[aLow]) / (Q_s[aHigh] - Q_s[aLow])
                            pHigh = 1 - pLow
                            sP[aLow] = pLow
                            sP[aHigh] = pHigh
                    print("s",s,"aleph",aleph_s,"Vtarget",Vtarget_s,"Q",Q_s,"sP",sP)
                    for a in range(self.A):
                        successorProbabilities = P[a][s]
                        successors = np.where(successorProbabilities > 0)[0]
                        stack.update(successors)
                        R_sa = R[a][s]
                        ERsquared_sa = ERsquared[a][s]
                        # update Q[s,a] and G[s,a]:
                        EV = successorProbabilities.dot(V)
                        Qnext[s,a] = R_sa + gamma*EV
                        Gnext[s,a] = ERsquared_sa + 2*gamma*R_sa*EV + gamma**2*successorProbabilities.dot(W)
                        # compute alephs of all successor states:
                        Q_sa_or_aleph_s = Q_s[a] if Vtarget_s == aleph_s and sP[a] > 0 else aleph_s
                        # compute expectation of reward plus gamma times Vbottom or Vtop over all possible successor states:
                        Qbottom_sa = R_sa + gamma*successorProbabilities.dot(Vbottom)
                        Qtop_sa = R_sa + gamma*successorProbabilities.dot(Vtop)
                        # compute relative aspiration level and use it to set rescaled absolute aspiration levels for all successor states:
                        l = 0.5 if Qtop_sa == Qbottom_sa else (Q_sa_or_aleph_s - Qbottom_sa) / (Qtop_sa - Qbottom_sa)
                        aleph[successors] = (1-l) * Vbottom[successors] + l * Vtop[successors]
                    # update V[s]:
                    Vnext[s] = Vtarget_s
                    Wnext[s] = sP.dot(Gnext[s,:])
                    stochasticPolicy[s,:] = sP

            print(Wnext)

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

