#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:24:21 2022
"""
import numpy as np
from collections import deque
from scipy.linalg import blas

try:
    # Relative import (when used a package module)
    from .core import IMEXSDCCore
except ImportError:
    from core import IMEXSDCCore


class IMEXSDC(IMEXSDCCore):

    # -------------------------------------------------------------------------
    # Base constructor and specific methods
    # -------------------------------------------------------------------------
    def __init__(self, u0, lambdaI, lambdaE):

        self.M = self.getM()

        model = np.asarray(np.asarray(lambdaI) + np.asarray(lambdaE) + 0.)
        c = lambda : np.zeros_like(model)

        self.lambdaI, self.lambdaE = c(), c()
        np.copyto(self.lambdaI, lambdaI)
        np.copyto(self.lambdaE, lambdaE)

        self.u = c()  # Equivalent to solver state
        np.copyto(self.u, u0)  # Initialize state

        self.rhs, self.u0 = c(), c()
        self.lamIU = deque([[c() for _ in range(self.M)] for _ in range(2)])
        self.lamEU = deque([[c() for _ in range(self.M)] for _ in range(2)])
        if not self.leftIsNode():
            self.lamEU0, self.lamIU0 = c(), c()

        # Instanciate list of solver (ony first time step)
        self.lhs = [None] * self.M

        self.dt = None
        self.axpy = blas.get_blas_funcs('axpy', dtype=self.u.dtype)

    def _evalImplicit(self, lamIU):
        np.copyto(lamIU, self.u)
        lamIU *= self.lambdaI

    def _evalExplicit(self, lamEU):
        np.copyto(lamEU, self.u)
        lamEU *= self.lambdaE

    def _updateLHS(self, dt):
        """Update LHS coefficients

        Parameters
        ----------
        dt : float
            New time-step for the updated LHS.
        """
        # Attribute references
        qI = self.QDeltaI
        for i in range(self.M):
            self.lhs[i] = 1 - dt*qI[i, i]*self.lambdaI

    def _storeU0(self):
        np.copyto(self.u0, self.u)

    def _solveAndStoreState(self, iNode):
        np.copyto(self.u, self.rhs)
        self.u /= self.lhs[iNode]

    def _initSweep(self, iType='QDELTA'):
        """
        Initialize node terms for one given time-step

        Parameters
        ----------
        iType : str, optional
            Type of initialization, can be :
            - iType="QDELTA" : use QDelta[I,E] for coarse time integration.
            - iType="COPY" : just copy the values from the initial solution.
        """
        # Attribute references
        qI, qE = self.QDeltaI, self.QDeltaE
        dt = self.dt
        rhs, u0 = self.rhs, self.u0
        lamIUk, lamEUk = self.lamIU[0], self.lamEU[0]
        if not self.leftIsNode():
            lamEU0, lamIU0 = self.lamEU0, self.lamIU0
        axpy = self.axpy

        if iType == 'QDELTA':

            # Prepare initial field evaluation
            if not self.leftIsNode():
                self._evalExplicit(lamEU0)
                lamEU0 *= dt*self.dtauE
                if self.dtauI != 0.0:
                    self._evalImplicit(lamIU0)
                    axpy(a=dt*self.dtauI, x=lamIU0, y=lamEU0)

            # Loop on all quadrature nodes
            for i in range(self.M):
                # Build RHS
                # -- initialize with U0 term
                np.copyto(rhs, u0)
                # -- add initial field evaluation
                if not self.leftIsNode():
                    rhs += lamEU0
                # -- add explicit and implicit terms (already computed)
                for j in range(i):
                    axpy(a=dt*qE[i, j], x=lamEUk[j], y=rhs)
                    axpy(a=dt*qI[i, j], x=lamIUk[j], y=rhs)
                # Solve system and store node solution in solver state
                self._solveAndStoreState(i)
                # Evaluate implicit and implicit terms with current state
                self._evalImplicit(lamIUk[i])
                self._evalExplicit(lamEUk[i])

        elif iType == 'COPY':  # also called "spread" in pySDC
            self._evalImplicit(lamIUk[0])
            self._evalExplicit(lamEUk[0])
            for i in range(1, self.M):
                np.copyto(lamIUk[i], lamIUk[0])
                np.copyto(lamEUk[i], lamEUk[0])

        else:
            raise NotImplementedError(f'iType={iType}')

    def _sweep(self):
        """Perform a sweep for the current time-step"""
        # Attribute references
        qI, qE, q = self.QDeltaI, self.QDeltaE, self.Q
        dt = self.dt
        rhs, u0 = self.rhs, self.u0
        lamIUk, lamEUk = self.lamIU[0], self.lamEU[0]
        lamIUk1, lamEUk1 = self.lamIU[1], self.lamEU[1]
        axpy = self.axpy

        # Loop on all quadrature nodes
        for i in range(self.M):
            # Build RHS
            # -- initialize with U0 term
            np.copyto(rhs, u0)
            # -- add quadrature terms
            for j in range(self.M):
                axpy(a=dt*q[i, j], x=lamEUk[j], y=rhs)
                axpy(a=dt*q[i, j], x=lamIUk[j], y=rhs)
            # -- add explicit and implicit terms from iteration k+1
            for j in range(i):
                axpy(a=dt*qE[i, j], x=lamEUk1[j], y=rhs)
                axpy(a=dt*qI[i, j], x=lamIUk1[j], y=rhs)
            # -- add explicit and implicit terms from iteration k
            for j in range(i):
                axpy(a=-dt*qE[i, j], x=lamEUk[j], y=rhs)
                axpy(a=-dt*qI[i, j], x=lamIUk[j], y=rhs)
            axpy(a=-dt*qI[i, i], x=lamIUk[i], y=rhs)
            # Solve system and store node solution in solver state
            self._solveAndStoreState(i)
            # Evaluate implicit and implicit terms with current state
            self._evalImplicit(lamIUk1[i])
            self._evalExplicit(lamEUk1[i])

        # Inverse position for iterate k and k+1 in storage
        # ie making the new evaluation the old for next iteration
        self.lamEU.rotate()
        self.lamIU.rotate()

    def _prolongation(self):
        """Compute prolongation, or collocation update"""
        w, dt = self.weights, self.dt
        rhs, u0 = self.rhs, self.u0
        lamIUk, lamEUk = self.lamIU[0], self.lamEU[0]
        axpy = self.axpy

        # Compute update formula
        # -- initialize with u0 term
        np.copyto(rhs, u0)
        # -- add quadrature terms
        for i in range(self.M):
            axpy(a=dt*w[i], x=lamEUk[i], y=rhs)
            axpy(a=dt*w[i], x=lamIUk[i], y=rhs)
        # -- store to state
        np.copyto(self.u, rhs)

    def step(self, dt):
        """
        Compute the next time-step solution using the time-stepper method,
        and modify to state solution vector u

        Parameters
        ----------
        dt : float
            Lenght of the current time-step.
        """

        # Initialize and/or update LHS terms, depending on dt
        if dt != self.dt:
            self._updateLHS(dt)
            self.dt = dt

        # Store U0 for the whole time step
        self._storeU0()

        # Initialize node values
        self._initSweep(iType=self.initSweep)

        # Performs sweeps
        for k in range(self.nSweep):
            if self._setSweep(k): self._updateLHS(dt)
            self._sweep()

        # Compute prolongation if needed
        if self.doProlongation():
            self._prolongation()

    @classmethod
    def imagStability(cls):
        zoom = 5
        lamBnd = -4*zoom, 4*zoom, 201
        lam = np.linspace(*lamBnd)
        lams = 1j*lam

        nSweepPrev = cls.nSweep
        cls.nSweep = 1
        solver = cls(1.0, lams, 0)
        solver.step(1.)
        cls.nSweep = nSweepPrev
        uNum = solver.u
        stab = np.abs(uNum)

        return lam[np.argwhere(stab <= 1)].max()


if __name__ == '__main__':
    # Basic testing
    import matplotlib.pyplot as plt

    nStep = 15
    dt = 2*np.pi/nStep
    times = np.linspace(0, 2*np.pi, nStep+1)

    u0 = 1.0
    lambdaE = 1j
    lambdaI = -0.1

    IMEXSDC.setParameters(
        M=3, quadType='LOBATTO', nodeDistr='LEGENDRE',
        implSweep='BEpar', explSweep='PIC', initSweep='COPY',
        forceProl=False)
    IMEXSDC.nSweep = 1
    solver = IMEXSDC(u0, lambdaI, lambdaE)

    u = [solver.u.copy()]
    for i in range(nStep):
        solver.step(dt)
        u += [solver.u.copy()]

    u = np.array(u)
    plt.plot(u.real[0], u.imag[0], 's', ms=15)
    plt.plot(u.real, u.imag, 'o-')
    uTh = u0*np.exp(times*(lambdaE+lambdaI))
    plt.plot(uTh.real, uTh.imag, '^-')

    print(IMEXSDC.imagStability())
