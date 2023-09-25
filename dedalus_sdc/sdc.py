#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 14:00:36 2022
"""
# Python imports
import numpy as np
from scipy.linalg import blas
from collections import deque

# Dedalus import
from dedalus.core.system import CoeffSystem
from dedalus.tools.array import csr_matvecs

from dedalus_sdc.core import IMEXSDCCore

class SpectralDeferredCorrectionIMEX(IMEXSDCCore):

    steps = 1
    infos = ""

    # -------------------------------------------------------------------------
    # Instance methods
    # -------------------------------------------------------------------------
    def __init__(self, solver):

        # Store class attributes as instance attributes
        SpectralDeferredCorrectionIMEX.infos = self.getInfos()

        # Store solver as attribute
        self.solver = solver

        # need this for solver.log_stats()
        self.stages = self.M
        # Create coefficient systems for steps history
        c = lambda: CoeffSystem(solver.subproblems, dtype=solver.dtype)
        self.MX0, self.RHS = c(), c()
        self.LX = deque([[c() for _ in range(self.M)] for _ in range(2)])
        self.F = deque([[c() for _ in range(self.M)] for _ in range(2)])


        if not self.leftIsNode:
            self.F0, self.LX0 = c(), c()

        # Instantiate M solver, needed only when prolongation is used
        if self.doProlongation or self.calcResidual:
            for sp in solver.subproblems:
                if solver.store_expanded_matrices:
                    np.copyto(sp.LHS.data, sp.M_exp.data)
                else:
                    sp.LHS = sp.M_min @ sp.pre_right
                sp.M_solver = solver.matsolver(sp.LHS, solver)

        if self.calcResidual:
            self.Fr = [c() for _ in range(self.M)]
            self.LXr = [c() for _ in range(self.M)]
            # permanently store node information for calculating the residual

            # function that creates fields
            f = self.solver.problem.dist.Field
            vf = self.solver.problem.dist.VectorField

            # state contains the problems fields
            state = self.solver.state

            # identify vector fields/there is probably some better way
            # that I haven't figured out yet
            id_vector = [len(x) for x in [_.data.shape for _ in state]]
            id_vector = np.asarray(id_vector)
            id_vector -= min(id_vector)
            self.id_vector = id_vector
            # get basis of every field in the problem
            # this may restrict redidual to 1d problems (only 0 index)...
            basis = [_.get_basis(_.dist.get_coordsystem(0)) for _ in state]

            #initialise fields for every node to calculate residual
            self.U = [[None for x in range(len(state))] for _ in range(self.M)]
            self.residualState = [None for _ in state]
            if self.modeControl:
                self.setTol = True
                self.tol = 1e-6
                # this stores maximum residual for each individual component. this then gets multiplied with
                # self.tol to set an individual residual tolerance
                self.tols = [None for _ in state]
                self.tolState = [float(-1) for x in range(len(state))]
                self.modeState = [[None for x in range(len(state))] for _ in range(self.M)]
                self.modeDiff = deque([[[None for x in range(len(state))] for _ in range(self.M)] for b in range(self.nSweep)])
                self.mask = [[None for _ in range(len(self.residualState))] for _ in range(self.M)]
                self.maskCheck = [None for _ in range(self.M)]
                self.maskInv = [[None for _ in range(len(self.residualState))] for _ in range(self.M)]
                self.finalMask = [None for _ in range(self.M)]
                self.finalMaskInv = [None for _ in range(self.M)]

            if self.logResIter:
                self.residualStateIter = [[None for x in range(len(state))]
                                          for _ in range(self.nSweep)]
            for k in range(self.nSweep):
                for j in range(len(state)):
                    coords = state[j].dist.get_coordsystem(0)
                    basis = state[j].get_basis(coords)
                    if id_vector[j]:
                        self.residualState[j] = vf(coords, bases=basis)
                    else:
                        self.residualState[j] = f(basis)
                        if self.modeControl:
                            self.shape = self.residualState[j]['c'][:].shape
                            self.nModes = np.prod(self.shape)

                    for i in range(self.M):
                        if id_vector[j]:
                            if self.modeControl:
                                self.modeState[i][j] = vf(coords, bases=basis)
                                self.modeDiff[k][i][j] = vf(coords, bases=basis)
                            self.U[i][j] = vf(coords, bases=basis)
                        else:
                            if self.modeControl:
                                self.modeState[i][j] = f(basis)
                                self.modeDiff[k][i][j] = f(basis)
                            self.U[i][j] = f(basis)
                    if self.logResIter:
                        for i in range(self.nSweep):
                            if id_vector[j]:
                                self.residualStateIter[i][j] =vf(coords, bases=basis)
                            else:
                                self.residualStateIter[i][j] =f(basis)


            #self.U = [[f(b) for b in basis] for _ in range(self.M)]
            #self.residualState = [f(b) for b in basis]
            self.MXi = c()

            if self.logResidual:
                self.residualLog = np.zeros([self.nSweep, self.M])
                if self.modeControl:
                    self.nModesShut = np.zeros([self.nSweep, self.M])

        # Attributes
        self.axpy = blas.get_blas_funcs('axpy', dtype=solver.dtype)
        self.dt = None
        self.firstEval = True
        self.isInit = False

    @property
    def M(self):
        return len(self.nodes)

    @property
    def rightIsNode(self):
        return np.allclose(self.nodes[-1], 1.0)

    @property
    def leftIsNode(self):
        return np.allclose(self.nodes[0], 0.0)

    @property
    def doProlongation(self):
        return not self.rightIsNode or self.forceProl

    @property
    def doResidual(self):
        return not self.isInit and self.calcResidual

    def _computeMX0(self, state, MX0):
        """
        Compute MX0 term used in RHS of both initStep and sweep methods

        Update the MX0 attribute of the timestepper object.
        """
        self._requireStateCoeffSpace(state)

        # Compute and store MX0
        MX0.data.fill(0)
        for sp in self.solver.subproblems:
            spX = sp.gather(state)
            csr_matvecs(sp.M_min, spX, MX0.get_subdata(sp))

    def _updateLHS(self, dt, init=False):
        """Update LHS and LHS solvers for each subproblem

        Parameters
        ----------
        dt : float
            Time-step for the updated LHS.
        init : bool, optional
            Wether or not initialize the LHS_solvers attribute for each
            subproblem. The default is False.
        """
        # Attribute references
        qI = self.QDeltaI
        solver = self.solver

        # Update LHS and LHS solvers for each subproblems
        for sp in solver.subproblems:
            if init:
                # Eventually instanciate list of solver (ony first time step)
                sp.LHS_solvers = [None] * self.M
            for i in range(self.M):
                if solver.store_expanded_matrices:
                    np.copyto(sp.LHS.data,
                              sp.M_exp.data + dt*qI[i, i]*sp.L_exp.data)
                else:
                    sp.LHS = (sp.M_min + dt*qI[i, i]*sp.L_min) @ sp.pre_right
                sp.LHS_solvers[i] = solver.matsolver(sp.LHS, solver)

    def _evalLX(self, LX):
        """
        Evaluate LX using the current state, and store it

        Parameters
        ----------
        LX : dedalus.core.system.CoeffSystem
            Where to store the evaluated fields.

        Returns
        -------
        None.

        """
        # Attribute references
        solver = self.solver

        self._requireStateCoeffSpace(solver.state)

        # Evaluate matrix vector product and store
        LX.data.fill(0)
        for sp in solver.subproblems:
            spX = sp.gather(solver.state)
            csr_matvecs(sp.L_min, spX, LX.get_subdata(sp))

    def _evalF(self, F, time, dt, wall_time):
        """
        Evaluate the F operator from the current solver state

        Note
        ----
        After evaluation, state fields are left in grid space

        Parameters
        ----------
        time : float
            Time of evaluation.
        F : dedalus.core.system.CoeffSystem
            Where to store the evaluated fields.
        dt : float
            Current time step.
        wall_time : float
            Current wall time.
        """

        solver = self.solver
        # Evaluate non linear term on current state
        t0 = solver.sim_time
        solver.sim_time = time
        if self.firstEval:
            solver.evaluator.evaluate_scheduled(
                wall_time=wall_time, timestep=dt, sim_time=time,
                iteration=solver.iteration)
            self.firstEval = False
        else:
            solver.evaluator.evaluate_group(
                'F', wall_time=wall_time, timestep=dt, sim_time=time,
                iteration=solver.iteration)
        # Initialize F with zero values
        F.data.fill(0)
        # Store F evaluation
        for sp in solver.subproblems:
            spX = sp.gather(solver.F)
            csr_matvecs(sp.pre_left, spX, F.get_subdata(sp))
        # Put back initial solver simulation time
        solver.sim_time = t0

    def _solveAndStoreState(self, iNode):
        """
        Solve LHS * X = RHS using the LHS associated to a given node,
        and store X into the solver state.
        It uses the current RHS attribute of the object.

        Parameters
        ----------
        iNode : int
            Index of the nodes.
        """
        # Attribute references
        solver = self.solver
        RHS = self.RHS

        self._presetStateCoeffSpace(solver.state)

        if self.doResidual:
            self._presetStateCoeffSpace(self.U[iNode])

        # Solve and store for each subproblem
        for sp in solver.subproblems:
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)[:sp.LHS.shape[0]]
            # Solve using LHS of the node
            spX = sp.LHS_solvers[iNode].solve(spRHS)
            # Make output buffer including invalid components for scatter
            spX2 = np.zeros(
                (sp.pre_right.shape[0], len(sp.subsystems)),
                dtype=spX.dtype)
            # Store X to state_fields
            csr_matvecs(sp.pre_right, spX, spX2)
            sp.scatter(spX2, solver.state)
            if self.doResidual:
                #print('U was set')
                sp.scatter(spX2, self.U[iNode])

    def _requireStateCoeffSpace(self, state):
        """Transform current state fields in coefficient space.
        If already in coefficient space, doesn't do anything."""
        for field in state:
            field.require_coeff_space()

    def _presetStateCoeffSpace(self, state):
        """Allow to write fields in coefficient space into current state
        fields, without transforming current state in coefficient space."""
        for field in state:
            field.preset_layout('c')

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
        tau, qI, qE = self.nodes, self.QDeltaI, self.QDeltaE
        solver = self.solver
        t0, dt, wall_time = solver.sim_time, self.dt, self.wall_time
        RHS, MX0, Fk, LXk = self.RHS, self.MX0, self.F[0], self.LX[0]
        if not self.leftIsNode:
            F0, LX0 = self.F0, self.LX0
        axpy = self.axpy
        self.isInit = False

        if iType == 'QDELTA':
            #print('DO INITIAL SWEEP')
            # Prepare initial field evaluation
            if not self.leftIsNode:
                self._evalF(F0, t0, dt, wall_time)
                F0.data *= dt*self.dtauE
                if self.dtauI != 0.0:
                    self._evalLX(LX0)
                    axpy(a=-dt*self.dtauI, x=LX0.data, y=F0.data)

            # Loop on all quadrature nodes
            for i in range(self.M):
                # Build RHS
                if RHS.data.size:
                    # Initialize with MX0 term
                    np.copyto(RHS.data, MX0.data)
                    # Add initial field evaluation
                    if not self.leftIsNode:
                        RHS.data += F0.data
                    # Add F and LX terms (already computed)
                    for j in range(i):
                        axpy(a=dt*qE[i, j], x=Fk[j].data, y=RHS.data)
                        axpy(a=-dt*qI[i, j], x=LXk[j].data, y=RHS.data)
                # Solve system and store node solution in solver state
                #print('DO SOLVEANDSTORE INITIAL')
                self._solveAndStoreState(i)
                #print((self.solver.state[0]['c'] == self.U[i][0]['c'])[0:4])
                #print(self.solver.state[0]['c'])
                #print(self.U[i][0]['c'])
                # Evaluate and store LX with current state
                self._evalLX(LXk[i])
                # Evaluate and store F(X, t) with current state
                self._evalF(Fk[i], t0+dt*tau[i], dt, wall_time)

                if self.modeControl:
                    self._setModeStates(i)

        elif iType == 'COPY':
            self._evalLX(LXk[0])
            self._evalF(Fk[0], t0, dt, wall_time)
            for i in range(1, self.M):
                np.copyto(LXk[i].data, LXk[0].data)
                np.copyto(Fk[i].data, Fk[0].data)

            if self.calcResidual:
                self._presetUi()
                for i in range(self.M):
                    if self.modeControl:
                        self._setModeStates(i)

        else:
            raise NotImplementedError(f'iType={iType}')

        self.isInit = False

    def _sweep(self):
        """Perform a sweep for the current time-step"""
        # Attribute references
        tau, qI, qE, q = self.nodes, self.QDeltaI, self.QDeltaE, self.Q
        solver = self.solver
        t0, dt, wall_time = solver.sim_time, self.dt, self.wall_time
        RHS, MX0 = self.RHS, self.MX0
        Fk, LXk, Fk1, LXk1 = self.F[0], self.LX[0], self.F[1], self.LX[1]
        axpy = self.axpy

        # Loop on all quadrature nodes
        for i in range(self.M):
            # Build RHS
            if RHS.data.size:
                # Initialize with MX0 term
                np.copyto(RHS.data, MX0.data)
                # Add quadrature terms
                for j in range(self.M):
                    axpy(a=dt*q[i, j], x=Fk[j].data, y=RHS.data)
                    axpy(a=-dt*q[i, j], x=LXk[j].data, y=RHS.data)
                # Add F and LX terms from iteration k+1
                for j in range(i):
                    axpy(a=dt*qE[i, j], x=Fk1[j].data, y=RHS.data)
                    axpy(a=-dt*qI[i, j], x=LXk1[j].data, y=RHS.data)
                # Add F and LX terms from iteration k
                for j in range(i):
                    axpy(a=-dt*qE[i, j], x=Fk[j].data, y=RHS.data)
                    axpy(a=dt*qI[i, j], x=LXk[j].data, y=RHS.data)
                axpy(a=dt*qI[i, i], x=LXk[i].data, y=RHS.data)
            # Solve system and store node solution in solver state
            self._solveAndStoreState(i)
            # revert state back to previous iteration IF mask condition is met
            if self.modeControl:
                self._applyModeControl(i)
                self._setModeStates(i)

            # Evaluate and store LX with current state
            self._evalLX(LXk1[i])
            # Evaluate and store F(X, t) with current state
            self._evalF(Fk1[i], t0+dt*tau[i], dt, wall_time)


        # Inverse position for iterate k and k+1 in storage 
        # ie making the new evaluation the old for next iteration
        self.F.rotate()
        self.LX.rotate()
        # store iteration
        if self.modeControl:
            self.modeDiff.rotate(-1)

    def _residual(self, k):
        from mpi4py import MPI
        """
        Compute residual of SDC Sweep.
        TODO: 1. implement residual - done
        TODO: 2. check if residual is working correctly - done # !?
        """
        solver = self.solver
        q = self.Q
        dt = self.dt
        RHS, Fk, LXk = self.RHS, self.F[0], self.LX[0]
        MX0, MXi = self.MX0, self.MXi
        axpy = self.axpy
        self.residual = [0 for _ in range(self.M)]
        # Build RHS
        if RHS.data.size:

            # Add quadrature terms
            for i in range(self.M):
            # Initialize with MX0 term
                np.copyto(RHS.data, MX0.data)
                for j in range(self.M):
                    axpy(a=dt*q[i,j], x=Fk[j].data, y=RHS.data)
                    axpy(a=-dt*q[i,j], x=LXk[j].data, y=RHS.data)
                #subtract SDC solution
                self._computeMX0(self.U[i], MXi)
                axpy(a=-1, x=MXi.data, y=RHS.data)


                self._presetStateCoeffSpace(self.residualState)
                if self.logResIter:
                    self._presetStateCoeffSpace(self.residualStateIter[k])

                # Solve y and store for each subproblem
                for sp in solver.subproblems:
                    # Slice out valid subdata, skipping invalid components
                    spRHS = RHS.get_subdata(sp)[:sp.LHS.shape[0]]
                    # Solve using LHS of the node
                    spX = sp.M_solver.solve(spRHS)
                    # Make output buffer including invalid components for scatter
                    spX2 = np.zeros(
                        (sp.pre_right.shape[0], len(sp.subsystems)),
                        dtype=spX.dtype)
                    # Store X to state_fields
                    csr_matvecs(sp.pre_right, spX, spX2)
                    sp.scatter(spX2, self.residualState)
                    if self.logResIter:
                        sp.scatter(spX2, self.residualStateIter[k])

                for j in range(len(self.residualState)):
                    mx = np.max(np.abs(self.residualState[j]['c']))
                    self.residual[i] = mx if self.residual[i] < mx else self.residual[i]
                    if ((k == 0 and self.modeControl) and self.setTol):
                        #print(f'k: {k}, m: {i}, j: {j}')
                        self.tolState[j] = mx if self.tolState[j] < mx else self.tolState[j]
                        self.tols[j]=self.tolState[j]*self.tol
                        temp = MPI.COMM_WORLD.allreduce(self.tols[j], op=MPI.MAX)
                        self.tols[j]=temp
                        MPI.COMM_WORLD.Barrier()

                if self.modeControl:
                    #print('_______')
                    #print(self.residual[i])
                    self._calcMask(i, k)
            if (self.modeControl and self.setTol):
                self.setTol=False

    def _logResidual(self, k):
        self.residualLog[k] = self.residual

    def _calcMask(self, m, k):
        self.nModesShut[k, m] = 0
        variable_counter = 0
        self.maskCheck[m] = np.zeros(self.shape)

        for j in range(len(self.residualState)):
            self.tols[j] = 1e-9
            temp = np.where(np.abs(self.residualState[j]['c']) < self.tols[j], 0, 1) #

            try:
                if k == 0:
                    if self.id_vector[j]:
                        vector_dim = tuple([len(temp)])
                        shape = vector_dim + self.shape
                        self.mask[m][j] = np.ones(shape)
                    else:
                        self.mask[m][j] = np.ones(self.shape)
                if self.mask[m][j] == None:
                    if self.id_vector[j]:
                        vector_dim = tuple(list(len(temp)))
                        shape = vector_dim + self.shape
                        self.mask[m][j] = np.ones(shape)
                    else:
                        self.mask[m][j] = np.ones(self.shape)
            except:
                pass

            # make sure modes switched off can NOT be switched on again
            #print(np.shape(self.mask[m][j]))
            #print(np.shape(temp))
            #print('____________')
            # This prevents stuff working when using QDELTA for initial s

            #self.mask[m][j] = self.mask[m][j]*temp

            self.mask[m][j] = temp
            # allow switching on


            # make sure neighbouring elements in real coefficient space are
            # linked to prevent phase locking
            # in 2D complex we need to link stuff as well but I dont know how... yet!

            #if self.solver.dtype == np.float64:
            #    #print('WARNING: applyied neighbour coupling which will only work in 1D!')
            #    comp = self.mask[m][j][0::2] - self.mask[m][j][1::2]
            #    add02 = np.where(comp == -1, 1, 0)
            #    add12 = np.where(comp == 1, 1, 0)
            #    self.mask[m][j][0::2] += add02
            #    self.mask[m][j][1::2] += add12

            # mask check: make sure modes in nodes are coupled
            if self.id_vector[j]:
                for h in range(len(self.mask[m][j])):
                    self.maskCheck[m] += self.mask[m][j][h]
                    variable_counter += 1
            else:
                self.maskCheck[m] += self.mask[m][j]
                variable_counter += 1
        if m == self.M-1:
            maskCheck = np.sum(self.maskCheck, axis=0)
            fullMask = np.where(maskCheck > 0, 1, 0)
            for i in range(self.M):
                for j in range(len(self.residualState)):
                    if self.id_vector[j]:
                        for h in range(len(self.mask[i][j])):
                            self.mask[i][j][h] = fullMask
                    else:
                        self.mask[i][j] = fullMask
                    self.maskInv[i][j] = abs(self.mask[i][j]-1)

                    if self.id_vector[j]:
                        for h in range(len(self.mask[i][j])):
                            self.nModesShut[k, i] += np.sum(self.maskInv[i][j][h])
                    else:
                        self.nModesShut[k, i] += np.sum(self.maskInv[i][j])
                    #pass
                #    self.maskInv[m][k] = abs(self.mask[m][k]-1)
                self.nModesShut[k, i] = self.nModesShut[k, i]/(variable_counter*self.nModes)

        #' OLD'
        #for j in range(len(self.residualState)):
        #    self.mask[m][j] = np.where(self.maskCheck[m] > 0, 1, 0)
        #    self.maskInv[m][j] = abs(self.mask[m][j]-1)
        #    if self.id_vector[j]:
        #        for h in range(len(mask[m][j])):
        #            self.nModesShut[k, m] += np.sum(self.maskInv[m][j][h])
        #    else:
        #        self.nModesShut[k, m] += np.sum(self.maskInv[m][j])
        #    #pass
        ##    self.maskInv[m][k] = abs(self.mask[m][k]-1)
        #self.nModesShut[k, m] = self.nModesShut[k, m]/(variable_counter*self.nModes)

    def _presetUi(self):
        #needed for residual as stopping criterion IF init sweep is COPY
        # populates the Ui with solver.state
        for i in range(self.M):
            for j in range(len(self.solver.state)):
                if self.id_vector[j]:
                    for k in range(len(self.solver.state[j]['c'])):
                        np.copyto(self.U[i][j]['c'][k], self.solver.state[j]['c'][k])
                else:
                        np.copyto(self.U[i][j]['c'], self.solver.state[j]['c'])

    def _setModeStates(self, m):
        #pass
        # stores previous timesteps which are usually not needed since we only 
        # need the function evaluations of previous steps
        for j in range(len(self.solver.state)):
            if self.id_vector[j]:
                for k in range(len(self.U[m][j]['c'])):
                    np.copyto(self.modeState[m][j]['c'][k], self.U[m][j]['c'][k])
            else:
                np.copyto(self.modeState[m][j]['c'], self.U[m][j]['c'])

    def _applyModeControl(self, m):
        #from copy import deepcopy
        for j in range(len(self.solver.state)):
            if self.id_vector[j]:
                # falls fehler liegt er evtl an modeState, oder der Addition von modeState und U[i]
                for k in range(len(self.solver.state[j]['c'])):
                    self.solver.state[j]['c'][k] = self.U[m][j]['c'][k]*self.mask[m][j][k] + self.modeState[m][j]['c'][k]*self.maskInv[m][j][k]
                    #self.solver.state[j]['c'][k] = self.U[i][j]['c'][k]*0 + self.modeState[i][j]['c'][k]*1
                    self.U[m][j]['c'][k] = self.solver.state[j]['c'][k]
            else:
                #print(f'variable is {j}, node is {m}')
                #print(f'number of kept modes {np.sum(self.mask[m][j])}')
                #print((self.solver.state[j]['c'] == self.U[m][j]['c']*self.mask[m][j] + self.modeState[m][j]['c']*self.maskInv[m][j])[0:4])
                self.solver.state[j]['c'] = self.U[m][j]['c']*self.mask[m][j] + self.modeState[m][j]['c']*self.maskInv[m][j]
                #self.solver.state[j]['c'] = self.U[m][j]['c']*0 + self.modeState[m][j]['c']*1
                self.U[m][j]['c'] = self.solver.state[j]['c']
            # index 0 is ok because modeDiff gets rotated before _applyModeControl
            self.modeDiff[0][m][j]['c'] =  self.U[m][j]['c'] - self.modeState[m][j]['c']
            # substitutes setmodestates... but not everywhere ...
            #self.modeState[m][j]['c'] = self.solver.state[j]['c']

    def _prolongation(self):
        """Compute prolongation (needed if last node != 1)"""
        # Attribute references
        solver = self.solver
        w, dt = self.weights, self.dt
        RHS, MX0, Fk, LXk = self.RHS, self.MX0, self.F[0], self.LX[0]
        axpy = self.axpy

        # Build RHS
        if RHS.data.size:
            # Initialize with MX0 term
            np.copyto(RHS.data, MX0.data)
            # Add quadrature terms
            for i in range(self.M):
                axpy(a=dt*w[i], x=Fk[i].data, y=RHS.data)
                axpy(a=-dt*w[i], x=LXk[i].data, y=RHS.data)

        self._presetStateCoeffSpace(solver.state)

        # Solve and store for each subproblem
        for sp in solver.subproblems:
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)[:sp.LHS.shape[0]]
            # Solve using LHS of the node
            spX = sp.M_solver.solve(spRHS)
            # Make output buffer including invalid components for scatter
            spX2 = np.zeros(
                (sp.pre_right.shape[0], len(sp.subsystems)),
                dtype=spX.dtype)
            # Store X to state_fields
            csr_matvecs(sp.pre_right, spX, spX2)
            sp.scatter(spX2, solver.state)

    def step(self, dt, wall_time):
        """
        Compute the next time-step solution using the time-stepper method,
        and modify to state field of the solver

        Note
        ----
        State fields should be left in grid space after at the end of the step.

        Parameters
        ----------
        dt : float
            Lenght of the current time-step.
        wall_time : float
            Current wall time for the simulation.
        """
        self.wall_time = wall_time

        # Initialize and/or update LHS terms, depending on dt
        if dt != self.dt:
            self._updateLHS(dt, init=self.dt is None)
            self.dt = dt

        # Compute MX0 for the whole time step
        self._computeMX0(self.solver.state, self.MX0)
        # Initialize node values
        self._initSweep(iType=self.initSweep)
        # Performs sweeps
        self.residualLog[:] = np.nan
        for k in range(self.nSweep):
            self.k = k
            self._setSweep(k)
            if self._setSweep(k): self._updateLHS(dt)
            if self.calcResidual:
                self._residual(k)
                if self.logResidual:
                    self._logResidual(k)
                if self.modeControl:
                    if k == 0:
                        for m in range(self.M):
                            #print('HELLO')
                            self._applyModeControl(m)
                            #print('DONE')
                            t0, dt, wall_time, tau = self.solver.sim_time, self.dt, self.wall_time, self.nodes
                            Fk, LXk = self.F[0], self.LX[0]
                            self._evalLX(LXk[m])
                            self._evalF(Fk[m], t0+dt*tau[m], dt, wall_time)
            #if np.max(self.residual) < 1e-10:
            #    print(k)
            #    break
            self._sweep()

        # Compute prolongation if needed
        if self.doProlongation:
            self._prolongation()

        # Update simulation time and reset evaluation tag
        self.solver.sim_time += dt
        self.firstEval = True
