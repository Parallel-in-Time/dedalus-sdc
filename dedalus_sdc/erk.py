#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 09:15:54 2022
"""
from dedalus.core.system import CoeffSystem
from dedalus.tools.array import csr_matvecs

import numpy as np
from scipy.linalg import blas
from collections import deque

class ERK4:
    """DOCTODO"""

    steps = 1
    stages = 4

    def __init__(self, solver):

        self.solver = solver
        self.RHS = CoeffSystem(solver.subproblems, dtype=solver.dtype)

        # Create coefficient systems for steps history
        self.X0 = CoeffSystem(solver.subproblems, dtype=solver.dtype)
        self.LX = CoeffSystem(solver.subproblems, dtype=solver.dtype)
        self.K = {i : CoeffSystem(solver.subproblems, dtype=solver.dtype)
                  for i in range(1, self.stages)}

        # Attributes
        self._iteration = 0
        self._LHS_params = None
        self.axpy = blas.get_blas_funcs('axpy', dtype=solver.dtype)
        self.update_LHS = True

    def step(self, dt, wall_time):
        """Advance solver by one timestep."""

        # Solver references
        solver = self.solver
        subproblems = solver.subproblems
        evaluator = solver.evaluator
        state_fields = solver.state
        F_fields = solver.F
        sim_time_0 = solver.sim_time
        iteration = solver.iteration
        STORE_EXPANDED_MATRICES = solver.store_expanded_matrices

        # Other references
        RHS = self.RHS
        X0 = self.X0
        K = self.K
        LX = self.LX
        axpy = self.axpy
        update_LHS = self.update_LHS

        # Initialize LFS_solvers attributes
        # --> here only one, since Mass matrix M is constant
        if update_LHS:
            for sp in subproblems:
                sp.LHS_solver = None

        # Ensure coeff space before subsystem gathers
        for field in state_fields:
            field.require_coeff_space()

        # Store X0 from state (copy, after multiplying by pre_left)
        X0.data.fill(0)
        for sp in subproblems:
            spX = sp.gather(state_fields)
            csr_matvecs(sp.pre_left, spX, X0.get_subdata(sp))

        # RK4 stage 1 ---------------------------------------------------------
        # -- K1 = f(t, X0) = M^{-1} * ( F(X0, t) - L * X0 )
        # -- X1 = X0 + dt/2 * K1

        # Compute L * X0
        LX.data.fill(0)
        for sp in subproblems:
            spX = sp.gather(state_fields)
            csr_matvecs(sp.L_min, spX, LX.get_subdata(sp))

        # RHS := F(X0, t)
        solver.sim_time = sim_time_0
        evaluator.evaluate_scheduled(
            wall_time=wall_time, timestep=dt, sim_time=solver.sim_time,
            iteration=iteration)
        RHS.data.fill(0)
        for sp in subproblems:
            spX = sp.gather(F_fields)
            csr_matvecs(sp.pre_left, spX, RHS.get_subdata(sp))

        # RHS -= L * X0
        if RHS.data.size:
            axpy(a=-1, x=LX.data, y=RHS.data)

        # Ensure coeff space before subsystem scatters
        for field in state_fields:
            field.preset_layout('c')

        # Solve M * K1 = RHS and compute X1
        for sp in subproblems:
            # Build LHS (only for the first time step)
            if update_LHS:
                if STORE_EXPANDED_MATRICES:
                    np.copyto(sp.LHS.data, sp.M_exp.data)
                else:
                    sp.LHS = sp.M_min @ sp.pre_right
                sp.LHS_solver = solver.matsolver(sp.LHS, solver)
                self.update_LHS = False
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)[:sp.LHS.shape[0]]
            spX = sp.LHS_solver.solve(spRHS)
            # Save to K1
            np.copyto(K[1].get_subdata(sp), spX)
            # Compute X1
            spX *= dt/2
            spX += X0.get_subdata(sp)
            # Make output buffer including invalid components for scatter
            spX2 = np.zeros(
                (sp.pre_right.shape[0], len(sp.subsystems)),
                dtype=spX.dtype)
            # Store X1 to state_fields
            csr_matvecs(sp.pre_right, spX, spX2)
            sp.scatter(spX2, state_fields)

        # RK4 stage 2 ---------------------------------------------------------
        # -- K2 = f(t+dt/2, X1) = M^{-1} * ( F(X1, t+dt/2) - L * X1 )
        # -- X2 = X0 + dt/2 * K2

        # Ensure coeff space before subsystem gathers
        for field in state_fields:
            field.require_coeff_space()

        # RHS := F(X1, t+dt/2)
        solver.sim_time = sim_time_0 + dt/2
        evaluator.evaluate_group(
            'F', wall_time=wall_time, timestep=dt, sim_time=solver.sim_time,
            iteration=iteration)
        RHS.data.fill(0)
        for sp in subproblems:
            spX = sp.gather(F_fields)
            csr_matvecs(sp.pre_left, spX, RHS.get_subdata(sp))

        # Compute L * X1
        LX.data.fill(0)
        for sp in subproblems:
            spX = sp.gather(state_fields)
            csr_matvecs(sp.L_min, spX, LX.get_subdata(sp))

        # RHS -= L * X1
        if RHS.data.size:
            axpy(a=-1, x=LX.data, y=RHS.data)

        # Ensure coeff space before subsystem scatters
        for field in state_fields:
            field.preset_layout('c')

        # Solve M * K2 = RHS and compute X2
        for sp in subproblems:
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)[:sp.LHS.shape[0]]
            spX = sp.LHS_solver.solve(spRHS)
            # Save to K2
            np.copyto(K[2].get_subdata(sp), spX)
            # Compute X2
            spX *= dt/2
            spX += X0.get_subdata(sp)
            # Make output buffer including invalid components for scatter
            spX2 = np.zeros(
                (sp.pre_right.shape[0], len(sp.subsystems)),
                dtype=spX.dtype)
            # Store X2 to state_fields
            csr_matvecs(sp.pre_right, spX, spX2)
            sp.scatter(spX2, state_fields)

        # RK4 stage 3 ---------------------------------------------------------
        # -- K3 = f(t+dt/2, X2) = M^{-1} * ( F(X2, t+dt/2) - L * X2 )
        # -- X3 = X0 + dt * K2

        # Ensure coeff space before subsystem gathers
        for field in state_fields:
            field.require_coeff_space()

        # Compute L * X2
        LX.data.fill(0)
        for sp in subproblems:
            spX = sp.gather(state_fields)
            csr_matvecs(sp.L_min, spX, LX.get_subdata(sp))

        # RHS := F(X2, t+dt/2)
        solver.sim_time = sim_time_0 + dt/2
        evaluator.evaluate_group(
            'F', wall_time=wall_time, timestep=dt, sim_time=solver.sim_time,
            iteration=iteration)
        RHS.data.fill(0)
        for sp in subproblems:
            spX = sp.gather(F_fields)
            csr_matvecs(sp.pre_left, spX, RHS.get_subdata(sp))

        # RHS -= L * X2
        if RHS.data.size:
            axpy(a=-1, x=LX.data, y=RHS.data)

        # Ensure coeff space before subsystem scatters
        for field in state_fields:
            field.preset_layout('c')

        # Solve M * K3 = RHS and compute X3
        for sp in subproblems:
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)[:sp.LHS.shape[0]]
            spX = sp.LHS_solver.solve(spRHS)
            # Save to K3
            np.copyto(K[3].get_subdata(sp), spX)
            # Compute X3
            spX *= dt
            spX += X0.get_subdata(sp)
            # Make output buffer including invalid components for scatter
            spX2 = np.zeros(
                (sp.pre_right.shape[0], len(sp.subsystems)),
                dtype=spX.dtype)
            # Store X3 to state_fields
            csr_matvecs(sp.pre_right, spX, spX2)
            sp.scatter(spX2, state_fields)

        # RK4 stage 4 ---------------------------------------------------------
        # -- K4 = f(t+dt, X3) = M^{-1} * ( F(X3, t+dt) - L * X3 )
        # -- XNext = X0 + dt/6 * (K1 + 2*K2 + 2*K3 + K4)

        # RHS := F(X3, t+dt)
        solver.sim_time = sim_time_0 + dt
        evaluator.evaluate_group(
            'F', wall_time=wall_time, timestep=dt, sim_time=solver.sim_time,
            iteration=iteration)
        RHS.data.fill(0)
        for sp in subproblems:
            spX = sp.gather(F_fields)
            csr_matvecs(sp.pre_left, spX, RHS.get_subdata(sp))

        # Compute L * X3
        LX.data.fill(0)
        for sp in subproblems:
            spX = sp.gather(state_fields)
            csr_matvecs(sp.L_min, spX, LX.get_subdata(sp))

        # RHS -= L * X3
        if RHS.data.size:
            axpy(a=-1, x=LX.data, y=RHS.data)

        # Prepare K data
        K[2].data *= 2
        K[3].data *= 2

        # Ensure coeff space before subsystem scatters
        for field in state_fields:
            field.preset_layout('c')

        # Solve M * K4 = RHS and compute XNext
        for sp in subproblems:
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)[:sp.LHS.shape[0]]
            spX = sp.LHS_solver.solve(spRHS)
            # Compute XNext
            spX += K[1].get_subdata(sp)
            spX += K[2].get_subdata(sp)
            spX += K[3].get_subdata(sp)
            spX *= dt/6
            spX += X0.get_subdata(sp)
            # Make output buffer including invalid components for scatter
            spX2 = np.zeros(
                (sp.pre_right.shape[0], len(sp.subsystems)),
                dtype=spX.dtype)
            # Store XNext to state_fields
            csr_matvecs(sp.pre_right, spX, spX2)
            sp.scatter(spX2, state_fields)
