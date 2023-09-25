#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dedalus script simulating the viscous shallow water equations on a sphere. This
script demonstrates solving an initial value problem on the sphere. It can be
ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_sphere.py` script can be used to produce
plots from the saved data. The simulation should about 5 cpu-minutes to run.

The script implements the test case of a barotropically unstable mid-latitude
jet from Galewsky et al. 2004 (https://doi.org/10.3402/tellusa.v56i5.14436).
The initial height field balanced the imposed jet is solved with an LBVP.
A perturbation is then added and the solution is evolved as an IVP
"""
import os
import logging
import argparse
import numpy as np
from dedalus.tools.logging import add_file_handler, MPI
import dedalus.public as d3

from dedalus_sdc import SpectralDeferredCorrectionIMEX
from dedalus_sdc.simu import PROB_PARAMS, SDC_PARAMS, probInfos


TIMESTEPPERS = {
    'RK111': d3.RK111,
    'RK222': d3.RK222,
    'RK443': d3.RK443,
    'SDC': SpectralDeferredCorrectionIMEX,
    }

# Argument parser
descr = """Run the Galewsky test problem with Dedalus"""

parser = argparse.ArgumentParser(
    prog='python galewsky.py',
    description=descr,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Positional arguments (mandatory)
parser.add_argument(
    'dt', help='time-step size (in seconds)', type=int)
parser.add_argument(
    'method', help='time-integration method used',
    choices=TIMESTEPPERS.keys())
# Simulation parameters
parser.add_argument(
    '-o', '--outDir', help='directory where store the results',
    default='IT4R')
parser.add_argument(
    '-logFile', help='deactivate stdout and keep only log file',
    action='store_true')
# Problem parameters
for key, val in PROB_PARAMS.items():
    parser.add_argument(*key, **val)
# SDC parameters
for key, val in SDC_PARAMS.items():
    parser.add_argument(*key[1:], **val)

try:
    eval('__IPYTHON__')
    # When run from Spyder => default argument are given
    # with dt=600, method=SDC
    args = parser.parse_args(['600', 'SDC'])
except NameError:
    # When run from terminal => expect program arguments
    args = parser.parse_args()

# Eventually move run root directory
rootDir = os.path.dirname(args.outDir)
if rootDir != '':
    os.chdir(os.path.dirname(args.outDir))

# Logging to file (deactivate stdout if --logFile is given in argument)
if args.logFile:
    logging.root.handlers.clear()
logPrefix = os.path.join(os.path.basename(args.outDir), 'logs', 'simu')
add_file_handler(logPrefix, 'info')
if MPI.COMM_WORLD.rank == 0:
    try:
        os.remove(args.outDir+'.log')
    except Exception:
        pass
    finally:
        os.symlink(logPrefix+'_p0.log', os.path.basename(args.outDir)+'.log')

logger = logging.getLogger('script')

# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600

# Parameters
Nphi = args.nPhi
Ntheta = args.nTheta
dealias = 3/2
R = 6.37122e6 * meter
Omega = 7.292e-5 / second
nu = 1e5 * meter**2 / second / 32**2 # Hyperdiffusion matched at ell=32
g = 9.80616 * meter / second**2
H = 1e4 * meter
timestep = args.dt * second
stepPerHour = int(np.round(hour/timestep))
if not np.isclose(stepPerHour*timestep, hour):
    raise ValueError(f'timestep ({timestep}) is not a divider of one hour')
stop_sim_time = args.tEnd * hour
nStep = int(np.round(stop_sim_time/timestep))
if not np.isclose(stepPerHour*timestep, hour):
    raise ValueError(
        f'timestep ({timestep}) is not a divider of tEnd ({args.tEnd})')
dtype = np.float64

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(
    coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
h = dist.Field(name='h', bases=basis)

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))

# Time-Stepper settings
SpectralDeferredCorrectionIMEX.setParameters(
    M=args.sdcNNodes, quadType=args.sdcQuadType,
    implSweep=args.sdcImplSweep, explSweep=args.sdcExplSweep,
    initSweep=args.sdcInitSweep, forceProl=args.sdcForceProl)
SpectralDeferredCorrectionIMEX.nSweep = args.sdcNSweep

timeStepper = TIMESTEPPERS[args.method]

# Initial conditions: zonal jet
phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi
umax = 80 * meter / second
lat0 = np.pi / 7
lat1 = np.pi / 2 - lat0
en = np.exp(-4 / (lat1 - lat0)**2)
jet = (lat0 <= lat) * (lat <= lat1)
u_jet = umax / en * np.exp(1 / (lat[jet] - lat0) / (lat[jet] - lat1))
u['g'][0][jet]  = u_jet

# Initial conditions: balanced height
c = dist.Field(name='c')
problem = d3.LBVP([h, c], namespace=locals())
problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
problem.add_equation("ave(h) = 0")
solver = problem.build_solver()
solver.solve()

# Initial conditions: perturbation
lat2 = np.pi / 4
hpert = 120 * meter
alpha = 1 / 3
beta = 1 / 15
roll = int(round(Nphi/2))
h['g'] += hpert * np.cos(lat) * np.roll(np.exp(-((phi-np.pi)/alpha)**2), roll) * np.exp(-((lat2-lat)/beta)**2)

# Problem
problem = d3.IVP([u, h], namespace=locals())
problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = - u@grad(u)")
problem.add_equation("dt(h) + nu*lap(lap(h)) + H*div(u) = - div(h*u)")

# Solver
solver = problem.build_solver(timeStepper)
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler(
    args.outDir, iter=stepPerHour, max_writes=10)
# log scalar and vector fields
snapshots.add_task(h, name='height')
snapshots.add_task(u, name='velocity')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

if args.method == 'SDC':
    # log residuals of those fields
    # U field residual, DOES not work?
    snapshots.add_task(solver.timestepper.residualState[0], name='residualV')
    # h field residual
    snapshots.add_task(solver.timestepper.residualState[1], name='residualH')

    if timeStepper.logResIter:
        for k in range(solver.timestepper.nSweep):
            snapshots.add_task(solver.timestepper.residualStateIter[k][0], name=f'residualV{k}')
            snapshots.add_task(solver.timestepper.residualStateIter[k][1], name=f'residualH{k}')

# Simulation infos in the output directory
if MPI.COMM_WORLD.rank == 0:
    with open(os.path.join(args.outDir, '00_simInfos.txt'), 'w') as f:
        f.write(probInfos(
            args.dt, args.method, args.tEnd, args.nPhi, args.nTheta))
        if args.method == 'SDC':
            f.write('\n'+timeStepper.infos)

try:
    logger.info('Starting main loop')
    for n in range(nStep):
        solver.step(timestep)
        if n % 10 == 0:
            logger.info(
                f'Iteration={solver.iteration}, '
                f'Time={solver.sim_time}, '
                f'dt={timestep}')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
