#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:19:20 2022

@author: cir3339
"""
import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt

from dedalus_sdc import SpectralDeferredCorrectionIMEX

nnodes =  512 # is used in FWSW Paper

# Bases and field
coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=nnodes, bounds=(0, 1))
u = dist.Field(name='u', bases=xbasis)
p = dist.Field(name='p', bases=xbasis)


# Initial condition
x1 = 0.25
x0 = 0.75
sigma = 0.1
k = 7.2*np.pi


def p0(x, sigma=sigma):
    return np.exp(-x**2/sigma**2)

def p1(x, p0=p0, sigma=sigma, k=k):
    return p0(x)*np.cos(k*x/sigma)

# set coeff to 1 to add rapidly oscillating perturbations
def p_init(x, p0=p0, p1=p1, x0=x0, x1=x1, coeff=1.):
    return p0(x-x0) + coeff*p1(x-x1)


def CFL(U, dt, dx):
    return U*dt/dx

x = xbasis.local_grid()
p0 = p_init(x)
np.copyto(p['g'], p0)

plt.figure('Initial solution')
plt.plot(x, p['g'], label='Real space')
plt.plot(x, p['c'], 'o', label='Coefficient space')
plt.legend()
plt.grid()

# Problem
c_s = 1.0              # speed of sound
U = 0.05                  # mean flow
dx = lambda f: d3.Differentiate(f, coords['x'])
problem = d3.IVP([u, p], namespace={
    'c_s': c_s, 'U': U, 'dx': dx, 'u': u, 'p': p})
problem.add_equation("dt(u) + c_s*dx(p) = - U*dx(u)")
problem.add_equation("dt(p) + c_s*dx(u) = - U*dx(p)")


# setup SDC
SpectralDeferredCorrectionIMEX.setParameters(
    M=3, quadType='RADAU-RIGHT', nodeDistr='LEGENDRE',
    implSweep='BE', explSweep='PIC', initSweep='COPY',
    forceProl=True, calcResidual=True) # copy or qdelta
# add switch for adding prolongation or not?

nSweep = 2
timeStepper = SpectralDeferredCorrectionIMEX
timeStepper.nSweep = nSweep

# Build solver
solver = problem.build_solver(timeStepper)
solver.stop_sim_time = 3
name = timeStepper.__name__
nsteps = 20*154 # 154 is used in FWSW Paper
dt = solver.stop_sim_time/(nsteps)

# animation
storage = np.zeros([2, nsteps+1, nnodes])
storage[0, 0] = u['g']
storage[1, 0] = p['g']

for _ in range(nsteps):
    solver.step(dt)
    storage[0, _+1] = u['g']
    storage[1, _+1] = p['g']

plt.figure('Final Solution')
plt.plot(x, u['g'], label='u')
plt.plot(x, p['g'], label='p')
plt.legend()
plt.savefig("acousticAdv_finalSolution.pdf", bbox_inches="tight")

plt.figure('Instability')
plt.plot(u['c'], label='u', marker='o', linestyle='')
plt.plot(p['c'], label='p', marker = 'o', linestyle='')
plt.legend()
plt.savefig("acousticAdv_instability.pdf", bbox_inches="tight")

print('fast CFL: ' + str(CFL(c_s, dt, 1/nnodes)))
print('slow CFL: ' + str(CFL(U, dt, 1/nnodes)))

# ani = animate(x, storage)
# ani.run()
