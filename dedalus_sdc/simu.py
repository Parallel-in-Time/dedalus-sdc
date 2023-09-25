#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:04:27 2022

Utility module for handling and monitoring galewsky simulation run

@author: telu
"""
import os
import sys
import time
import glob
import psutil
import subprocess
from datetime import datetime
import pandas as pd

dirname = os.path.dirname
basename = os.path.basename
realpath = os.path.realpath
join = os.path.join

# -----------------------------------------------------------------------------
# Problems parameters
# -----------------------------------------------------------------------------
# Defaults
PROB_DEFAULT = {
    'tEnd': 360,
    'nPhi': 256,
    'nTheta': 128
    }

# Script parameters
PROB_PARAMS = {
    ('-tEnd',):
        dict(help='end time of simulation (in hour)', type=int,
             default=PROB_DEFAULT['tEnd']),
    ('-nPhi',):
        dict(help='number of point for phi (longitude)', type=int,
             default=PROB_DEFAULT['nPhi']),
    ('-nTheta',):
        dict(help='number of point for theta (latitude)', type=int,
             default=PROB_DEFAULT['nTheta'])
    }

# Printing function
def probInfos(dt, method, tEnd, nPhi, nTheta):
    return f"""
Problem settings
-- tEnd : {tEnd}h
-- nPhi : {nPhi}
-- nTheta : {nTheta}
Solver settings
-- timeStepper : {method}
-- dt : {dt}s
""".strip()

# -----------------------------------------------------------------------------
# Import SDC parameters
# -----------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from core import DEFAULT as SDC_DEFAULT, PARAMS as SDC_PARAMS, sdcInfos

SDC_PTYPES = {k: type(v) for k, v in SDC_DEFAULT.items()}

# -----------------------------------------------------------------------------
# Utility macro variables
# -----------------------------------------------------------------------------
EXEC_PATH = join(dirname(dirname(realpath(__file__))), 'problems',
                'galewsky.py')

TIME_FMT = "%Y-%m-%d %H:%M:%S,%f"

LOG_INFOS = [
    'status', 'simTime (h)', 'nIter done', 'speed (sH/s)',
    'tComp (current)', 'tComp (left)']

CHECK_TIME = 1.5  # Waiting time before checking simulation started well


class SimHandler(object):

    def __init__(self, outDir, **params):
        self.path = os.path.abspath(outDir)
        self.infos = {'problem': {},
                      'solver': {}}
        if os.path.isdir(self.path):
            # Parameters are read from the simulation output directory
            read = self.readInfos()
            if not read:
                raise ValueError(
                    f'output directory {self.path} already exists '
                    'but does not contain any info file to read. '
                    'Please remove empty directory before.')
        else:
            # Parameters are given as keyword arguments
            if 'dt' not in params or 'method' not in params:
                raise ValueError(
                    'dt and method must be defined for unset simulations')
            self.infos['solver']['dt'] = params.pop('dt')
            self.infos['solver']['timeStepper'] = params.pop('method')
            for key, val in PROB_PARAMS.items():
                pID = key[0][1:]
                self.infos['problem'][pID] = params.pop(pID, val['default'])
            if self.infos['solver']['timeStepper'] == 'SDC':
                for key, val in SDC_PARAMS.items():
                    pID = key[0]
                    par = params.pop(pID, SDC_DEFAULT[pID])
                    # Convert to correct type
                    if SDC_PTYPES[pID] == bool:
                        par = True if par == 'True' else False
                    else:
                        par = SDC_PTYPES[pID](par)
                    self.infos['solver'][pID] = par
            for key, val in params.items():
                print(f'WARNING : {key}={val} parameter given but not used')
            # Write simulation information in output dir
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
            with open(self.infoFile, 'w') as f:
                pInfos = self.infos['problem']
                sInfos = self.infos['solver']
                f.write(probInfos(
                    sInfos['dt'], sInfos['timeStepper'],
                    pInfos['tEnd'], pInfos['nPhi'], pInfos['nTheta']))
                if sInfos['timeStepper'] == 'SDC':
                    f.write('\n'+sdcInfos(**sInfos))

        # Attribute for simulation run subprocess
        self.proc = None

    def __str__(self):
        return f"SimHandler ({self.name}, {self.status})"

    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        return basename(self.path)

    @property
    def infoFile(self):
        return join(self.path, '00_simInfos.txt')

    def readInfos(self):
        path = self.infoFile
        if path is None:
            return False

        with open(path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]

        pInfos = self.infos['problem']
        sInfos = self.infos['solver']
        i = 1
        while not lines[i].startswith('Solver'):
            key, val = lines[i][3:].split(' : ')
            pInfos[key] = val
            i += 1
        i += 1
        while i < len(lines):
            key, val = lines[i][3:].split(' : ')
            sInfos[key] = val
            i += 1

        pInfos['tEnd'] = int(float(pInfos['tEnd'][:-1]))
        for key in ['nPhi', 'nTheta']:
            pInfos[key] = int(pInfos[key])
        sInfos['dt'] = int(sInfos['dt'][:-1])
        for key, pType in SDC_PTYPES.items():
            if key in sInfos:
                sInfos[key] = pType(sInfos[key])

        return True

    @property
    def runCommand(self):
        dt = self.infos['solver']['dt']
        method = self.infos['solver']['timeStepper']
        cmd = f"python {EXEC_PATH} {dt} {method}"
        cmd += f" -logFile -o {self.path}"
        for args, dic in PROB_PARAMS.items():
            arg = args[0]
            pID = arg[1:]
            val = self.infos['problem'][pID]
            cmd += f" {arg} {val}"
        if method == 'SDC':
            for args, dic in SDC_PARAMS.items():
                pID = args[0]
                arg = args[-1]
                val = self.infos['solver'][pID]
                if SDC_PTYPES[pID] == bool:
                    cmd += f" {arg}" if val else ""
                else:
                    cmd += f" {arg} {val}"
        return cmd

    @property
    def pidFile(self):
        return join(self.path, '01_pidInfos.txt')

    @property
    def pidInfos(self):
        if self.proc is not None:
            return self.proc.pid, self.proc.nProc
        if os.path.isfile(self.pidFile):
            with open(self.pidFile, 'r') as f:
                infos = f.read().strip().split()
                pid = int(infos[0])
                nProc = int(infos[1])
            return pid, nProc

    @property
    def pid(self):
        try:
            return self.pidInfos[0]
        except Exception:
            return None

    def run(self, nProc=4):
        # Security check
        if self.isRunning:
            # Current process is running simulation
            print("WARNING : attempt to run a simulation already running "
                  f"({self.name}), skipping ...")
            return
        if self.status == 'completed':
            # Log file indicated completed simulation
            print("WARNING : attempt to run a simulation already completed "
                  f"({self.name}), skipping ...")
            return
        # Lauch run process
        if nProc > 1:
            cmd = f"mpirun -n {nProc} "
        else:
            cmd = ""
        cmd += self.runCommand
        print(f"Launching simulation for {self.name} with {nProc} proc(s)")
        print(f" -- {cmd}")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, shell=True)
        # Check if simulation correctly started
        tCheck = CHECK_TIME
        print(f" -- waiting {tCheck} sec. to check everything went well ...")
        time.sleep(tCheck)
        poll = self.proc.poll()
        if poll is not None:
            if poll == 0:
                print(f'WARNING : simulation for {self.name} finished '
                      f'in less that {tCheck} sec., within returning '
                      'any error code (that\'s weird ...)')
                self.proc = None
            else:
                errors = self.proc.stderr.read().decode()
                print(f'ERROR : simulation for {self.name} did not launched '
                      'correctly, see error message below:\n'
                      f'{errors}')
                raise SystemError(f'Run failed for {self.name}')

        # Save run process information (pid, number of mpi processes)
        self.proc.nProc = nProc
        with open(self.pidFile, 'w') as f:
            f.write(f'{self.proc.pid} {nProc}')
        print(f' -- simulation correctly started, with pid={self.proc.pid}')

    @property
    def nRunProc(self):
        if not self.isRunning:
            return 0
        if self.proc is None:
            infos = self.pidInfos
            try:
                return infos[1]
            except TypeError:
                raise SystemError(
                    'ERROR : {self.name} is running, but no informations '
                    'on the running process !!!')
        else:
            return self.proc.nProc

    @property
    def isRunning(self):
        pid = self.pid
        if pid is None:
            return False
        if self.proc is not None:
            # Process is running the simulation
            if self.proc.poll() is not None:
                # Already completed
                self.proc = None
                return 0
            # Still running
            return self.proc.nProc
        # Process is not running the simulation
        try:
            proc = psutil.Process(pid)
            if proc.status() == 'zombie':
                # Simulation process has not been correctly terminated
                proc.terminate()
                return False
            else:
                # Still runnint
                return True
        except Exception:
            # Simulation has been completed
            return False


    def kill(self):
        if self.isRunning:
            self.proc.terminate()

    def reset(self):
        if self.isRunning:
            raise SystemError('cannot reset a running simulation')
        if os.path.isdir(self.path):
            os.system(f'rm -rf {self.path}/01*')
            os.system(f'rm -rf {self.path}/logs*')
            os.system(f'rm -rf {self.path}/*h5')
            os.system(f'rm -rf {self.path}.log')

    @property
    def logFile(self):
        path = join(self.path, 'logs', 'simu_p0.log')
        if os.path.isfile(path):
            return path

    @property
    def logInfos(self):

        sTimeEnd = self.infos['problem']['tEnd']

        if self.logFile is None:
            infos = [
                'pending', f"0.0/{sTimeEnd}", "0 (0%)",
                "??", "??h/??m/??s", "??h/??m/??s"]
            return {key: val for key, val in zip(LOG_INFOS, infos)}

        with open(self.logFile, 'r') as f:
            lines = [l.strip() for l in f.readlines()]

        # Get rid of first initialization lines
        i = 0
        for i, l in enumerate(lines):
            if 'INFO :: Starting main loop' in l:
                break

        if i == len(lines)-1:
            infos = [
                'initializing', f"0.0/{sTimeEnd}", "0 (0%)",
                "??", "??h/??m/??s", "??h/??m/??s"]
            return {key: val for key, val in zip(LOG_INFOS, infos)}

        lines = lines[i:]

        # Get last iteration line
        for i, l in enumerate(lines):
            if "INFO :: Final iteration:" in l:
                i -= 1
                break

        status = 'completed'
        if self.isRunning:
            status = 'running'
        elif len(lines) == 0:
            status == 'error'
        elif lines[i] == lines[-1]:
            status = 'killed'
        elif "ERROR :: Exception raised," in lines[i+1]:
            status = 'stopped'

        # Get current state infos
        tBeg = datetime.strptime(lines[0][:23], TIME_FMT)
        tEnd = datetime.strptime(lines[i][:23], TIME_FMT)
        try:
            nIter = int(lines[i].split('Iteration=')[1].split(',')[0])
            sTime = float(lines[i].split('Time=')[1].split(',')[0])
            dt = float(lines[i].split('dt=')[1])
        except IndexError:
            infos = [
                'starting', f"0.0/{sTimeEnd}", "0 (0%)",
                "??", "??h/??m/??s", "??h/??m/??s"]
            return {key: val for key, val in zip(LOG_INFOS, infos)}
        if status == 'completed':
            while "Final iteration: " not in lines[i]:
                i += 1
            tmp = lines[i].split("Final iteration: ")
            tEnd = datetime.strptime(tmp[0][:23], TIME_FMT)
            nIter = int(tmp[-1])
            while "Final sim time: " not in lines[i]:
                i += 1
            sTime = float(lines[i].split("Final sim time: ")[-1])

        # Computation time and speed
        diff = tEnd-tBeg
        tComp = diff.total_seconds()
        hComp, sComp = int(tComp // 3600), tComp % 3600
        mComp, sComp = int(sComp // 60), sComp % 60
        compSpeed = dt*nIter/tComp

        # Computation time until simulation end
        sTimeEnd = self.infos['problem']['tEnd']
        sTimeLeft = 0 if status == 'completed' else sTimeEnd - sTime
        nIterEnd = int(round(sTimeEnd/dt))
        tCompLeft = sTimeLeft/compSpeed
        hLeft, sLeft = int(tCompLeft // 3600), tCompLeft % 3600
        mLeft, sLeft = int(sLeft // 60), sLeft % 60

        tilde = '' if status == 'completed' else '~'
        infos = [
            status,
            f"{sTime:3.1f}/{sTimeEnd}",
            f"{nIter} ({nIter/nIterEnd*100:1.1f}%)",
            f"{compSpeed:1.3f}",
            f"{hComp}h/{mComp}m/{sComp:1.2f}s",
            f"{tilde}{hLeft}h/{mLeft}m/{sLeft:1.1f}s"]
        return {key: val for key, val in zip(LOG_INFOS, infos)}

    @property
    def status(self):
        return self.logInfos['status']

    def printLogInfos(self):
        infos = self.logInfos
        df = pd.DataFrame(infos, index=[self.name], columns=infos)
        print(df.to_markdown())



class SimMonitor(object):

    def __init__(self, runDir='.'):
        prefix = realpath(runDir)
        lSimu = glob.glob(join(prefix, '*/00_simInfos.txt'))
        lSimu = [dirname(f) for f in lSimu]
        lSimu = [path for path in lSimu if os.path.isdir(path)]

        self.prefix = prefix
        self.handlers = [SimHandler(path) for path in lSimu]

    def getHandlers(self, status=['completed']):
        return [h for h in self.handlers if h.status in status]

    @property
    def simNames(self):
        return [h.name for h in self.handlers]

    def addSimu(self, outDir, **params):
        """
        Set a new simulation for the monitor's scheduler

        Parameters
        ----------
        outDir : str
            Name of the output directory for the simulation.
        **params : keyword arguments
            Any parameter defined in SDC_DEFAULT or PROB_DEFAULT.
        """
        if not basename(outDir) in self.simNames:
            self.handlers.append(SimHandler(outDir, **params))

    def printLogInfos(self):
        df = pd.DataFrame({}, columns=LOG_INFOS)
        for h in self.handlers:
            df.loc[h.name] = h.logInfos
        df.sort_index(inplace=True)
        print(df.to_markdown())

    @property
    def nRunProc(self):
        return sum(h.nRunProc for h in self.handlers)

    def runAll(self, nSimProc=4, nProcMax=20):
        for h in self.handlers:
            while self.nRunProc+nSimProc > nProcMax:
                time.sleep(10)
            h.run(nSimProc)

    def waitAll(self, status='completed'):
        for h in self.handlers:
            while h.status != status:
                time.sleep(10)

if __name__ == '__main__':
    monitor = SimMonitor('../problems')
