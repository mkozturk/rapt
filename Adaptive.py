# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:27:46 2016

@author: kaan
"""
import numpy as np
from rapt import Particle, GuidingCenter

class Adaptive:
    # The Adaptive class automatically decides whether to use Particle or GuidingCenter
    # by looking at local and instantaneous conditions and at the particle state.

    # This class is different from Particle or GuidingCenter in that it does not
    # store the trajectory data as a single 2D array. Instead, it stores a
    # list of Particle and GuidingCenter objects, which themselves store the
    # trajectory info. However, the interface is identical.

    def __init__(self, pos=None, vel=None, t0=0, mass=None, charge=None, field=None):
        self.pos = pos  # initial position array
        self.vel = vel  # initial velocity array
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  #  the field object
        p = Particle(pos,vel,t0,mass,charge,field)
        if p.isadiabatic():
            g = GuidingCenter()
            g.init(p)
            self.trajlist = [g]
        else:
            self.trajlist = [p]

    def advance(self, delta):
        t = 0
        gctimestep = 2   # follow the GC for that many seconds before checking for adiabaticity
        partimestep = 10 # follow the particle for that many periods before checking for adiabaticity
        while t < delta:
            # Check if the adiabaticity condition holds in the last state.
            # If adiabatic, proceed with GuidingCenter type. 
            # Otherwise proceed with Particle type.
            current = self.trajlist[-1] # a Particle or a GuidingCenter
            if current.isadiabatic():
                # If conditions are adiabatic, follow a GC.
                # If not already a GC, create a GuidingCenter and initialize with the latest particle.
                # If already a GC, just advance the solution.
                if not isinstance(current, GuidingCenter):
                    g = GuidingCenter()
                    g.init(current)
                    self.trajlist.append(g)
                    current = self.trajlist[-1]
                    print("Switched to guiding center at time", current.tcur)
                
                if t + gctimestep < delta:
                    current.advance(gctimestep)
                else: # the next time step would exceed final time "delta"
                    current.advance(delta-t)
                
            else: # conditions not adiabatic. Follow particle path instead of GC.
                # If not already a particle, create a Particle and initialize with the latest guiding center position.
                if not isinstance(current, Particle):
                    p = Particle()
                    p.init(current)
                    self.trajlist.append(p)
                    current = self.trajlist[-1]
                    print("Switched to particle at time", current.tcur)

                dt = partimestep*current.cycper()
                if t + dt < delta:
                    current.advance(dt)
                else: # the next time step would exceed final time "delta"
                    current.advance(delta-t)
                    break

            t = current.trajectory[-1,0]

    def gett(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.gett() ) )
        return res
    def getx(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getx()) )
        return res
    def gety(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.gety()) )
        return res
    def getz(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getz()) )
        return res
    def getr(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getr()) )
        return res
    def getphi(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getphi()) )
        return res
    def gettheta(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.gettheta()) )
        return res
    def getv(self):
        res = np.array([])
        for p in self.trajlist:
            if isinstance(p, Particle):
                res = np.concatenate(res, p.getv())
            if isinstance(p, GuidingCenter):
                v = np.full( (len(p.trajectory),1), p.v )
                res = np.concatenate( (res, v) )
        return res
    
    def ke(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.ke()) )
        return res
    
    def kenr(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.kenr()) )
        return res