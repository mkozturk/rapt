# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:21:10 2015

@author: kaan
"""
import numpy as np
from scipy.integrate import odeint

import rapt.utils as ru
from rapt import c, params
from rapt.flutils import halfbouncepath, bounceperiod, gradI
    
class BounceCenter:
    def __init__(self, pos=None, vel=None, t0=None, mass=None, charge=None, field=None):
        self.pos = pos  # initial position array
        self.v = np.sqrt(np.dot(vel,vel))    # speed of the particle
        self.tcur = t0     # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  # the field object
        if not field.isstatic:
            raise RuntimeError("BounceCenter does not work with nonstatic fields or electric fields.")
        tpos = np.concatenate([[t0],pos])
        # The initial pitch angle:
        self.pa = np.arccos( np.dot(vel, field.B(tpos)) / (self.v * field.magB(tpos)) )
        
        # the first invariant value (constant)
        self.mu = ru.magnetic_moment(t0, pos, self.v*np.cos(self.pa), self.v, field, mass)
        assert self.mu>0
        if not (pos==None or vel==None or t0==None): # if initial state is given explicitly
            self.trajectory = np.concatenate(([t0], pos))
            self.trajectory = np.reshape(self.trajectory, (1,4))

    def setpa(self, pa):
        """Orients the velocity vector such that the angle between velocity and field vectors is pa degrees."""
        self.pa = pa * np.pi / 180  # convert to radians
        self.mu = ru.magnetic_moment(self.trajectory[0], self.trajectory[1:4], self.v*np.cos(self.pa), self.v, self.field, self.mass)
        
    def advance(self, delta):
        """Advance the position for time 'delta' starting at the current time, position, and velocity."""
        v = self.v
        gamma = 1.0/np.sqrt(1 - (v / c)**2)
        Bm = self.mass*gamma**2*v**2/(2*self.mu)
        bp = bounceperiod(self.trajectory[-1,:4], self.field, Bm, v)
        dt = params['BCtimestep'] * bp
        
        def deriv(Y, t=0):
            out = np.zeros(4)
            out[0] = 1        # dt/dt = 1
            Bvec = self.field.B(Y[:4])
            magBsq = np.dot(Bvec, Bvec)
            Sb = halfbouncepath(Y[:4], self.field, Bm)
            gI = gradI(Y[:4], self.field, Bm)
            out[1:4] = gamma*self.mass*v*v/(self.charge*Sb*magBsq) * np.cross(gI,Bvec)  # d(pos)/dt = vel
            return out
            
        rtol, atol = params["solvertolerances"]
        times = np.arange(self.tcur, self.tcur+delta, dt)
        traj = odeint(deriv, self.trajectory[-1,:], times, rtol=rtol, atol=atol)
        self.trajectory = np.concatenate((self.trajectory, traj[1:,:]))
        self.tcur = self.trajectory[-1,0]
        
    def gett(self):
        return self.trajectory[:,0]
    def getx(self):
        return self.trajectory[:,1]
    def gety(self):
        return self.trajectory[:,2]
    def getz(self):
        return self.trajectory[:,3]