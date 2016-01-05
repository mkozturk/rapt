# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:21:10 2015

@author: kaan
"""
import numpy as np
from scipy.integrate import odeint

import rapt.utils as ru
from rapt import c, m_pr, e, params
from rapt.flutils import halfbouncepath, bounceperiod, gradI
from rapt.rkf import rkf

def bounce_av_vel_p(tpos, field, pa, v):
    """Bounce-averaged velocity of a proton.
    tpos: (time,x,y,z)
    field: The field object
    pa: Pitch angle in degrees
    v: Particle speed"""
    Bvec = field.B(tpos)
    magBsq = np.dot(Bvec, Bvec)
    gamma = 1.0/np.sqrt(1 - (v/c)**2)
    Bm = ru.Bm(tpos, pa, field)
    bp = bounceperiod(tpos, field, Bm, v)
    gI = gradI(tpos, field, Bm)
    return 2*gamma*m_pr*v/(e*bp*magBsq) * np.cross(gI,Bvec)
    
class BounceCenter:
    def __init__(self, pos=None, speed=None, t0=None, pa=None, mass=None, charge=None, field=None):
        self.pos = pos  # initial position array
        self.v0 = speed    # initial speed of the particle
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  # the field object
        if not field.isstatic:
            raise RuntimeError("BounceCenter does not work with nonstatic fields or electric fields.")
        self.pa = pa/180*np.pi  # The initial pitch angle
#        self.vd = bounce_av_vel_p(np.concatenate([[t0],pos]), field, pa, speed)
#        self.vd = np.sqrt(np.dot(self.vd, self.vd))
        self.fcalls = 0  # for debugging
        self.vectors = []  # for debugging. Holds the vectors returned by deriv() at each call.
        self.mu = ru.magnetic_moment(t0, pos, speed*np.cos(self.pa), speed, field, mass) # the first invariant value (constant)
        assert self.mu>0
        if not (pos==None or speed==None or t0==None): # if initial state is given explicitly
            self.trajectory = np.concatenate(([t0], pos))
            self.trajectory = np.reshape(self.trajectory, (1,4))

    def advance(self, delta):
        """Advance the position for time 'delta' starting at the current time, position, and velocity."""
 
        v = self.v0
        gamma = 1.0/np.sqrt(1 - (v / c)**2)
        Bm = self.mass*gamma**2*v**2/(2*self.mu)
        bp = bounceperiod(self.trajectory[-1,:4], self.field, Bm, v)
        dt = params['BCtimestep'] * bp
        
        def deriv(Y, t=0):
            self.fcalls += 1
            out = np.zeros(4)
            out[0] = 1        # dt/dt = 1
            Bvec = self.field.B(Y[:4])
            magBsq = np.dot(Bvec, Bvec)
            Sb = halfbouncepath(Y[:4], self.field, Bm)
            gI = gradI(Y[:4], self.field, Bm)
            out[1:4] = gamma*self.mass*v*v/(self.charge*Sb*magBsq) * np.cross(gI,Bvec)  # d(pos)/dt = vel
            self.vectors.append(np.concatenate((Y,out)))
            return out
            
        rtol, atol = params["solvertolerances"]
        times = np.arange(self.tcur, self.tcur+delta, dt)
        traj,info = odeint(deriv, self.trajectory[-1,:], times, rtol=rtol, atol=atol, full_output=True)
        self.odeintinfo = info
        self.trajectory = np.concatenate((self.trajectory, traj[1:,:]))
        self.tcur = self.trajectory[-1,0]
#        while self.tcur < tfinal:
#            print("CURRENT TIME:",self.tcur)
#            t, traj = rkf(deriv, self.tcur, self.tcur+dt, self.trajectory[-1,:], tol=atol, hmax = dt, hmin = minstep)        
#            self.trajectory = np.concatenate((self.trajectory, traj[1:,:]))
#            self.tcur = self.trajectory[-1,0]
        
    def gett(self):
        return self.trajectory[:,0]
    def getx(self):
        return self.trajectory[:,1]
    def gety(self):
        return self.trajectory[:,2]
    def getz(self):
        return self.trajectory[:,3]