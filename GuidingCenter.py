# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:26:26 2016

@author: kaan
"""
import numpy as np
from scipy.integrate import odeint

import rapt.utils as ru
from rapt import c, params, Particle

class GuidingCenter:
    # TO DO: Not compatible with electric fields and time-varying magnetic fields.
    # The advance() method needs to have:
    #   (1) The ExB term
    #   (2) The speed as a separate variable that's updated in time.
    #       Update the kinetic energy at each step and get speed from it.
    def __init__(self, pos=None, vpar=None, speed=None, t0=None, mass=None, charge=None, field=None, eq='brizardchan'):
        self.pos = pos  # initial position array
        self.vp = vpar  # initial parallel speed
        self.v = speed    # speed of the particle
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  # magnetic field function, taking position array and returning field array
        self.eq = eq     # Equation to solve. 'northropteller' or 'brizardchan'
        if not (pos==None or vpar==None or speed==None or t0==None): # if initial state is given explicitly
            self.mu = ru.magnetic_moment(self.tcur, self.pos, self.vp, self.v, self.field, self.mass)
            self.trajectory = np.concatenate(([t0], pos, [vpar]))
            self.trajectory = np.reshape(self.trajectory, (1,5))
    
    def init(self, p):  # Initialization with another Particle or GuidingCenter instance
        if isinstance(p, GuidingCenter):
            self.tcur = p.trajectory[-1,0]            
            self.pos = p.trajectory[-1,1:4]
            self.vp = p.trajectory[-1,4]
            self.v = p.v
            self.mass = p.mass
            self.charge = p.charge
            self.field = p.field
            self.eq = p.eq
            self.mu = ru.magnetic_moment(self.tcur, self.pos, self.vp, self.v, self.field, self.mass)
            self.trajectory = np.concatenate(([self.tcur], self.pos, [self.vp]))
            self.trajectory = np.reshape(self.trajectory, (1,5))
        elif isinstance(p, Particle):
            self.mass = p.mass
            self.charge = p.charge
            self.field = p.field
            self.tcur = p.trajectory[-1,0]
            self.pos, self.vp, self.v = ru.guidingcenter(self.tcur, p.trajectory[-1,1:4], p.trajectory[-1,4:], self.field, self.mass, self.charge)
            self.mu = ru.magnetic_moment(self.tcur, self.pos, self.vp, self.v, self.field, self.mass)
            self.trajectory = np.concatenate(([self.tcur], self.pos, [self.vp]))
            self.trajectory = np.reshape(self.trajectory, (1,5))
        else:
            raise(ValueError, "Particle or GuidingCenter objects required.")
        
    def isadiabatic(self):
        """Returns True if the particle's motion satisfies the adiabaticity conditions
        at the present location."""
        # Spatial and temporal adiabaticity thresholds
        eps_sp = params['epss']
        eps_t = params['epst'] 

        # Conditions:
        # gyroradius / lengthscale < eps_sp
        # and
        # gyroperiod / timescale < eps_t
        if self.field.static:
            return self.cycrad() / self.field.lengthscale(self.trajectory[-1,:4]) < eps_sp
        else:
            return self.cycrad() / self.field.lengthscale(self.trajectory[-1,:4]) < eps_sp \
            and self.cycper() / self.field.timescale(self.trajectory[-1,:4]) < eps_t
    
    def advance(self, delta):
        if self.eq == "northropteller":
            self.NorthropTellerAdvance(delta)
        elif self.eq == "brizardchan":
            self.BrizardChanAdvance(delta)
    
    def NorthropTellerAdvance(self, delta):
        """Advance the GC position and parallel speed for time 'delta' starting at the current time, position, parallel speed."""
        dt = params["GCtimestep"]
        gamma = 1.0/np.sqrt(1 - (self.v/c)**2)
        
        def deriv(Y, t=0):
            Bvec = self.field.B(Y[:4])
            B = np.sqrt(np.dot(Bvec,Bvec))
            bdir = Bvec / B
            gB = self.field.B(Y[:4])
            return np.concatenate(
            (
            [1],
            gamma*self.mass/(2*self.charge*B**2) * (self.v**2 + Y[4]**2)*np.cross(bdir,gB) + Y[4]*bdir,
            [-self.mu/(self.mass*gamma**2) * np.dot(bdir, gB)]
            )
            )
        if delta <= dt:
            times = np.array([self.tcur, self.tcur+delta])
        else:
            times = np.arange(self.tcur, self.tcur+delta, dt)
        rtol, atol = params["solvertolerances"]
        traj = odeint(deriv, self.trajectory[-1,:], times, rtol=rtol, atol=atol)
        self.trajectory = np.concatenate((self.trajectory, traj[1:,:]))
        self.tcur = self.trajectory[-1,0]
    
    def BrizardChanAdvance(self, delta):
        dt = params["GCtimestep"]
        gamma = 1.0/np.sqrt(1 - (self.v/c)**2)

        def deriv(Y, t=0):
            B = self.field.B(Y[:4])
            Bmag = np.sqrt(np.dot(B,B))
            unitb = B / Bmag
            gB = self.field.gradB(Y[:4])
            cb = self.field.curlb(Y[:4])
            Bstar = B + gamma * self.mass * Y[4] * cb / self.charge
            Bstarpar = np.dot(B,Bstar) / Bmag
            retval = np.ones(5) 
            retval[1:4] = (Y[4] * Bstar  + self.mu * np.cross(unitb, gB) / (self.charge * gamma) ) / Bstarpar
            retval[4] = -self.mu * np.dot(Bstar, gB) / (self.mass * gamma*gamma * Bstarpar)
            return retval
        if delta <= dt:
            times = np.array([self.tcur, self.tcur+delta])
        else:
            times = np.arange(self.tcur, self.tcur+delta, dt)
        rtol, atol = params["solvertolerances"]
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
    def getvp(self):
        return self.trajectory[:,4]
    def getr(self):  # Distance from the origin
        return np.sqrt( self.getx()**2 + self.gety()**2 + self.getz()**2 )
    def gettheta(self): # Polar angle
        return np.arctan2( self.gety(), self.getx() )
    def getphi(self):  # Azimuthal angle
        return np.arccos( self.getz() / self.getr() )
    def getB(self):
        out = np.zeros(len(self.trajectory))
        for i, row in enumerate(self.trajectory):
            out[i] = self.field.magB(row[:4])
        return out
    
    def cycrad(self):
        """The current cyclotron radius."""
        t, r, vp = self.trajectory[-1, 0], self.trajectory[-1, 1:4], self.trajectory[-1, 4]
        return ru.cyclotron_radius2(t, r, vp, self.v, self.field, self.mass, self.charge)

    def cycper(self):
        """The current cyclotron period."""
        t, r = self.trajectory[-1, 0], self.trajectory[-1, 1:4]
        return ru.cyclotron_period2(t, r, self.v, self.field, self.mass, self.charge)

    def ke(self):
        # BUG: Drift speed term not added.
        mc2 = self.mass*c**2
        gammasq = 1/(1 - (self.v/c)**2)
        return np.sqrt(2*self.mu*self.getB()*mc2 + gammasq*self.mass*mc2*self.trajectory[:,4]**2 + mc2**2) - mc2
    
    def kenr(self):
        # BUG: Drift speed term not added.
        return self.mu * self.getB() + 0.5*self.mass*self.trajectory[:,4]**2
