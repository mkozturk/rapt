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
    def __init__(self, pos=None, vel=None, t0=None, mass=None, charge=None, field=None):
        self.pos = pos  # initial position array
        tpos = np.concatenate([[t0], pos])        
        self.vp = np.dot(vel, field.B(tpos)) / field.magB(tpos)  # initial parallel speed
        self.v = np.sqrt(np.dot(vel,vel))    # speed of the particle
        self.t0 = t0 # initial time
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field
        if not (pos==None or vel==None or t0==None): # if initial state is given explicitly
            self.mu = ru.magnetic_moment(self.tcur, self.pos, self.vp, self.v, self.field, self.mass)
            self.trajectory = np.concatenate(([t0], pos, [self.vp]))
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
            
    def setpa(self, pa):
        """Resets the velocity vector so that the pitch angle is pa degrees. Reinitializes the object."""
        field = self.field
        tpos = self.trajectory[-1,0:4]
        v = self.trajectory[-1,4:] # velocity
        
        # Construct a vector that has the same length as v,
        # lies on the plane spanned by v and B,
        # and makes an angle of pa with B.
        
        # the unit vector in the direction of the field.
        b = field.unitb(tpos)
        # component of v perpendicular to b:
        vperp = v - np.dot(v,b)*b
        # unit vector of vperp
        p = vperp / np.sqrt(np.dot(vperp,vperp))
        # speed
        s = np.sqrt(np.dot(v,v))
        # The new velocity vector
        w = s*np.sin(pa*np.pi/180)*b + s*np.cos(pa*np.pi/180)*p
        # Reinitialize with the new velocity
        self.__init__(self.pos, w, self.t0, self.mass, self.charge, self.field)

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
        if self.field.isstatic:
            if params["GCeq"] == "northropteller":
                self.NorthropTellerAdvance(delta)
            elif params["GCeq"] == "brizardchan":
                self.BrizardChanAdvance(delta)
            else:
                raise Exception("Guiding-center equation "+params["GCeq"]+" not implemented.")
        else:
            self.TaoChanBrizardAdvance(delta)
    
    def TaoChanBrizardAdvance(self, delta):
        # To be used when electric fields are nonzero and/or fields vary in time.
        # Reference: Tao, X., A. A. Chan, and A. J. Brizard (2007),
        # Hamiltonian theory of adiabatic motion of relativistic charged particles,
        # Phys. Plasmas, 14, 092107, doi:10.1063/1.2773702
        dt = params["GCtimestep"]
        m = self.mass
        def deriv(Y, t=0):
            B = self.field.B(Y[:4])
            Bmag = np.sqrt(np.dot(B,B))
            unitb = B / Bmag
            gamma = np.sqrt(1 + self.mu*Bmag/(m*c*c) + (Y[4]/(m*c)**2))
            cb = self.field.curlb(Y[:4])
            Bstar = B + Y[4] * cb / self.charge
            Bstarpar = np.dot(Bstar,unitb)
            E = self.field.E(Y[:4])
            dbdt = self.field.dbdt(Y[:4])
            gB = self.field.gradB(Y[:4])
            Estar = E - (Y[4]*dbdt - self.mu * gB / gamma)/self.charge
            
            retval = np.ones(5) 
            retval[1:4] = (Y[4] * Bstar / (gamma*m) + np.cross(Estar,unitb) ) / Bstarpar
            retval[4] = self.charge*np.dot(Estar,Bstar) / Bstarpar
            return retval
        if delta <= dt:
            times = np.array([self.tcur, self.tcur+delta])
        else:
            times = np.arange(self.tcur, self.tcur+delta, dt)
        rtol, atol = params["solvertolerances"]
        traj = odeint(deriv, self.trajectory[-1,:], times, rtol=rtol, atol=atol)
        self.trajectory = np.concatenate((self.trajectory, traj[1:,:]))
        self.tcur = self.trajectory[-1,0]
    
    def NorthropTellerAdvance(self, delta):
        # The "classic" equations of motion for the guiding center.
        # Superseded by other equations such as the Brizard-Chan.
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
        # Highly accurate GC equations of motion.
        # Reference: A. J. Brizard and A. A. Chan, 
        # Nonlinear relativistic gyrokinetic Vlasov-Maxwell equations, 
        # Phys. Plasmas 6, 4548 (1999)
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
