# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:26:26 2016

@author: kaan
"""
import numpy as np
from scipy.integrate import odeint

import rapt.utils as ru
from rapt import c, params

class GuidingCenter:
    def __init__(self, pos=None, v=None, pa=None, ppar=None, t0=0, mass=None, charge=None, field=None):
        # pos: initial position (array of 3)
        # v: initial speed (number)
        # pa: initial pitch angle
        # ppar = initial parallel momentum
        #     (N.B. Either pa or ppar should be provided. If both given, pa is used.)
        # t0: initial time
        # mass: particle mass in kg
        # charge: particle charge in C
        # field: The field object
    
        self.pos = pos  # initial position array
        self.v = v
        if pa != None:
            gamma = 1/np.sqrt(1-(v/c)**2)
            ppar = gamma * mass * v * np.cos(pa*np.pi/180)
        
        self.t0 = t0 # initial time
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field
        if not (pos==None or v==None or ppar==None): # if initial state is given explicitly
            self.mu = ru.magnetic_moment(self.tcur, self.pos, ppar/gamma/mass,
                                         self.v, self.field, self.mass)
            self.trajectory = np.concatenate(([t0], pos, ppar))
            self.trajectory = np.reshape(self.trajectory, (1,5))
    
    def init(self, p):  # Initialization with another Particle or GuidingCenter instance
        from rapt import Particle # Import here to avoid circular imports. 
        if isinstance(p, GuidingCenter):
            B = p.field.magB(p.trajectory[-1,:4])            
            gammasq = 1 + 2*p.mu*B/(p.mass*c*c) + (p.trajectory[-1,4]/p.mass/c)**2
            v = c * np.sqrt(1-1/gammasq)
            self.__init__(pos = p.trajectory[-1,1:4],
                          v = v,
                          ppar = p.trajectory[-1,4],
                          t0 = p.trajectory[-1,0],
                          mass = p.mass, 
                          charge = p.charge,
                          field = p.field)
                
        elif isinstance(p, Particle):
            pos, vp, v = ru.guidingcenter(p.trajectory[-1,0],
                                          p.trajectory[-1,1:4],
                                          p.trajectory[-1,4:], 
                                          p.field, 
                                          p.mass,
                                          p.charge)
            gamma = 1 / np.sqrt(1-(v/c)**2)
            self.__init__(pos = pos,
                          v = v,
                          ppar = p.mass*gamma*vp,
                          t0 = p.trajectory[-1,0],
                          mass = p.mass, 
                          charge = p.charge,
                          field = p.field)
        else:
            raise(ValueError, "Particle or GuidingCenter objects required.")

#   Not used anymore:            
#    def setpa(self, pa):
#        """Resets the velocity vector so that the pitch angle is pa degrees. Reinitializes the object."""
#        field = self.field
#        tpos = self.trajectory[-1,0:4]
#        v = self.trajectory[-1,4:] # velocity
#        
#        # Construct a vector that has the same length as v,
#        # lies on the plane spanned by v and B,
#        # and makes an angle of pa with B.
#        
#        # the unit vector in the direction of the field.
#        b = field.unitb(tpos)
#        # component of v perpendicular to b:
#        vperp = v - np.dot(v,b)*b
#        # unit vector of vperp
#        p = vperp / np.sqrt(np.dot(vperp,vperp))
#        # speed
#        s = np.sqrt(np.dot(v,v))
#        # The new velocity vector
#        w = s*np.sin(pa*np.pi/180)*b + s*np.cos(pa*np.pi/180)*p
#        # Reinitialize with the new velocity
#        self.__init__(self.pos, w, self.t0, self.mass, self.charge, self.field)

    def isadiabatic(self):
        """Returns True if the particle's motion satisfies the adiabaticity conditions
        at the present location."""
        # IMPROVEMENT: Functions cycrad and cycper repeat the calcuation of v.
        # Evaluate here directly, reusing v values.
        
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
        self.TaoChanBrizardAdvance(delta)
#        if self.field.isstatic:
#            if params["GCeq"] == "northropteller":
#                self.NorthropTellerAdvance(delta)
#            elif params["GCeq"] == "brizardchan":
#                self.BrizardChanAdvance(delta)
#            else:
#                raise Exception("Guiding-center equation "+params["GCeq"]+" not implemented.")
#        else:
#            self.TaoChanBrizardAdvance(delta)
    
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
            gamma = np.sqrt(1 + 2*self.mu*Bmag/(m*c*c) + (Y[4]/(m*c)**2))
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
        return self.trajectory[:,4]  # parallel momentum
#        if self.field.isstatic:
#            return self.trajectory[:,4]
#        else:
#            # The nonstatic case stores the parallel momentum. Convert to parallel speed.
#            B = self.getB()
#            gamma = np.sqrt( 1 + 2*self.mu*B/(self.mass*c*c) + (self.trajectory[:,4]/self.mass/c)**2)
#            return self.trajectory[:,4]/gamma/self.mass
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
        if self.field.isstatic:
            return ru.cyclotron_radius2(t, r, vp, self.v, self.field, self.mass, self.charge)
        else:            
            # The advancer stores parallel momentum at last element. Convert to parallel speed.
            Bmag = self.field.magB(self.trajectory[-1,:4])
            gamma = np.sqrt(1 + 2*self.mu*Bmag/(self.mass*c*c) + (vp/(self.mass*c)**2))
            vp = vp / self.mass / gamma # parallel speed
            v = c * np.sqrt(1-1/gamma**2)
            return ru.cyclotron_radius2(t, r, vp, v, self.field, self.mass, self.charge)

    def cycper(self):
        """The current cyclotron period."""
        t, r = self.trajectory[-1, 0], self.trajectory[-1, 1:4]
        if self.field.isstatic:
            return ru.cyclotron_period2(t, r, self.v, self.field, self.mass, self.charge)
        else:
            Bmag = self.field.magB(self.trajectory[-1,:4])
            gamma = np.sqrt(1 + 2*self.mu*Bmag/(self.mass*c*c) + (vp/(self.mass*c)**2))
            v = c * np.sqrt(1-1/gamma**2)
            return ru.cyclotron_period2(t, r, v, self.field, self.mass, self.charge)

    def ke(self):
        # BUG: Drift speed term not added.
        mc2 = self.mass*c**2
        gammasq = 1/(1 - (self.v/c)**2)
        return np.sqrt(2*self.mu*self.getB()*mc2 + gammasq*self.mass*mc2*self.trajectory[:,4]**2 + mc2**2) - mc2
    
    def kenr(self):
        # BUG: Drift speed term not added.
        return self.mu * self.getB() + 0.5*self.mass*self.trajectory[:,4]**2
