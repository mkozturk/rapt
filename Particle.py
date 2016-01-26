# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:25:23 2016

@author: kaan
"""

import numpy as np
from scipy.integrate import odeint

import rapt.utils as ru
from rapt import c, params

class Particle:
    def __init__(self, pos=None, vel=None, t0=0, mass=None, charge=None, field=None):
        self.pos = pos  # initial position array
        self.vel = vel  # initial velocity array
        self.t0 = t0    # initial time
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  #  the field object
        if not (pos == None or vel == None): # if initial state is given explicitly
            self.trajectory = np.concatenate(([self.tcur], self.pos, self.vel))
            self.trajectory = np.reshape(self.trajectory, (1,7))
            
    def init(self, p, gyrophase=0):  # Initialization with another Particle or GuidingCenter instance
        from rapt import GuidingCenter # Import here to avoid circular imports.        
        if isinstance(p, Particle):
            self.__init__(pos=p.trajectory[-1,1:4], 
                          vel=p.trajectory[-1,4:],
                          t0=p.trajectory[-1,0], 
                          mass=p.mass, 
                          charge=p.charge,
                          field=p.field)
#            # This can be written more briefly by calling __init__
#            self.mass = p.mass
#            self.charge = p.charge
#            self.field = p.field
#            self.tcur = p.trajectory[-1,0]            
#            self.pos = p.trajectory[-1,1:4]
#            self.vel = p.trajectory[-1,4:]
#            self.trajectory = np.concatenate(([self.tcur], self.pos, self.vel))
#            self.trajectory = np.reshape(self.trajectory, (1,7))
        elif isinstance(p, GuidingCenter):
            B = p.field.magB(p.trajectory[-1,:4])            
            gammasq = 1 + 2*p.mu*B/(p.mass*c*c) + (p.trajectory[-1,4]/p.mass/c)**2
            v = c * np.sqrt(1-1/gammasq)
            vpar = p.trajectory[-1,4]/p.mass/np.sqrt(gammasq)
            self.pos, self.vel = ru.GCtoFP(self.tcur, 
                                           p.trajectory[-1,1:4], 
                                           vpar, 
                                           v, 
                                           p.field, 
                                           p.mass, 
                                           p.charge, 
                                           gyrophase)
#             This can be written more briefly by calling __init__
#            self.mass = p.mass
#            self.charge = p.charge
#            self.field = p.field
#            self.tcur = p.trajectory[-1,0]
#            self.pos, self.vel = ru.GCtoFP(self.tcur, p.trajectory[-1,1:4], p.trajectory[-1,4], p.v, self.field, self.mass, self.charge, gyrophase)
#            self.trajectory = np.concatenate(([self.tcur], self.pos, self.vel))
#            self.trajectory = np.reshape(self.trajectory, (1,7))
        else:
            raise(ValueError, "Particle or GuidingCenter objects required.")

    def setKE(self,KE, unit="ev"):
        """Scales the velocity vector with the speed corresponding to the given kinetic energy.
        Reinitializes the object."""
        s = ru.speedfromKE(KE, self.mass, unit)
        v = self.trajectory[-1,4:]
        v = v*(s/np.sqrt(np.dot(v,v)))
        self.__init__(self.pos, v, self.t0, self.mass, self.charge, self.field)

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
        spar = np.dot(v,b)   # parallel speed
        
        if spar < 1e-12:  # if velocity is parallel to b
            p = ru.getperp(b)  # get a vector perp. to b, direction otherwise 'random'
        else:
            vperp = v - spar*b # component of v perpendicular to b
            p = vperp / np.sqrt(np.dot(vperp,vperp)) # unit vector of vperp
        # speed
        s = np.sqrt(np.dot(v,v))
        # The new velocity vector
        w = s*np.sin(pa*np.pi/180)*b + s*np.cos(pa*np.pi/180)*p
        # Reinitialize with the new velocity
        self.__init__(self.pos, w, self.t0, self.mass, self.charge, self.field)
        
    def advance(self, delta):
        """Advance the particle position and velocity for time 'delta' starting at the current time, position, and velocity."""
        t = self.trajectory[-1,0]        
        pos = self.trajectory[-1,1:4]
        vel = self.trajectory[-1,4:]
        # set resolution of cyclotron motion
        res = ru.cyclotron_period(t, pos, vel, self.field, self.mass, self.charge) / params['cyclotronresolution']
     
        def deriv(Y, t=0):
            gamma = 1.0/np.sqrt(1 - np.dot(Y[4:], Y[4:]) / c**2)
            # later optimization: put gamma outside if the E-field is always zero.
            out = np.zeros(7) 
            out[0] = 1        # dt/dt = 1
            out[1:4] = Y[4:]  # d(pos)/dt = vel
            out[4:] = (self.field.E(Y[:4]) + 
            self.charge * np.cross(Y[4:], self.field.B(Y[:4]))
            ) / (self.mass*gamma) # d(vel)/dt = E(t, pos) / m + (q/m)vel x B(t, pos)
            return out
        
        times = np.arange(self.tcur, self.tcur+delta, res)
        rtol, atol = params["solvertolerances"]
        traj = odeint(deriv, self.trajectory[-1,:], times, rtol=rtol, atol=atol)
        self.trajectory = np.concatenate((self.trajectory, traj[1:,:]))
        self.tcur = self.trajectory[-1,0]
    
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
        
    def gett(self):
        return self.trajectory[:,0]
    def getx(self):
        return self.trajectory[:,1]
    def gety(self):
        return self.trajectory[:,2]
    def getz(self):
        return self.trajectory[:,3]
    def getvx(self):
        return self.trajectory[:,4]
    def getvy(self):
        return self.trajectory[:,5]
    def getvz(self):
        return self.trajectory[:,6]
    def getv(self):  # total speed of the particle
        return np.sqrt( self.getvx()**2 + self.getvy()**2 + self.getvz()**2 )
    def getr(self):  # Distance from the origin
        return np.sqrt( self.getx()**2 + self.gety()**2 + self.getz()**2 )
    def gettheta(self): # Polar angle
        return np.arctan2( self.gety(), self.getx() )
    def getphi(self):  # Azimuthal angle
        return np.arccos( self.getz() / self.getr() )
    def ke(self):      # Kinetic energy
        gamma = 1/np.sqrt(1 - self.getv()**2/c**2)
        return (gamma-1)*self.mass*c**2
    def kenr(self):    # Nonrelativistic kinetic energy
        return 0.5*self.mass*self.getv()**2
    def guidingcenter(self):
        out = []
        for row in self.trajectory:
            t,r,v = row[0], row[1:4], row[4:]
            rgc, vp, spd = ru.guidingcenter(t, r, v, self.field, self.mass, self.charge)
            out.append(list(rgc)+[vp,spd])
        return np.array(out)
    def mu(self):
        out = []
        for row in self.trajectory:
            t, r,v = row[0], row[1:4], row[4:]
            rgc, vp, spd = ru.guidingcenter(t, r, v, self.field, self.mass, self.charge)
            out.append(ru.magnetic_moment(t, rgc, vp, spd, self.field, self.mass))
        return np.array(out)
    def cycrad(self):
        """The current cyclotron radius."""
        t, r, v = self.trajectory[-1, 0], self.trajectory[-1, 1:4], self.trajectory[-1, 4:]
        return ru.cyclotron_radius(t, r, v, self.field, self.mass, self.charge)
    def cycper(self):
        """The current cyclotron period."""
        t, r, v = self.trajectory[-1, 0], self.trajectory[-1, 1:4], self.trajectory[-1, 4:]
        return ru.cyclotron_period(t, r, v, self.field, self.mass, self.charge)