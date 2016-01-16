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
    def __init__(self, pos=None, v=None, t0=0, pa=None, mass=None, charge=None, field=None):
        self.pos = pos  # initial position array
        self.v = v    # speed of the particle
        self.t0 = t0
        self.tcur = t0     # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  # the field object
        self.isequatorial = False  # Temporary. Later replace with an algorithm.
        if not field.isstatic:
            raise RuntimeError("BounceCenter does not work with nonstatic fields or electric fields.")
        # The initial pitch angle:
        self.pa = pa
        #tpos = np.concatenate([[t0],pos])
        #self.pa = np.arccos( np.dot(vel, field.B(tpos)) / (self.v * field.magB(tpos)) )
        
        # the first invariant value (constant)
        self.mu = ru.magnetic_moment(t0, pos, self.v*np.cos(self.pa), self.v, field, mass)
        assert self.mu>0
        if not (pos==None or v==None or pa==None): # if initial state is given explicitly
            self.trajectory = np.concatenate(([t0], pos))
            self.trajectory = np.reshape(self.trajectory, (1,4))

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
        # unit vector of vperp:
        p = vperp / np.sqrt(np.dot(vperp,vperp))
        # speed:
        s = np.sqrt(np.dot(v,v))
        # The new velocity vector:
        w = s*np.sin(pa*np.pi/180)*b + s*np.cos(pa*np.pi/180)*p
        # Reinitialize with the new velocity:
        self.__init(self.pos, w, self.t0, self.mass, self.charge, self.field)
    
    def _v_grad(self, tpos):
        """The gradient-drift velocity"""
        gamma = 1.0/np.sqrt(1 - (self.v / c)**2)
        Bvec = self.field.B(tpos)
        magBsq = np.dot(Bvec, Bvec)
        gB = self.field.gradB(tpos)
        return self.mu*np.cross(Bvec,gB)/(self.charge*gamma*magBsq)
    
    def _v_inert(self,tpos):
        """Evaluate the inertial drift velocity term."""
        vg = self._v_grad(tpos)  # gradient drift velocity vector
        # In the absence of parallel speed, electric fields and time dependent fields,
        # the inertial drift reduces to dvg/dt = (vg . del) vg
        
        # Evaluate the directional derivative of vg in its own direction.
        vgdir = np.zeros(4)
        vgmag = np.sqrt(np.dot(vg,vg))        
        vgdir[0] = tpos[0]
        vgdir[1:] = vg / vgmag
        d = self.field.gradientstepsize
        v1 = self._v_grad(tpos - d*vgdir)
        v2 = self._v_grad(tpos + d*vgdir)
        Bvec = self.field.B(tpos)
        magBsq = np.dot(Bvec, Bvec)
        return self.mass * vgmag / (self.charge*magBsq*2*d) * np.cross(Bvec,(v2-v1))
    
    def advance(self, delta):
        """Advance the position for time 'delta' starting at the current time, position, and velocity."""
        v = self.v
        gamma = 1.0/np.sqrt(1 - (v / c)**2)
        Bm = self.mass*gamma**2*v**2/(2*self.mu)
        bp = bounceperiod(self.trajectory[-1,:4], self.field, Bm, v)
        dt = params['BCtimestep'] * bp
        
        def deriv(Y, t=0):
            if self.isequatorial:
                out = np.zeros(4)
                out[0] = 1        # dt/dt = 1
                out[1:4] = self._v_grad(Y[:4])+self._v_inert(Y[:4])
                return out
            else:
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