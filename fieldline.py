# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:54:01 2015

@author: kaan
"""
import numpy as np
from scipy.integrate import odeint
from rapt.rkf import rkf
from rapt import params


class Fieldline:
    def __init__(self, tpos, field, ds=0, stopcond=None, Bmin=None, Bmax=None):
        self.time = tpos[0]        
        self.initpt = np.concatenate(([0],tpos[1:]))
        self.curve = np.zeros((1,4))  # each row: s, x, y, z
        self.curve[0,:] = self.initpt
        self.field = field
        self.Bmin = Bmin
        self.Bmax = Bmax
        self.solver = params['flsolver']   # 'rkf' or 'odeint'
        if ds == 0:
            self.ds = self._estimate_step()
        else:
            self.ds = ds
        if stopcond == None:
            self.stopcond = lambda a,b,c: False
        else:
            self.stopcond = stopcond    

    def _estimate_step(self):
        n = params['fieldlineresolution']
        # get the length scale at current position
        k = self.field.curvature(np.concatenate( ([self.time],self.curve[-1,1:]) ))
        return 1/k/n

    def trace(self):        
        # Solve the equation d[s,x,y,z]/ds = [1, bx, by, bz]
        def derivfw(Y,t=0): # solve forward
            tpos = np.concatenate([[self.time], Y[1:]])
            out = np.zeros(4)
            out[0] = 1 # ds/ds = 1
            out[1:] = self.field.unitb(tpos)  # dr/ds = b
            return out
            
        def derivbw(Y,t=0): # solve backward
            tpos = np.concatenate([[self.time], Y[1:]])
            out = np.zeros(4)
            out[0] = -1 # ds/ds = -1
            out[1:] = -self.field.unitb(tpos)  # dr/ds = -b
            return out
        
        # The trace always has at least 3 points. Initial point, and at least one on each side.
        forward = np.zeros((1,4))
        forward[0] = self.initpt
        backward = np.zeros((1,4))
        backward[0] = self.initpt
        
        while not self.stopcond(self.time, forward[-1,1:], self.field):
            # Follow the field line for length ds, starting from the last point.             
            
            if self.solver=='odeint':            
                r = odeint(derivfw, forward[-1,:], [0, self.ds], rtol=1e-6, atol=1e-6)
            if self.solver == 'rkf':
                s, r = rkf(derivfw, 0, self.ds,  forward[-1,:], tol = 1e-10, hmax=self.ds, hmin=1e-6)            
            
            # Append the new data to existing curve data.
            # Exclude the first point in r as it is already the last point in curve.
            forward = np.concatenate( (forward, r[1:,:]) )
            # Get the current time and position vector
            tpos = np.concatenate([[self.time], forward[-1,1:]])
            # Stop iteration if we passed maximum or minimum field points.
            B = self.field.magB(tpos)
            if self.Bmin != None and B < self.Bmin:
                break
            if self.Bmax != None and B > self.Bmax:
                break
        
        # Repeat the same in the antiparallel direction
        while not self.stopcond(self.time, backward[-1,1:], self.field):
            # Follow the field line for length -ds, starting from the last point.             
            if self.solver=='odeint':
                r = odeint(derivbw, backward[-1,:], [0, self.ds], rtol=1e-6, atol=1e-6)
            if self.solver=='rkf':
                s, r = rkf(derivbw, 0, self.ds,  backward[-1,:], tol = 1e-10, hmax=self.ds, hmin=1e-6)            

            # Append the new data to existing curve data.
            # Exclude the first point in r as it is already the lastpoint in curve.
            backward = np.concatenate( (backward, r[1:,:]) )
            # Get the current time and position vector
            tpos = np.concatenate([[self.time], backward[-1,1:]])
            # Stop iteration if we passed maximum or minimum field points.
            B = self.field.magB(tpos)
            if self.Bmin != None and B < self.Bmin:
                break
            if self.Bmax != None and B > self.Bmax:
                break

        # Join the forward and backward parts. Reverse the backward trace
        # and leave out its first element to avoid repetition.
        self.curve = np.concatenate( (backward[:0:-1,:], forward) )
        
        # N.B.: The curve can extend for several points beyond the mirror point
        # (or, generally, the limit point) with large step sizes, because the
        # solver will return the intermediate points within the last ds.
        # When determining the mirror field, search accordingly.
        
    def reset(self):
        self.curve = np.zeros((1,4))
    
    def gets(self):
        return self.curve[:,0]
    def getx(self):
        return self.curve[:,1]
    def gety(self):
        return self.curve[:,2]
    def getz(self):
        return self.curve[:,3]
    def getB(self):
        out = np.zeros(len(self.curve))
        for i, row in enumerate(self.curve):
            out[i] = self.field.magB(np.concatenate( ([self.time],row[1:]) ))
        return out
    def getr(self):  # Distance from the origin
        return np.sqrt( self.getx()**2 + self.gety()**2 + self.getz()**2 )
    def gettheta(self): # Polar angle
        return np.arctan2( self.gety(), self.getx() )
    def getphi(self):  # Azimuthal angle
        return np.arccos( self.getz() / self.getr() )
