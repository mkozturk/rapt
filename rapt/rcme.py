# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:00:31 2016

@author: kaan
"""

import re
import numpy as np
from rapt import Re
from rapt.fields import Grid

class RCME(Grid):
    B0 = 30574*1e-9 # the mean value of the magnetic field at the magnetic equator on the Earth's surface (Tesla)
    _coeff = -3*B0*Re**3
    def __init__(self,filelist):
        Grid.__init__(self,filelist)
    def parsefile(self,fname):
        f = open(fname)
        f.readline()
        f.readline()
        thirdline = f.readline()
        m = re.search(r'FRICTION-IT=(\d+).*I=(\d+).*J=(\d+).*K=(\d+)',thirdline)
        time = float(m.group(1))*60
        nx = int(m.group(2))
        ny = int(m.group(3))
        nz = int(m.group(4))
        
        x = np.zeros(nx)
        y = np.zeros(ny)
        z = np.zeros(nz)
        Bx = np.zeros((nx,ny,nz))
        By = np.zeros((nx,ny,nz))
        Bz = np.zeros((nx,ny,nz))
        Ex = np.zeros((nx,ny,nz))
        Ey = np.zeros((nx,ny,nz))
        Ez = np.zeros((nx,ny,nz))
        
        # Read each line and fill up arrays using indices I,J,K.
        for line in f:
            L = [float(a) for a in line.split()]
            i,j,k,xi,yj,zk,inside,bx,by,bz,ex,ey,ez = L
            # Subtract 1 from the index values to make them 0-base.
            i = int(i-1)
            j = int(j-1)
            k = int(k-1)
            x[i] = xi
            y[j] = yj
            z[k] = zk
            Bx[i,j,k] = bx
            By[i,j,k] = by
            Bz[i,j,k] = bz
            Ex[i,j,k] = ex
            Ey[i,j,k] = ey
            Ez[i,j,k] = ez
        f.close()
        
        # The first and last element in position arrays should be discarded.
        x = x[1:-1]
        y = y[1:-1]
        z = z[1:-1]
        Bx = Bx[1:-1, 1:-1, 1:-1]
        By = By[1:-1, 1:-1, 1:-1]
        Bz = Bz[1:-1, 1:-1, 1:-1]
        Ex = Ex[1:-1, 1:-1, 1:-1]
        Ey = Ey[1:-1, 1:-1, 1:-1]
        Ez = Ez[1:-1, 1:-1, 1:-1]
        
        # To reduce interpolation errors, 
        # remove the dipole component, output only the residual.        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        R2 = X**2 + Y**2 + Z**2
        R5inv = R2 ** (-5/2)
        rr = 1
        coeff = (-3*self.B0*rr**3)*R5inv
        Bdipx = coeff * X*Z
        Bdipy = coeff * Y*Z
        Bdipz = coeff * (Z**2 - R2/3)
        return dict({'time':time, 'B0': self.B0,
            'x': x*Re, 'y': y*Re, 'z': z*Re, 
            'Bx': Bx - Bdipx, 'By': By - Bdipy, 'Bz': Bz - Bdipz,
            "Ex": Ex*1e-3, "Ey": Ey*1e-3, "Ez": Ez*1e-3})
    
    def B(self, tpos):
        t,x,y,z = tpos[:]
        Bnorth = self.Bgrid([t,x,y,abs(z)])
        if z<0:
            Bg = Bnorth * [-1,-1,1]
        else:
            Bg = Bnorth
        # Add the dipole component
        r2 = x*x+y*y+z*z
        Bdip =  self._coeff / pow(r2, 2.5) * np.array([x*z, y*z, (z*z-r2/3)])
        return Bg+Bdip
    
    def E(self,tpos):
        t,x,y,z = tpos[:]
        Enorth = self.Egrid([t,x,y,abs(z)])
        if z<0:
            return Enorth * [1,1,-1]
        else:
            return Enorth
