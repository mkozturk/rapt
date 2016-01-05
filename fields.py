import numpy as np
from rapt import Re, B0
class Field:
    # A general field class.
    # Will be subclassed; not for direct use.

    # Matrix to calculate the curl with central differences
    _M1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0],
                [0, 0,-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0, 0],
                [0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                ])
    
    def __init__(self):
        self.gradientstepsize = 1e-6  # step size to evaluate spatial derivatives with central differences
        self.timederivstepsize = 1e-3 # step size to evaluate time derivatives with central differences
        self.static = True # whether the field varies in time

    def B(self, tpos):
        # tpos : 4-element array of time, x, y, z
        return np.zeros(3)
    
    def E(self, tpos):
        # tpos : 4-element array of time, x, y, z
        return np.zeros(3)

    def unitb(self, tpos):
        Bvec = self.B(tpos)
        return Bvec / np.sqrt(np.dot(Bvec, Bvec))
        
    def magB(self,tpos):
        Bvec = self.B(tpos)
        return np.sqrt(np.dot(Bvec, Bvec))
        
    def gradB(self,tpos):
            d=self.gradientstepsize

            return np.array([
            ( self.magB(tpos + (0,d,0,0)) - self.magB(tpos - (0,d,0,0)) ) / (2*d),
            ( self.magB(tpos + (0,0,d,0)) - self.magB(tpos - (0,0,d,0)) ) / (2*d),
            ( self.magB(tpos + (0,0,0,d)) - self.magB(tpos - (0,0,0,d)) ) / (2*d)
            ])

    def curvature(self, tpos):
        Bvec = self.B(tpos)
        B = np.sqrt(np.dot(Bvec, Bvec))
        gB = self.gradB(tpos)
        gBperp = gB - (np.dot(gB,B)/B**2) * Bvec
        return np.sqrt(np.dot(gBperp, gBperp))/B
        
    def curlb(self,tpos):
        d=self.gradientstepsize
        beta = np.concatenate((
            self.unitb(tpos + (0,d,0,0)),
            self.unitb(tpos - (0,d,0,0)),
            self.unitb(tpos + (0,0,d,0)),
            self.unitb(tpos - (0,0,d,0)),
            self.unitb(tpos + (0,0,0,d)),
            self.unitb(tpos - (0,0,0,d))
            ))
        return np.dot(self._M1, beta) / (2*d)
    
    def dBdt(self, tpos):
        if self.static:
            return 0
        else:
            d = self.timederivstepsize
            B1 = self.magB(tpos - [d,0,0,0])
            B2 = self.magB(tpos + [d,0,0,0])
            return (B2-B1)/d/2
    
    def lengthscale(self, tpos):
        """Returns B/| grad |B| |, the length scale of the change of the magnetic field strength."""
        g = self.gradB(tpos)
        return self.magB(tpos) / np.sqrt(np.dot(g,g))
    
    def timescale(self, tpos):
        """Returns B / d|B|/dt, the time scale of the change of the magnetic field strength."""
        return self.magB(tpos) / self.dBdt(tpos)
    
class EarthDipole(Field):
    coeff = -3*B0*Re**3
    def __init__(self):
        Field.__init__(self)
        self.gradientstepsize = Re*1e-6
    def B(self,tpos):
        t,x,y,z = tpos
        r2 = x*x+y*y+z*z
        
        return self.coeff / pow(r2, 2.5) * np.array([x*z, y*z, (z*z-r2/3)])

class DoubleDipole(Field):
    def __init__(self, standoff=10*Re, imagestrength=1):
        Field.__init__(self)
        self.gradientstepsize = Re/1000
        self.dd = 2*standoff  # distance between two dipoles
        self.k = imagestrength   # >=1. Relative strength of the image dipole.
    def B(self, tpos):
        t,x,y,z = tpos
        B1 =  np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)]) / pow(x*x+y*y+z*z, 5.0/2.0)
        x -= self.dd
        B2 = self.k * np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)]) / pow(x*x+y*y+z*z, 5.0/2.0)
        return (-B0*Re**3)*(B1+B2)
        
class UniformBz(Field):
    def __init__(self, Bz=1):
        Field.__init__(self)
        self.Bz = Bz
    def B(self,tpos):
        return np.array((0,0,self.Bz))

class UniformCrossedEB(UniformBz):
    def __init__(self, Ey=1, Bz=1):
        self.Ey = Ey
        self.Bz = Bz
    def E(self,tpos):
        return np.array((0,self.Ey,0))

class VarEarthDipole(Field):
    # Earth dipole.
    # Strength sinusoidally oscillating in time around the nominal value.
    def __init__(self,amp=0.1,period=10):
        Field.__init__(self)
        self.gradientstepsize = Re/1000
        self.amp = amp
        self.period = period
    def B(self,tpos):
        t,x,y,z = tpos
        return -B0*Re**3 * (1+self.amp*np.sin(2*np.pi*t/self.period)) * np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)]) / pow(x*x+y*y+z*z, 5.0/2.0)

class Parabolic(Field):
    # A parabolic model field imitating the tail.
    def __init__(self, B0=10.0, Bn=1.0, d=0.2):
        Field.__init__(self)
        self.B0 = B0
        self.Bn = Bn
        self.d = d
    def B(self, tpos):
        z = tpos[3]
        if abs(z)<=1.0:
            return [self.B0*z/self.d, 0, self.Bn]
        else:
            return [np.sign(z)*B0, 0, self.Bn]

class DipoleAndLinear(Field):
    def __init__(self):
        Field.__init__(self)
    def B(self, tpos):
        t,x,y,z = tpos
        return B0*x/(10*Re)**4 - B0*Re**3 / pow(x*x+y*y+z*z, 5.0/2.0) * np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)])
