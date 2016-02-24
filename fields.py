import numpy as np
from rapt import Re, B0,utils
from scipy.interpolate import RegularGridInterpolator

class _Field:
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
        self.isstatic = True # whether the field varies in time

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
    
    def dBdt(self, tpos):  # time derivative of the magnetic field magnitude.
        if self.static:
            return 0
        else:
            d = self.timederivstepsize
            B1 = self.magB(tpos - [d,0,0,0])
            B2 = self.magB(tpos + [d,0,0,0])
            return (B2-B1)/d/2
    
    def dbdt(self, tpos):   # time derivative of the magnetic field direction.
        if self.isstatic:
            return 0
        else:
            d = self.timederivstepsize
            b1 = self.unitb(tpos - [d,0,0,0])
            b2 = self.unitb(tpos + [d,0,0,0])
            return (b2-b1)/d/2

    def lengthscale(self, tpos):
        """Returns B/| grad |B| |, the length scale of the change of the magnetic field strength."""
        g = self.gradB(tpos)
        return self.magB(tpos) / np.sqrt(np.dot(g,g))
    
    def timescale(self, tpos):
        """Returns B / d|B|/dt, the time scale of the change of the magnetic field strength."""
        if self.isstatic:
            return None
        else:
            return self.magB(tpos) / self.dBdt(tpos)
    
class EarthDipole(_Field):
    coeff = -3*B0*Re**3
    def __init__(self):
        _Field.__init__(self)
        self.gradientstepsize = Re*1e-6
    def B(self,tpos):
        t,x,y,z = tpos
        r2 = x*x+y*y+z*z
        
        return self.coeff / pow(r2, 2.5) * np.array([x*z, y*z, (z*z-r2/3)])

class EarthDipoleC(_Field):
    import ctypes
    _dip = ctypes.CDLL("./rapt/libdip.so")
    _dip.dipole.argtypes = [ctypes.c_double,
                            ctypes.c_double,
                            ctypes.c_double,
                            ctypes.POINTER(ctypes.c_double)]
    _dip.dipole.restype = None
    def B(self,tpos):
        res = (self.ctypes.c_double * 3)()
        t,x,y,z = tpos
        self._dip.dipole(x/Re,y/Re,z/Re,res)
        return -B0*np.array(res)
 
class DoubleDipoleC(_Field):
    import ctypes
    _dip = ctypes.CDLL("./rapt/libdip.so")
    _dip.doubledipole.argtypes = [ctypes.c_double,
                                  ctypes.c_double,
                                  ctypes.c_double,
                                  ctypes.POINTER(ctypes.c_double)]
    _dip.doubledipole.restype = None
    def __init__(self, standoff=10*Re, imagestrength=1):
        _Field.__init__(self)
        self.gradientstepsize = Re/1000
        self.dd = 2*standoff  # distance between two dipoles
        self.k = imagestrength   # >=1. Relative strength of the image dipole.
    def B(self, tpos):
        res = (self.ctypes.c_double * 3)()
        t,x,y,z = tpos
        self._dip.doubledipole(x/Re,y/Re,z/Re,res)
        return -B0*np.array(res)

class DoubleDipole(_Field):
    def __init__(self, standoff=10*Re, imagestrength=1):
        _Field.__init__(self)
        self.gradientstepsize = Re/1000
        self.dd = 2*standoff  # distance between two dipoles
        self.k = imagestrength   # >=1. Relative strength of the image dipole.
    def B(self, tpos):
        t,x,y,z = tpos
        B1 =  np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)]) / pow(x*x+y*y+z*z, 5.0/2.0)
        x -= self.dd
        B2 = self.k * np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)]) / pow(x*x+y*y+z*z, 5.0/2.0)
        return (-B0*Re**3)*(B1+B2)

class UniformBz(_Field):
    def __init__(self, Bz=1):
        _Field.__init__(self)
        self.Bz = Bz
    def B(self,tpos):
        return np.array((0,0,self.Bz))

class UniformCrossedEB(UniformBz):
    def __init__(self, Ey=1, Bz=1):
        self.isstatic = False
        self.Ey = Ey
        self.Bz = Bz
    def E(self,tpos):
        return np.array((0,self.Ey,0))

class VarEarthDipole(_Field):
    # Earth dipole.
    # Strength sinusoidally oscillating in time around the nominal value.
    def __init__(self,amp=0.1,period=10):
        _Field.__init__(self)
        self.gradientstepsize = Re/1000
        self.isstatic = False
        self.amp = amp
        self.period = period
    def B(self,tpos):
        t,x,y,z = tpos
        return -B0*Re**3 * (1+self.amp*np.sin(2*np.pi*t/self.period)) * np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)]) / pow(x*x+y*y+z*z, 5.0/2.0)

class Parabolic(_Field):
    # A parabolic model field imitating the tail.
    def __init__(self, B0=10.0, Bn=1.0, d=0.2):
        _Field.__init__(self)
        self.B0 = B0
        self.Bn = Bn
        self.d = d
    def B(self, tpos):
        z = tpos[3]
        if abs(z)<=1.0:
            return [self.B0*z/self.d, 0, self.Bn]
        else:
            return [np.sign(z)*B0, 0, self.Bn]

class DipoleAndLinear(_Field):
    def __init__(self):
        _Field.__init__(self)
    def B(self, tpos):
        t,x,y,z = tpos
        return B0*x/(10*Re)**4 - B0*Re**3 / pow(x*x+y*y+z*z, 5.0/2.0) * np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)])

class Grid(_Field):
    def __init__(self):
        _Field.__init__(self)
        self.times = []  # list of time values for each stored grid
        self.grids = []  # list of interpolated field functions, same order as times.
        # Each element of grids: [Bx,By,Bz,Ex,Ey,Ez] (6 interpolated functions)

    def generate(self,f,tmin,tmax,xmin,xmax,ymin,ymax,zmin,zmax,nt,nx,ny,nz):
        # Generate grids using the specified field object.

        x = np.linspace(xmin,xmax,nx)
        y = np.linspace(ymin,ymax,ny)
        z = np.linspace(zmin,zmax,nz)
        self.times = np.linspace(tmin,tmax,nt)
        self.grids = []
        for t in self.times:
            B_x = np.zeros((nx,ny,nz),dtype='float')
            B_y = np.zeros((nx,ny,nz),dtype='float')
            B_z = np.zeros((nx,ny,nz),dtype='float')
            E_x = np.zeros((nx,ny,nz),dtype='float')
            E_y = np.zeros((nx,ny,nz),dtype='float')
            E_z = np.zeros((nx,ny,nz),dtype='float')
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        Bvec = f.B([t,x[i],y[j],z[k]])
                        B_x[i][j][k] = Bvec[0]
                        B_y[i][j][k] = Bvec[1]
                        B_z[i][j][k] = Bvec[2]
                        Evec = f.E([t,x[i],y[j],z[k]])
                        E_x[i][j][k] = Evec[0]
                        E_y[i][j][k] = Evec[1]
                        E_z[i][j][k] = Evec[2]
            self.grids.append( [RegularGridInterpolator((x,y,z),B_x),
                        RegularGridInterpolator((x,y,z),B_y),
                        RegularGridInterpolator((x,y,z),B_z),
                        RegularGridInterpolator((x,y,z),E_x),
                        RegularGridInterpolator((x,y,z),E_y),
                        RegularGridInterpolator((x,y,z),E_z)] )

    def readgrid(self, t, x, y, z, Bx, By, Bz, Ex, Ey, Ez):
        self.times.append(t)
        B = [RegularGridInterpolator((x,y,z),Bx),
             RegularGridInterpolator((x,y,z),By),
             RegularGridInterpolator((x,y,z),Bz)]
        E = [RegularGridInterpolator((x,y,z),Ex),
             RegularGridInterpolator((x,y,z),Ey),
             RegularGridInterpolator((x,y,z),Ez)]
        self.grids.append(B+E)

    def B(self, tpos):
        t,x,y,z = tpos
        # Warning: Works only with single time instant
        bg = self.grids[0]
        Bx = bg[0](tpos[1:])[0]
        By = bg[1](tpos[1:])[0]
        Bz = bg[2](tpos[1:])[0]
        return np.array([Bx,By,Bz])
        
    def E(self, tpos):
        # Warning: Works only with single time instant
        bg = self.grids[0]
        Ex = bg[3](tpos[1:])[0]
        Ey = bg[4](tpos[1:])[0]
        Ez = bg[5](tpos[1:])[0]
        return np.array([Ex,Ey,Ez])


class RCMF(Grid):
    coeff = -3*B0*Re**3
    zero3 = np.zeros(3)
    def __init__(self):
        super(RCMF,self).__init__()

    def readgrid(self, t, x, y, z, Bx, By, Bz):
        self.times.append(t)
        B = [RegularGridInterpolator((x,y,z),Bx),
             RegularGridInterpolator((x,y,z),By),
             RegularGridInterpolator((x,y,z),Bz)]

        self.grids.append(B)

    def B(self, tpos):
        t,x,y,z = tpos
        # Warning: Works only with single time instant
        bg = self.grids[0]
        
        if tpos[3]<0:
            tpos[3] *= -1
            Bx = -bg[0](tpos[1:])[0]
            By = -bg[1](tpos[1:])[0]
            Bz = bg[2](tpos[1:])[0]
        else:
            Bx = bg[0](tpos[1:])[0]
            By = bg[1](tpos[1:])[0]
            Bz = bg[2](tpos[1:])[0]
            
        # Add the dipole component
        r2 = x*x+y*y+z*z
        
        Bdip =  self.coeff / pow(r2, 2.5) * np.array([x*z, y*z, (z*z-r2/3)])
        return np.array([Bx,By,Bz])+Bdip
        
    def E(self, tpos):
        return self.zero3

class RCMF_tricubic(Grid):
    from tricubic import tricubic
    coeff = -3*B0*Re**3
    zero3 = np.zeros(3)
    def __init__(self):
        super(RCMF_tricubic,self).__init__()

    def readgrid(self, t, x, y, z, Bx, By, Bz):
        self.times.append(t)
        nx,ny,nz = Bx.shape
        Bxif = self.tricubic(list(Bx),[nx,ny,nz]).ip
        Byif = self.tricubic(list(By),[nx,ny,nz]).ip
        Bzif = self.tricubic(list(Bz),[nx,ny,nz]).ip
        B = [Bxif, Byif, Bzif]
        self.grids.append(B)

    def B(self, tpos):
        t,x,y,z = tpos
        # Warning: Works only with single time instant
        bg = self.grids[0]
        
        if tpos[3]<0:
            tpos[3] *= -1
            Bx = -bg[0](list(tpos[1:]))
            By = -bg[1](list(tpos[1:]))
            Bz = bg[2](list(tpos[1:]))
        else:
            Bx = bg[0](list(tpos[1:]))
            By = bg[1](list(tpos[1:]))
            Bz = bg[2](list(tpos[1:]))            
        # Add the dipole component
        r2 = x*x+y*y+z*z
        
        Bdip =  self.coeff / pow(r2, 2.5) * np.array([x*z, y*z, (z*z-r2/3)])
        return np.array([Bx,By,Bz])+Bdip
        
    def E(self, tpos):
        return self.zero3
    
class spdipole(_Field):
    import spacepy.time as spt
    import spacepy.coordinates as spc
    import spacepy.irbempy as ib
    def __init__(self):
        _Field.__init__(self)
    def B(self,tpos):
        t = self.spt.Ticktock(['2002-01-01T12:00:00'], 'ISO')
        pos = self.spc.Coords(tpos[1:]/Re, 'GSM', 'car')
        Bvec = self.ib.get_Bfield(t,pos,extMag='0',options=[0,0,0,0,5])
        
        return Bvec['Bvec'][0]*1e-9