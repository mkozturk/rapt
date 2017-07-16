import numpy as np
from rapt import Re, B0
from scipy.interpolate import RegularGridInterpolator

class _Field:
    """
    The superclass for fields. Not used directly, but subclassed. All field-
    related data and methods are defined in field objects.
    
    Attributes
    ----------
    gradientstepsize : float
        Step size to evaluate spatial derivatives with central differences.
    timederivstepsize : float
        Step size to evaluate time derivatives with central differences.
    static : bool
        True if the electric field is zero and the magnetic field is static,
        i.e., the fields do not change the speed of the particle.
    
    Notes
    -----
    The electric and magnetic fields are accessed with the`E` and `B` methods, 
    respectively. When subclassing, these need to be overridden. Other methods 
    defined here are usually extended by subclasses.
    
    All methods take a 4-element array consisting of time and coordinates
    (t,x,y,z) as parameter.
    
    All coordinates are Cartesian. SI units are used throughout.
    """
    
    # Matrix to calculate the curl with central differences
    _M1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0],
                [0, 0,-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0, 0],
                [0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                ])
    
    def __init__(self):
        self.gradientstepsize = 1e-6  # step size to evaluate spatial derivatives with central differences
        self.timederivstepsize = 1e-3 # step size to evaluate time derivatives with central differences
        self.static = True # True if dB/dt=0 or E=0, False otherwise. Essentially, True if the particle's speed stays constant (static magnetic field), and False otherwise.

    def B(self, tpos):
        """
        Return the magnetic field vector.
        
        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element array (Bx, By, Bz)
        """
        return np.zeros(3)
    
    def E(self, tpos):
        """
        Return the electric field vector.
        
        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element array (Ex, Ey, Ez)
        """
        # tpos : 4-element array of time, x, y, z
        return np.zeros(3)

    def unitb(self, tpos):
        """
        Return the direction of the magnetic field.

        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element unit vector B / |B|.
        """
        Bvec = self.B(tpos)
        return Bvec / np.sqrt(np.dot(Bvec, Bvec))
        
    def magB(self,tpos):
        """
        Return the magnitude of the magnetic field.

        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        float
            The magnetic field strength |B|.
        """
        
        Bvec = self.B(tpos)
        return np.sqrt(np.dot(Bvec, Bvec))
        
    def gradB(self,tpos):
        """
        Return the gradient of the magnetic field strength.

        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element vector :math: `\nabla |B|`
        """
        d=self.gradientstepsize

        return np.array([
        ( self.magB(tpos + (0,d,0,0)) - self.magB(tpos - (0,d,0,0)) ) / (2*d),
        ( self.magB(tpos + (0,0,d,0)) - self.magB(tpos - (0,0,d,0)) ) / (2*d),
        ( self.magB(tpos + (0,0,0,d)) - self.magB(tpos - (0,0,0,d)) ) / (2*d)
        ])

    def jacobianB(self,tpos):
        """
        Return the Jacobian matrix of the magnetic field.

        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-by-3 array with element (i,j) equal to dB_i / dx_j
        """

        d=self.gradientstepsize
        result = np.zeros((3,3))
        result[:,0] = (self.B(tpos + (0,d,0,0)) - self.B(tpos - (0,d,0,0)) ) / (2*d)
        result[:,1] = (self.B(tpos + (0,0,d,0)) - self.B(tpos - (0,0,d,0)) ) / (2*d)
        result[:,2] = (self.B(tpos + (0,0,0,d)) - self.B(tpos - (0,0,0,d)) ) / (2*d)
        return result

    def curvature(self, tpos):
        """
        Return the magnetic field line curvature.

        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        float
            The local field line curvature :math: `|\nabla_\perp B|/|B|`
        """

        Bvec = self.B(tpos)
        B = np.sqrt(np.dot(Bvec, Bvec))
        gB = self.gradB(tpos)
        gBperp = gB - (np.dot(gB,B)/B**2) * Bvec
        return np.sqrt(np.dot(gBperp, gBperp))/B
        
    def curlb(self,tpos):
        """
        Return the curl of the magnetic field direction.

        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element vector :math: `\nabla\times b`
        """

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
        """
        Return the time derivative of the magnetic field magntitude.

        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        float
            The time derivative d|B|/dt
        """

        if self.static:
            return 0
        else:
            d = self.timederivstepsize
            B1 = self.magB(tpos - [d,0,0,0])
            B2 = self.magB(tpos + [d,0,0,0])
            return (B2-B1)/d/2
    
    def dbdt(self, tpos):
        """
        Return the time derivative of the magnetic field direction.

        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element vector db/dt
        """
        if self.static:
            return 0
        else:
            d = self.timederivstepsize
            b1 = self.unitb(tpos - [d,0,0,0])
            b2 = self.unitb(tpos + [d,0,0,0])
            return (b2-b1)/d/2

    def lengthscale(self, tpos):
        """
        Return the length scale of the change of the magnetic field strength.

        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        float
            The length scale, |B| / max(Jacobian(B))
        """
        return self.magB(tpos) / np.max(abs(self.jacobianB(tpos)))
    
    def timescale(self, tpos):
        """
        Return the time scale of the change of the magnetic field strength.

        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        float
            The time scale, |B| / d|B|/dt.
        """
        if self.static:
            return None
        else:
            return self.magB(tpos) / abs(self.dBdt(tpos))
    
class EarthDipole(_Field):
    """
    The class representing the Earth's static dipole with zero tilt angle.
    
    Subclasses `_Field`. Overrides ``gradientstepsize`` and ``B()``.
    
    Parameters
    ----------
    B0 : float, optional
        The equatorial field strength at 1 Earth radius.
    
    """
    def __init__(self,B0=B0):
        """
        Initialize superclass and override the `gradientstepsize` attribute.
        """        
        _Field.__init__(self)
        self.gradientstepsize = Re*1e-6
        self._coeff = -3*B0*Re**3
    def B(self,tpos):
        """
        Return the magnetic field vector of the Earth's dipole.
        
        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element array (Bx, By, Bz)
        """
        t,x,y,z = tpos
        r2 = x*x+y*y+z*z
        return self._coeff / pow(r2, 2.5) * np.array([x*z, y*z, (z*z-r2/3)])

class DoubleDipole(_Field):
    """
    Field of two Earth dipoles with parallel magnetic moments.
    
    The dipole at x=y=0 represents Earth, and the dipole at x = distance is an
    "image dipole", whose field compresses the dipole at origin, simulating
    the dayside compression of the magnetosphere.
    
    Parameters
    ----------
    B0 : float, optional
        Dipole field strength at the equator (1 Re).
    distance : float, optional
        The distance between the two dipoles. Default 20 Re.
    imagestrength : float, optional
        The relative strength of the image dipole. Must be >=1. Default 1.
    """
    def __init__(self, B0=B0, distance=20*Re, imagestrength=1):
        _Field.__init__(self)
        self.gradientstepsize = Re/1000
        self._dd = distance  # distance between two dipoles
        assert imagestrength >= 1
        self._k = imagestrength   # >=1. Relative strength of the image dipole
        self._coeff = -B0*Re**3 
        
    def B(self, tpos):
        """
        Return the magnetic field vector of the double-dipole model.
        
        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element array (Bx, By, Bz)
        """
        t,x,y,z = tpos
        B1 =  np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)]) / pow(x*x+y*y+z*z, 5.0/2.0)
        x -= self._dd
        B2 = self._k * np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)]) / pow(x*x+y*y+z*z, 5.0/2.0)
        return self._coeff*(B1+B2)

class UniformBz(_Field):
    """
    Uniform static magnetic field in the z-direction, B = (0,0,Bz).
    
    Parameters
    ----------
    Bz : float, optional
        The constant field strength value in the z-direction. Default 1 T.
    """
    def __init__(self, Bz=1):
        _Field.__init__(self)
        self.Bz = Bz
    def B(self,tpos):
        """
        Return the uniform magnetic field vector.
        
        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element array (0, 0, Bz)
        """
        return np.array((0,0,self.Bz))

class UniformCrossedEB(UniformBz):
    """
    Perpendicular uniform static electric and magnetic fields.
    
    E = (0,Ey,0), B = (0,0,Bz)

    Extends `UniformBz`. Sets `static` to ``False``.

    Parameters
    ----------
    Ey : float, optional
        The constant electric field value in the y-direction (V/m). Default 1.
    Bz : float, optional
        The constant electric field value in th z-direction (T). Default 1.
    """
    # Uniform electric field in y-direction and uniform magnetic field in z-direction.
    def __init__(self, Ey=1, Bz=1):
        UniformBz.__init__(self)
        self.static = False
        self.Ey = Ey
        self.Bz = Bz
    def E(self,tpos):
        """
        Return the uniform electric field vector.
        
        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element array (0, Ey, 0)
        """
        return np.array((0,self.Ey,0))

class VarEarthDipole(_Field):
    """
    Time-varying Earth dipole.
    
    The magnetic moment oscillates sinusodially around the nominal value. 
    Illustrates time-dependent field setup.
    
    The induced electric field is ignored.
    
    Extends `_Field`. Sets ``static`` to False.
        
    Parameters
    ----------
    amp : float, optional
        The relative amplitude of the oscillations. Default 0.1. Unitless.
    period : float, optional
        The period of oscillations, in seconds. Default 10.
    """
    # Variable Earth dipole, as an example of time-dependent field.
    # Strength sinusoidally oscillating in time around the nominal value.
    def __init__(self,amp=0.1,period=10):
        _Field.__init__(self)
        self.gradientstepsize = Re/1000
        self.static = False
        self._amp = amp
        self._period = period
    def B(self,tpos):
        """
        Return the variable Earth dipole magnetic field vector.
        
        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element array (Bx, By, Bz) at time t.
        """
        t,x,y,z = tpos
        return -B0*Re**3 * (1+self._amp*np.sin(2*np.pi*t/self._period)) * np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)]) / pow(x*x+y*y+z*z, 5.0/2.0)

class Parabolic(_Field):
    """
    The parabolic magnetic field model imitating the current sheet in the 
    magnetotail.
    
    Parameters
    ----------
    B0 : float, optional
        The scale of the x-component of the field. Default 10.
    Bn : float, optional
        The z-component of the field. Default 1.
    d : float, optional
        The length scale, default 0.2. The field x-component increases by B0
        when we move by d in z-direction.
    
    Notes
    -----
    This model has the form :math: `B = (B0 z/d, 0, Bn)` if :math: `|z|<1` and
    :math: `B = (B0/d, 0, Bn)` otherwise. The field lines have a parabolic shape.
    Particles exhibit Speiser orbits, cucumber orbits and serpentine orbits.
    
    The parabolic model is well suited to testing the `Adaptive` mode because of
    the localized nonadiabaticity near z=0.
    
    References
    ----------
    
    """
    # A parabolic model field imitating the tail.
    def __init__(self, B0=10.0, Bn=1.0, d=0.2):
        _Field.__init__(self)
        self.B0 = B0
        self.Bn = Bn
        self.d = d
    def B(self, tpos):
        z = tpos[3]
        if abs(z)<=1.0:
            return np.array([self.B0*z/self.d, 0, self.Bn])
        else:
            return np.array([np.sign(z)*B0, 0, self.Bn])

class Grid(_Field):
    """
    A superclass for using fields sampled on a Cartesian grid.

    Extends the `_Field` class.
    
    Not for direct use; should be subclassed. The derived class should override 
    the `parsefile`, `E`, and `B` methods.
    
    Parameters
    ----------
    filelist : list of str
        A list of file names storing the field grid data, one for each time
        instant. The list must be ordered with respect to time.
    
    Raises
    ------
    ValueError
        If the requested time or coordinates are out of bounds.
    
    Notes
    -----
    
    Each file in the given list contains the electric and magnetic field data on
    discrete grid points. The details must be handled by the `parsefile` method.
    When a new model is implemented, users must override this method using
    the details and storage format of the data file they use.
   
    The `parsefile` method must return a dictionary with the following keys:
    
    * "time" : The time of the data, float.
    * "x" : 1D array of grid x-coordinates
    * "y" : 1D array of grid y-coordinates
    * "z" : 1D array of grid z-coordinates
    * "Bx" : 3D array of Bx values (similarly "By", "Bz")
    * "Ex" : 3D array of Ex values (similarly "Ey", "Ez")
    
    In a given `Grid` instance, the grid point coordinates must be the same for
    all files in the list. However, uniform spacing is not required.

    The interpolated field vectors are accessed with `Egrid` and `Bgrid` methods. The `E` and `B` methods can be overridden for further tweaking; such as adding a dipole component, or handling missing regions using the field symmetry.
    
    The class creates a 4-D linear interpolation for each of the six field components (Ex, Ey, Ez, Bx, By, Bz). Usually each data file is big, and MHD models are evaluated over a long period of time, resulting in many big data files. Loading the entire data set at once could be impossible for users with only several GB of memory. So the `Grid` object loads only the first three files when initialized, and then updates the interpolation as the tracer moves. There is always at most three time points in the interpolator.
    
    .. warning:: Once the field interpolator is updated, it forgets about earlier times. So after a tracer has advanced sufficiently, if we initialize another tracer, we will get a `ValueError` because the time for the new tracer is out of bounds.
    
    If the list contains only two files, a linear interpolation is done between two time points. Updates are not applicable.
    
    If the input list contains a single file, the field is considered independent of time. The interpolation is only 3-dimensional. The `_time_indep` attribute is set to ``True``. The methods `Egrid` and `Bgrid` adjust their behavior accordingly.
    
    """
    
    def __init__(self, filelist):
        """
        Grid constructor.
        
        Parameters
        ----------
        filelist: list of str
            The list of files where grid data is stored, one for each time point, in order of time. Length at least one.
        
        Parses the first three files and sets up the interpolator.
        """
        assert len(filelist)>0
        _Field.__init__(self)
        self.gradientstepsize = 1e-3*Re
        self.files = filelist[:]
        self._time_indep = False
        if len(self.files) >= 3:
            # parse the first three files and interpolate
            g0 = self.parsefile(self.files[0])
            self.g1 = self.parsefile(self.files[1])  # save for later use
            self.g2 = self.parsefile(self.files[2])  # save for later use
            self._set_interpolator(g0, self.g1, self.g2)
            del self.files[:3]
        elif len(self.files) == 2:
            # parse the first two files and interpolate
            g0 = self.parsefile(self.files[0])
            g1 = self.parsefile(self.files[1])
            self._set_interpolator(g0,g1)
            del self.files[:2]
        elif len(self.files)==1:
            # Parse the file and set up time-independent fields.
            self.time_indep = True  # used in B()
            g0 = self.parsefile(self.files[0])
            self._set_interpolator(g0)
            
    def parsefile(self, filename):
        """
        Parse one data file that stores the field data at one time point.
        
        Parameters
        ----------
        filename : str
            The name of the file storing the grid data.

        Notes
        -----
        The code of this method depends on the details of how the data is stored. When `Grid' is subclassed, users should override this method as appropriate.
        
        The method should return a dictionary with at least the following keys:

            * "time" : The time of the data, float.
            * "x" : 1D array of grid x-coordinates
            * "y" : 1D array of grid y-coordinates
            * "z" : 1D array of grid z-coordinates
            * "Bx", "By", "Bz" : 3D arrays of Bx, By, Bz values
            * "Ex", "Ey", "Ez" : 3D arrays of Ex, Ey, Ez values
        
        All values should be in SI units.
        """
        g = dict()
        return g

    def _set_interpolator(self, *glist):
        """
        Set up the interpolators for field components, given parsed data.
        Takes 1,2 or 3 data dictionaries generated by `parsefile`.
        """
        # Sets the interpolators for field components.
        assert 1<=len(glist)<=3
        g0 = glist[0]
        self.t0 = g0["time"]
        tlist = [self.t0]
        xg, yg, zg = g0["x"], g0["y"], g0["z"]
        nx, ny, nz = g0["Bx"].shape
        if len(glist) == 1: # One data file, time-independent, interpolation on 3D.
            Bx = g0["Bx"][:,:,:]        
            By = g0["By"][:,:,:]
            Bz = g0["Bz"][:,:,:]
            Ex = g0["Ex"][:,:,:]  
            Ey = g0["Ey"][:,:,:]
            Ez = g0["Ez"][:,:,:]
            # The following are called with three arguments only: Bxt_interp(x,y,z)
            self.Bxt_interp = RegularGridInterpolator( (xg,yg,zg), Bx)
            self.Byt_interp = RegularGridInterpolator( (xg,yg,zg), By)
            self.Bzt_interp = RegularGridInterpolator( (xg,yg,zg), Bz)
            self.Ext_interp = RegularGridInterpolator( (xg,yg,zg), Ex)
            self.Eyt_interp = RegularGridInterpolator( (xg,yg,zg), Ey)
            self.Ezt_interp = RegularGridInterpolator( (xg,yg,zg), Ez)

        else: # Two or three data files given. Time-dependent, 4D interpolation.
            Bxt = np.zeros(( len(glist), nx,ny,nz))
            Byt = np.zeros(( len(glist), nx,ny,nz))
            Bzt = np.zeros(( len(glist), nx,ny,nz))
            Ext = np.zeros(( len(glist), nx,ny,nz))
            Eyt = np.zeros(( len(glist), nx,ny,nz))
            Ezt = np.zeros(( len(glist), nx,ny,nz))
            Bxt[0,:,:,:] = g0["Bx"][:,:,:]        
            Byt[0,:,:,:] = g0["By"][:,:,:]
            Bzt[0,:,:,:] = g0["Bz"][:,:,:]
            Ext[0,:,:,:] = g0["Ex"][:,:,:]        
            Eyt[0,:,:,:] = g0["Ey"][:,:,:]
            Ezt[0,:,:,:] = g0["Ez"][:,:,:]
            if len(glist) >= 2:
                g1 = glist[1]
                self.t1 = g1["time"]
                tlist.append(self.t1)
                Bxt[1,:,:,:] = g1["Bx"][:,:,:]        
                Byt[1,:,:,:] = g1["By"][:,:,:]
                Bzt[1,:,:,:] = g1["Bz"][:,:,:]
                Ext[1,:,:,:] = g1["Ex"][:,:,:]        
                Eyt[1,:,:,:] = g1["Ey"][:,:,:]
                Ezt[1,:,:,:] = g1["Ez"][:,:,:]
            if len(glist) == 3:
                g2 = glist[2]
                self.t2 = g2["time"]
                tlist.append(self.t2)
                Bxt[2,:,:,:] = g2["Bx"][:,:,:]        
                Byt[2,:,:,:] = g2["By"][:,:,:]
                Bzt[2,:,:,:] = g2["Bz"][:,:,:]
                Ext[2,:,:,:] = g2["Ex"][:,:,:]        
                Eyt[2,:,:,:] = g2["Ey"][:,:,:]
                Ezt[2,:,:,:] = g2["Ez"][:,:,:]
            # The following are called with 4 arguments: Bxt_interp(t,x,y,z)
            self.Bxt_interp = RegularGridInterpolator( (tlist,xg,yg,zg), Bxt)
            self.Byt_interp = RegularGridInterpolator( (tlist,xg,yg,zg), Byt)
            self.Bzt_interp = RegularGridInterpolator( (tlist,xg,yg,zg), Bzt)
            self.Ext_interp = RegularGridInterpolator( (tlist,xg,yg,zg), Ext)
            self.Eyt_interp = RegularGridInterpolator( (tlist,xg,yg,zg), Eyt)
            self.Ezt_interp = RegularGridInterpolator( (tlist,xg,yg,zg), Ezt)

    def _update_interpolator(self):
        """
        Parses the next grid data and repeats interpolation with the last triplet of data.
        Removes the first file name from the list (oldest time).
        """
        assert len(self.files) >= 1
        g0 = self.g1
        self.g1 = self.g2
        self.g2 = self.parsefile(self.files[0])
        self._set_interpolator(g0, self.g1, self.g2)
        del self.files[0]
        
    def Bgrid(self, tpos):
        """
        Return the interpolated magnetic field vector.
        
        Parameters
        ----------
        tpos : array-like
            4-element vector of time and position x,y,z.

        Returns
        -------
        array
            3-element array (Bx,By,Bz) of the magnetic field at the specified time and position.
            
        Notes
        -----
        When called with time > (t1+t2)/2, where t0,t1,t2 are three time interpolation points,
        calls `_update_interpolator`. The interpolation is redone with grids at times t1,t2,t3.

        The `B` method is a wrapper around `Bgrid`. If further processing is required, override `B` when subclassing.      
        """
        if self._time_indep:
            Bx = self.Bxt_interp(tpos[1:])[0]
            By = self.Byt_interp(tpos[1:])[0]
            Bz = self.Bzt_interp(tpos[1:])[0]
        else:
            if self.files and tpos[0] > (self.t1 + self.t2)/2:
                self._update_interpolator()
            Bx = self.Bxt_interp(tpos)[0]
            By = self.Byt_interp(tpos)[0]
            Bz = self.Bzt_interp(tpos)[0]    
        return np.array([Bx,By,Bz])
        
    def Egrid(self, tpos):
        """
        Return the interpolated electric field vector.
        
        Parameters
        ----------
        tpos : array-like
            4-element vector of time and position x,y,z.

        Returns
        -------
        array
            3-element array (Ex,Ey,Ez) of the electric field at the specified time and position.
            
        Notes
        -----
        When called with time > (t1+t2)/2, where t0,t1,t2 are three time interpolation points,
        calls `_update_interpolator`. The interpolation is redone with grids at times t1,t2,t3.

        The `E` method is a wrapper around `Egrid`. If further processing is required, override `E` when subclassing.
        """
        if self._time_indep:
            Ex = self.Ext_interp(tpos[1:])[0]
            Ey = self.Eyt_interp(tpos[1:])[0]
            Ez = self.Ezt_interp(tpos[1:])[0]
        else:
            if self.files and tpos[0] > (self.t1 + self.t2)/2:
                self._update_interpolator()
            Ex = self.Ext_interp(tpos)[0]
            Ey = self.Eyt_interp(tpos)[0]
            Ez = self.Ezt_interp(tpos)[0]    
        return np.array([Ex,Ey,Ez])
    
    def B(self,tpos): # Override when subclassing
        """
        Return the magnetic field vector.
        
        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element array (Bx, By, Bz)
        
        Notes
        -----
        Currently a wrapper around Bgrid. When subclassing, add further processing in this method as needed.
        """
        res = self.Bgrid(tpos)
        # You can add further processing here; e.g. add a dipole component if necessary.
        return res

    def E(self,tpos): # Override when subclassing
        """
        Return the electric field vector.
        
        Parameters
        ----------
        tpos : array-like
            4-element array of time and x,y,z coordinates.
        
        Returns
        -------
        array
            3-element array (Ex, Ey, Ez)
        
        Notes
        -----
        Currently a wrapper around Egrid. When subclassing, add further processing in this method as needed.
        """
        res = self.Egrid(tpos)
        return res
