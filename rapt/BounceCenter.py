# -*- coding: utf-8 -*-
"""
BounceCenter class definition

AUTHOR:
    Kaan Ozturk <mkozturk@yahoo.com>
    
    2016, Rice University
"""
import numpy as np
from scipy.integrate import ode
import pickle
import rapt.utils as ru
from rapt import c, params
from rapt.flutils import halfbouncepath, bounceperiod, gradI
    
class BounceCenter:
    """
    Bounce-averaged tracer mode. Works only with static magnetic fields.
    
    Parameters
    ----------
    pos: list or array
        The initial position (x,y,z) of the particle, in meters.
    v: float
        The initial speed of the particle, in m/s.
    t0: float
        The time of simulation at the beginning (seconds).
    pa: float
        The initial pitch angle, in degrees
    mass: float
        The mass of the particle, in kg.
    charge: float
        The charge of the particle, in Coulombs.
    field: Field object
        The field object that provides electric and magnetic field vectors and
        related quantities.

    Attributes
    ----------    
    trajectory : n-by-4 array
        The trajectory array, each row in the form (t, x, y, z).
    isequatorial : bool
        If True, uses gradient and inertial drift velocities to advance the
        solution. For particles with no longitudinal motion.
                    
    See Also
    --------
    rapt.Particle
    rapt.GuidingCenter

    Notes
    -----
    The bounce-averaged motion of a particle is the pure drift motion resulting
    from averaging out the field-aligned longitudinal motion. This averaging
    requires that the change in the field from one bounce to the next is small,
    i.e., the first and the second invariants exist.
    
    Theoretically, bounce-averaged drift should be the most efficient mode of
    tracing, if the adiabaticity conditions are satisfied. However, each
    evaluation of the equation of motion involves determining the gradient of
    the second invariant and of the half-bounce path length, comprising 5 traces
    of field lines. Because the ode-solver evaluates the eom many times within 
    one integration step, the computation is very costly even for moderate error
    tolerances.
    
    In the cases we tried, following a `GuidingCenter` is faster than following a
    `BounceCenter`, even for low-energy electrons. Because of this, we did not
    extend BounceCenter to work with electric and nonstatic magnetic fields, and
    did not extend `Adaptive' to include a `BounceCenter` mode.
    
    
    """
    def __init__(self, pos=[], v=None, t0=0, pa=None, mass=None, charge=None, field=None):
        """
        Object constructor.
        
        Parameters
        ----------
        pos: list or array
            The initial position (x,y,z) of the particle, in meters.
        v: float
            The initial speed of the particle, in m/s.
        t0: float
            The time of simulation at the beginning (seconds).
        pa: float
            The initial pitch angle, in degrees
        mass: float
            The mass of the particle, in kg.
        charge: float
            The charge of the particle, in Coulombs.
        field: Field object
            The field object that provides electric and magnetic field vectors and
            related quantities.
        """
        self.pos = pos  # initial position array
        self.v = v    # speed of the particle
        self.t0 = t0
        self.tcur = t0     # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  # the field object
        self.isequatorial = False
        if not field.static:
            raise RuntimeError("BounceCenter does not work with nonstatic fields or electric fields.")
            return None
        # The initial pitch angle:
        self.pa = pa
        
        if not (pos==[] or v==None or pa==None): # if initial state is given explicitly
            self.trajectory = np.concatenate(([t0], pos))
            self.trajectory = np.reshape(self.trajectory, (1,4))
            # the first invariant value (constant)
            self.mu = ru.magnetic_moment(t0, pos, self.v*np.cos(self.pa), self.v, field, mass)
            assert self.mu>0

    def setpa(self, pa):
        """
        Reinitialize the object with the given pitch angle (in degrees).
        
        Modifies the velocity vector while keeping the energy constant so that
        the particle's pitch angle (angle between the velocity and magnetic field
        vectors) is `pa` degrees. Runs the `__init__` method, so any existing data
        will be lost.
        
        Parameters
        -----------
        pa : float
            The new pitch angle in degrees.
        """
        field = self.field
        tpos = self.trajectory[-1,0:4]
        v = self.trajectory[-1,4:] # velocity
        s = np.sqrt(np.dot(v,v))   # speed
        # Construct a vector that has the same length as v,
        # lies on the plane spanned by v and B,
        # and makes an angle of pa with B.
        
        # the unit vector in the direction of the field.
        b = field.unitb(tpos)
        # component of v perpendicular to b:
        spar = np.dot(v,b)   # parallel speed
        if abs(spar-s) < 1e-12:  # if velocity is parallel to b
            p = ru.getperp(b)  # get a vector perp. to b, direction otherwise 'random'
        else:
            vperp = v - spar*b # component of v perpendicular to b
            p = vperp / np.sqrt(np.dot(vperp,vperp)) # unit vector of vperp

        # The new velocity vector:
        w = s*np.sin(pa*np.pi/180)*p + s*np.cos(pa*np.pi/180)*b
        # Reinitialize with the new velocity:
        self.__init__(self.pos, w, self.t0, self.mass, self.charge, self.field)
    
    def _v_grad(self, tpos):
        """Return the gradient-drift velocity.

        Parameters
        ----------
        tpos : array
            4-vector of time and position (t,x,y,z)
        
        Returns
        -------
        array
            The gradient-drift velocity at the given time and position.
            :math: `\mathbf{v}_d = \frac{\mu}{q\gamma B^2} \mathbf{B}\times\nabla B`
        """
        gamma = 1.0/np.sqrt(1 - (self.v / c)**2)
        Bvec = self.field.B(tpos)
        magBsq = np.dot(Bvec, Bvec)
        gB = self.field.gradB(tpos)
        return self.mu*np.cross(Bvec,gB)/(self.charge*gamma*magBsq)
    
    def _v_inert(self,tpos):
        """Return the inertial drift velocity.

        Parameters
        ----------
        tpos : array
            4-vector of time and position (t,x,y,z)
        
        Returns
        -------
        array
            The gradient-drift velocity at the given time and position.
            :math: `\mathbf{v}_i = \frac{d\mathbf{v}_d}{dt} = (\mathbf{v}_d\cdot\nabla)\mathbf{v}_d`
            under the assumed conditions (no electric field, no time dependence,
            no parallel motion)
        """
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
        """
        Advance the particle position for the given duration.
        
        The trajectory is initialized at the latest state of the object
        and integrated for an additional `delta` seconds. Uses the 
        `scipy.integrate.ode` class with `"dopri5"` solver.
        
        This method can be called many times.
        
        Parameters
        ----------
        delta : float
            The number of seconds to advance the trajectory.
            
        Notes
        -----
        The explicit runge-kutta method of order 4(5) due to Dormand & Prince
        with stepsize control is used to solve for the motion. The relative 
        tolerance `rtol` and the absolute tolerance `atol` of the solver can be
        set with `rapt.params["solvertolerances"] = (rtol, atol)`

        """
        v = self.v
        gamma = 1.0/np.sqrt(1 - (v / c)**2)
        Bm = self.mass*gamma**2*v**2/(2*self.mu)
        bp = bounceperiod(self.trajectory[-1,:4], self.field, Bm, v)
        dt = params['BCtimestep'] * bp
        
        def deriv(t,Y):
            tpos = np.concatenate(([t],Y))
            if self.isequatorial:
                return self._v_grad(Y[:4])+self._v_inert(Y[:4])
            else:
                Bvec = self.field.B(tpos)
                magBsq = np.dot(Bvec, Bvec)
                Sb = halfbouncepath(tpos, self.field, Bm)
                gI = gradI(tpos, self.field, Bm)
                return gamma*self.mass*v*v/(self.charge*Sb*magBsq) * np.cross(gI,Bvec)
        rtol, atol = params["solvertolerances"]
        r = ode(deriv).set_integrator("dopri5", atol=atol, rtol=rtol)
        r.set_initial_value(self.trajectory[-1,1:], self.trajectory[-1,0])
        for t in np.arange(self.tcur, self.tcur+delta, dt):
            nextpt = np.hstack(([t],r.integrate(r.t+dt)))
            self.trajectory = np.vstack((self.trajectory,nextpt))
        self.tcur = self.trajectory[-1,0]
        
    def save(self,filename):
        """
        Save the object on disk.
        
        Uses the built-in pickle module.
        
        Parameters
        ----------
        filename : str
            The name of the file to store. If file exists, it will be overwritten.
        """
        f = open(filename,"wb")
        pickle.dump(self,f)
        f.close()
    
    def load(self,filename):
        """
        Load the object from disk.
        
        Uses the built-in pickle module. All existing data is replaced
        with the stored data.
        
        Parameters
        ----------
        filename : str
            The name of the file where the object is stored. 
        
        """

        f = open(filename, "rb")
        p = pickle.load(f)
        f.close()
        for k in p.__dict__.keys():
            self.__dict__[k] = p.__dict__[k]


    def gett(self):
        """Return a 1-d array of time values along the trajectory."""
        return self.trajectory[:,0]
    def getx(self):
        """Return a 1-d array of x-coordinate values along the trajectory."""
        return self.trajectory[:,1]
    def gety(self):
        """Return a 1-d array of x-coordinate values along the trajectory."""
        return self.trajectory[:,2]
    def getz(self):
        """Return a 1-d array of x-coordinate values along the trajectory."""
        return self.trajectory[:,3]
    def getr(self):  # Distance from the origin
        """Return a 1-d array of radial distance along the trajectory."""
        return np.sqrt( self.getx()**2 + self.gety()**2 + self.getz()**2 )
    def gettheta(self):
        """Return a 1-d array of azimuthal angle coordinate (radians) along the trajectory."""
        return np.arctan2( self.gety(), self.getx() )
    def getphi(self):
        """Return a 1-d array of polar angle coordinate (radians) along the trajectory."""
        return np.arccos( self.getz() / self.getr() )
    def getB(self):
        """Return a 1-d array of total magnetic field strength values along the 
        trajectory."""
        out = np.zeros(len(self.trajectory))
        for i, row in enumerate(self.trajectory):
            out[i] = self.field.magB(row[:4])
        return out
