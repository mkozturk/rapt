# -*- coding: utf-8 -*-
"""
Particle class definition

AUTHOR:
    Kaan Ozturk <mkozturk@yahoo.com>
    
    2016, Rice University
"""

import numpy as np
from scipy.integrate import ode
import pickle

import rapt.utils as ru
from rapt import c, params, Adiabatic

class Particle:
    """
    A classical particle (i.e., nonquantum, but relativistic), such as an electron or a proton,
    moving under the influence of given electric and magnetic fields.        
    
    Parameters
    ----------
    pos: list or array
        The initial position (x,y,z) of the particle, in meters.
    vel: list or array
        The initial velocity (vx, vy, vz) of the particle, in m/s.
    t0: float
        The time of simulation at the beginning (seconds), ignored for fields that do not depend on time.
    mass: float
        The mass of the particle, in kg.
    charge: float
        The charge of the particle, in Coulombs.
    field: Field object
        The field object that provides electric and magnetic field vectors and related quantities.

    Attributes
    ----------    
    tcur : float
        The current time value (seconds), updated after every integration step.
    trajectory : n-by-7 array
        The trajectory array, each row in the form (t, x, y, z, px, py, pz).
    check_adiabaticity : bool
        If True, checks for the adiabaticity condition after every integration step.
    
    See Also
    --------
    rapt.GuidingCenter
    rapt.Adaptive

    Notes
    -----

    Examples
    --------
    
    """
    def __init__(self, pos=[], vel=[], t0=0, mass=None, charge=None, field=None):
        """
        Object constructor.
        
        Parameters
        ----------
        pos: list or array
            The initial position (x,y,z) of the particle, in meters.
        vel: list or array
            The initial velocity (vx, vy, vz) of the particle, in m/s.
        t0: float
            The time of simulation at the beginning (seconds), ignored for fields that do not depend on time.
        mass: float
            The mass of the particle, in kg.
        charge: float
            The charge of the particle, in Coulombs.
        field: Field object
            The field object that provides electric and magnetic field vectors and related quantities.
        
        See Also
        --------
        init
        
        Notes
        -----
        All parameters above are optional. The object can be initialized with
        an empty parameter set if it is going to be initialized differently
        (e.g. with `init` method)
        
        
        """
        self.pos = np.array(pos)  # initial position array
        self.vel = np.array(vel)  # initial velocity array
        self.t0 = t0    # initial time
        self.tcur = t0    # current time
        """The current time value (seconds), updated after every integration step."""
        
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  #  the field object
        self.trajectory = np.zeros((1,7))
        """The trajectory array, each row in the form (t, x, y, z, px, py, pz)."""
        
        self.check_adiabaticity = False
        """If True, checks for the adiabaticity condition after every integration step."""

        if not (pos == [] or vel == [] or self.mass == None): # if initial state is given explicitly
            gamma = 1/np.sqrt(1 - np.dot(self.vel,self.vel)/c**2)
            mom = self.mass*gamma*self.vel
            self.trajectory = np.concatenate(([self.tcur], self.pos, mom))
            self.trajectory = np.reshape(self.trajectory, (1,7))
            
    def init(self, p, gyrophase=0):  # Initialization with another Particle or GuidingCenter instance
        """
        Initialize a Particle using the state of another Particle or GuidingCenter object.
        
        Parameters
        -----------
        p : Particle or GuidingCenter object.
            The object that is used to initialize the current Particle.
        gyrophase : float, optional.
            The phase angle of the gyromotion, if initializing with a GuidingCenter.
            
        See Also
        --------
        rapt.utils.GCtoFP
        
        Notes
        -----
        Takes the last state (position and momentum) of the given `Particle`
        or `GuidingCenter`, sets the initial conditions of self to match them,
        and runs `__init__` again. Therefore, all existing data is erased.
        
        Initialization from a `GuidingCenter` involves adding the gyroradius vector
        to the position. This process introduces an arbitrariness in the phase angle.
        This arbitrariness cannot be avoided because it is inherent in the definition
        of the guiding center. The optional `gyrophase` parameter allows the user to 
        set it to different values, e.g. to produce an ensemble of particles.
      
        """
        from rapt import GuidingCenter # Import here to avoid circular imports.        
        if isinstance(p, Particle):
            mom = p.trajectory[-1,4:]
            gm = np.sqrt(self.mass**2 + np.dot(mom,mom)/c**2)
            self.__init__(pos=p.trajectory[-1,1:4], 
                          vel=p.trajectory[-1,4:]/gm,
                          t0=p.trajectory[-1,0], 
                          mass=p.mass, 
                          charge=p.charge,
                          field=p.field)
        elif isinstance(p, GuidingCenter):
            B = p.field.magB(p.trajectory[-1,:4])            
            gammasq = 1 + 2*p.mu*B/(p.mass*c*c) + (p.trajectory[-1,4]/p.mass/c)**2
            if np.sqrt(gammasq)-1 < 1e-6: # nonrelativistic
                v = np.sqrt(2*p.mu*B/p.mass + (p.trajectory[-1,4]/p.mass)**2)
            else: # relativistic
                v = c * np.sqrt(1-1/gammasq)
            vpar = p.trajectory[-1,4]/p.mass/np.sqrt(gammasq)
            pos, vel = ru.GCtoFP(self.tcur, p.trajectory[-1,1:4], vpar, v,
                                 p.field, p.mass, p.charge, gyrophase)
            self.__init__(pos=pos, 
                          vel=vel,
                          t0=p.trajectory[-1,0], 
                          mass=p.mass, 
                          charge=p.charge,
                          field=p.field)
        else:
            raise(ValueError, "Particle or GuidingCenter objects required.")

    def setke(self,ke, unit="ev"):
        """
        Scale the velocity vector with the speed corresponding to the given kinetic energy.
        Reinitializes the object.
        
        Parameters
        -----------
        ke : float
            The kinetic energy of the particle (eV by default). Can be relativistic.
        unit : str, optional
            The unit of the energy. If "ev", electron volts, otherwise Joule.
        
        """
        assert ke > 0        
        s = ru.speedfromKE(ke, self.mass, unit)
        mom = self.trajectory[-1,4:]
        gm = np.sqrt(self.mass**2 + np.dot(mom,mom)/c**2) # gamma * m
        v = mom/gm # velocity
        v = v*(s/np.sqrt(np.dot(v,v)))
        self.__init__(self.pos, v, self.t0, self.mass, self.charge, self.field)

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
        mom = self.trajectory[-1,4:] # momentum
        gm = np.sqrt(self.mass**2 + np.dot(mom,mom)/c**2) # gamma * m
        v = mom/gm # velocity
        s = np.sqrt(np.dot(v,v))   # speed
        
        # Construct a vector that has the same length as v,
        # lies on the plane spanned by v and B,
        # and makes an angle of pa with B.
        
        # the unit vector in the direction of the field.
        b = field.unitb(tpos)
        spar = np.dot(v,b)   # component of v parallel to B
        
        if abs(spar-s) < 1e-12:  # if velocity is parallel to b
            p = ru.getperp(b)  # get a vector perp. to b, direction otherwise 'random'
        else:
            vperp = v - spar*b # component of v perpendicular to b
            p = vperp / np.sqrt(np.dot(vperp,vperp)) # unit vector of vperp
        
        # The new velocity vector
        w = s*np.sin(pa*np.pi/180)*p + s*np.cos(pa*np.pi/180)*b
        # Reinitialize with the new velocity
        self.__init__(self.pos, w, self.t0, self.mass, self.charge, self.field)
        
    def advance(self, delta):
        """
        Advance the particle position and momentum for a given duration.
        
        The trajectory is initialized at the latest state of the particle and 
        integrated for an additional `delta` seconds. Uses the `scipy.integrate.ode`
        class with `"dop853"` solver.
        
        This method can be called many times.
        
        Parameters
        ----------
        delta : float
            The number of seconds to advance the trajectory.
        
        Raises
        -------
        Adiabatic
            Only if the `check_adiabaticity` attribute is set to True.
            
        Notes
        -----
        The particle is subject to the Newton-Lorentz equation of motion.
        
        .. math::
        
            \\frac{\mathrm{d}\\vec{x}}{\mathrm{d}t} = \\frac{\\vec{p}}{\gamma m} \\\\

            \\frac{\mathrm{d}\\vec{p}}{\mathrm{d}t} = q\\vec{E} + q\\vec{v}\\times\\vec{B}

        where :math:`\gamma = \sqrt{1 + (|\\vec{p}|/mc)^2}` is the relativistic factor.
        
        The explicit runge-kutta method of order 8(5,3) due to Dormand & Prince
        with stepsize control is used to solve for the motion. The relative 
        tolerance `rtol` and the absolute tolerance `atol` of the solver can be
        set with: `rapt.params["solvertolerances"] = (rtol, atol)`
        
        The `beta` parameter to the `scipy.integrate.ode.setintegrator` method
        serves to stabilize the solution in gridded fields.
        """
        rtol, atol = params["solvertolerances"]        
        t0 = self.trajectory[-1,0]        
        pos = self.trajectory[-1,1:4]
        mom = self.trajectory[-1,4:]
        gm = np.sqrt(self.mass**2 + np.dot(mom,mom)/c**2) # gamma * mass
        vel = mom/gm
        # set resolution of cyclotron motion
        if params["Ptimestep"] != 0:
            dt = params["Ptimestep"]
        else:
            dt = self.cycper()/params["cyclotronresolution"]

        dt = ru.cyclotron_period(t0, pos, vel, self.field, self.mass, self.charge) / params['cyclotronresolution']
        
        def eom(t, Y):  # The Newton-Lorenz equation of motion
            # Y = x,y,z,px,py,pz
            nonlocal gm
            tpos = np.concatenate(([t],Y[:3]))
            mom = Y[3:]
            out = np.zeros(6)
            if not self.field.static:  # don't reevaluate gamma every time, if it is not supposed to change.
                gm = np.sqrt(self.mass**2 + np.dot(mom,mom)/c**2) # gamma * mass        
            # d(pos)/dt = vel = mom / (gamma*m) :
            out[0:3] = Y[3:]/gm    
            # dp/dt = q(E(t, pos) + (p/gamma*m) x B(t, pos)) :      
            out[3:] = self.charge * (self.field.E(tpos) + np.cross(Y[3:], self.field.B(tpos))/gm)
            if params["enforce equatorial"]:
                out[2] = out[5] = 0
            return out

        r = ode(eom).set_integrator("dop853",beta=0.1, rtol=rtol, atol=atol) 
        # Setting beta is necessary to reduce the error.
        
        r.set_initial_value(self.trajectory[-1,1:], self.trajectory[-1,0])
        while r.successful() and r.t < t0+delta:
            next = np.hstack(([r.t+dt],r.integrate(r.t+dt)))
            self.tcur = r.t+dt
            self.trajectory = np.vstack((self.trajectory,next))
            if self.check_adiabaticity and self.isadiabatic():
                raise Adiabatic
    
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
        
        Uses the built-in pickle module. All existing Particle data is replaced
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
        
    def isadiabatic(self):
        """
        Check if the particle motion is adiabatic at the current location.
        
        The adiabaticity condition is defined as
        
        .. math::

            \\rho_c / L < \\epsilon_s

            \\tau_c / T < \\epsilon_t
        
        where :math:`\\rho_c` is the cyclotron radius, :math:`\\tau_c` is the 
        cyclotron period, L is the field length scale, T is the field time scale,
        and :math:`\epsilon_s, \epsilon_t` are user-supplied thresholds for adiabaticity.
        The length scales are provided by the field object, and threshold 
        parameters set with `rapt.params["epss"]` and `rapt.params["epst"]`, 
        respectively.
        
        Returns
        -------
        bool
            True if the particle's motion satisfies the adiabaticity conditions
        at the present location, False otherwise.
        """
        
        # Needs to be rewritten to avoid repeated calculations of the field vector.
        
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
        """Return a 1-d array of time values along the trajectory."""
        return self.trajectory[:,0]
    def getx(self):
        """Return a 1-d array of x-coordinate values along the trajectory."""
        return self.trajectory[:,1]
    def gety(self):
        """Return a 1-d array of y-coordinate values along the trajectory."""
        return self.trajectory[:,2]
    def getz(self):
        """Return a 1-d array of z-coordinate values along the trajectory."""
        return self.trajectory[:,3]
    def getpx(self):
        """Return a 1-d array of the x-component of the momentum along the trajectory."""
        return self.trajectory[:,4]
    def getpy(self):
        """Return a 1-d array of the y-component of the momentum along the trajectory."""
        return self.trajectory[:,5]
    def getpz(self):
        """Return a 1-d array of the z-component of the momentum along the trajectory."""
        return self.trajectory[:,6]
    def getp(self):
        """Return a 1-d array of total momentum values along the trajectory."""
        return np.sqrt(self.getpx()**2 + self.getpy()**2 + self.getpz()**2)
    def getgamma(self):
        """Return a 1-d array of relativistic factor values along the trajectory.
        
        The relativistic factor is defined as :math:`\gamma = \sqrt{1 + (p/mc)^2}`
        """
        psq = self.trajectory[:,4]**2 + self.trajectory[:,5]**2 + self.trajectory[:,6]**2
        return np.sqrt(1 + psq/(self.mass*c)**2)

    def getvx(self):
        """Return a 1-d array of the x-component of the velocity along the trajectory."""
        return self.getpx()/self.getgamma()/self.mass
    def getvy(self):
        """Return a 1-d array of the y-component of the velocity along the trajectory."""
        return self.getpy()/self.getgamma()/self.mass
    def getvz(self):
        """Return a 1-d array of the z-component of the velocity along the trajectory."""
        return self.getpz()/self.getgamma()/self.mass
    def getv(self):  # total speed of the particle
        """Return a 1-d array of the particle speed along the trajectory."""
        psq = self.trajectory[:,4]**2 + self.trajectory[:,5]**2 + self.trajectory[:,6]**2
        gammasq = 1 +psq/(self.mass*c)**2
        return np.sqrt(1-gammasq)*c
        
    def getr(self):  # Distance from the origin
        """Return a 1-d array of radial distance along the trajectory."""
        return np.sqrt( self.getx()**2 + self.gety()**2 + self.getz()**2 )
    def gettheta(self):
        """Return a 1-d array of azimuthal angle coordinate (radians) along the trajectory."""
        return np.arctan2( self.gety(), self.getx() )
    def getphi(self):
        """Return a 1-d array of polar angle coordinate (radians) along the trajectory."""
        return np.arccos( self.getz() / self.getr() )
    def getke(self):      # Kinetic energy
        """Return a 1-d array of kinetic energy values (Joule) along the trajectory."""
        gamma = self.getgamma()
        # Try to vectorize this for better performance
        ke_nr = 0.5*(self.trajectory[:,4]**2 + self.trajectory[:,5]**2 + self.trajectory[:,6]**2)/self.mass
        res = np.zeros(gamma.shape)
        mc2 = self.mass*c*c
        for i,g in enumerate(gamma):
            if g-1 < 1e-6:  # nonrelativistic
                res[i] = ke_nr[i]
            else:  # relativistic
                res[i] = (g-1)*mc2
        return res
    
    def getB(self):
        """Return a 1-d array of total magnetic field strength values along the 
        trajectory."""
        out = np.zeros(len(self.trajectory))
        for i, row in enumerate(self.trajectory):
            out[i] = self.field.magB(row[:4])
    
    def guidingcenter(self):
        """Return a 2-d array of guiding center positions corresponding to particle positions."""
        out = []
        for row in self.trajectory:
            t,r,mom = row[0], row[1:4], row[4:]
            gm = np.sqrt(self.mass**2 + np.dot(mom,mom)/c**2) # gamma * mass
            v = mom/gm
            rgc, vp, spd = ru.guidingcenter(t, r, v, self.field, self.mass, self.charge)
            out.append(list(rgc)+[vp,spd])
        return np.array(out)
    def mu(self):
        """Return a 1-d array of the magnetic moment values along the trajectory."""
        out = []
        for row in self.trajectory:
            t, r, mom = row[0], row[1:4], row[4:]
            gm = np.sqrt(self.mass**2 + np.dot(mom,mom)/c**2) # gamma * mass
            v = mom/gm
            rgc, vp, spd = ru.guidingcenter(t, r, v, self.field, self.mass, self.charge)
            out.append(ru.magnetic_moment(t, rgc, vp, spd, self.field, self.mass))
        return np.array(out)
    def cycrad(self):
        """Return the cyclotron radius value at the final position."""
        t, r, mom = self.trajectory[-1, 0], self.trajectory[-1, 1:4], self.trajectory[-1, 4:]
        gm = np.sqrt(self.mass**2 + np.dot(mom,mom)/c**2) # gamma * mass
        v = mom/gm
        return ru.cyclotron_radius(t, r, v, self.field, self.mass, self.charge)
    def cycper(self):
        """Return the cyclotron period value at the final position."""
        t, r, mom = self.trajectory[-1, 0], self.trajectory[-1, 1:4], self.trajectory[-1, 4:]
        gm = np.sqrt(self.mass**2 + np.dot(mom,mom)/c**2) # gamma * mass
        v = mom/gm
        return ru.cyclotron_period(t, r, v, self.field, self.mass, self.charge)