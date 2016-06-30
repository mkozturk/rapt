# -*- coding: utf-8 -*-
"""
Adaptive class definition

AUTHOR:
    Kaan Ozturk <mkozturk@yahoo.com>
    
    2016, Rice University
"""
import numpy as np
import pickle
from rapt import Particle, GuidingCenter, Adiabatic, NonAdiabatic
class Adaptive:
    """
    A tracer that can switch between Particle and GuidingCenter modes depending
    on local adiabaticity conditions.    
    
    Parameters
    ----------
    pos: list or array
        The initial position (x,y,z) of the particle, in meters.
    v: float
        The initial speed of the particle, in m/s.
    pa: float
        The initial pitch angle, in degrees (not needed if ppar is given)
    ppar: float
        The initial parallel momentum, in kg m/s (not needed if pa is given)
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
    trajlist : list
        The list that stores `Particle` and `GuidingCenter` objects that comprise
        the whole trajectory.
                    
    See Also
    --------
    rapt.Particle
    rapt.GuidingCenter

    Notes
    -----
    An `Adaptive` object is a wrapper around `Particle` and `GuidingCenter`
    objects. It s initialized as a `Particle`. Initially, and after every
    integration step, the adiabaticity condition is checked. If it is satisfied
    and the current tracer is a `Particle`, a `GuidingCenter` is initialized
    with the last state. If the adiabaticity condition is violated while
    the current tracer is a `GuidingCenter`, a `Particle` is initialized with
    the last state. These different tracer objects are stored in the `trajlist`
    attribute, and the data is accessed with the getter members `gett`,
    `getx`, etc. seamlessly.

    The `Adaptive` tracer sets the `check_adiabaticity` attributes of `Particle`
    and `GuidingCenter` objects it generates. It detects the switching points by
    catching the exceptions `rapt.Adiabatic` and `rapt.NonAdiabatic`.

    Examples
    --------
    
    """
    def __init__(self, pos=None, vel=None, t0=0, mass=None, charge=None, field=None):
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
        """
        
        self.pos = np.array(pos)  # initial position array
        self.vel = np.array(vel)  # initial velocity array
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  #  the field object
        self.p = Particle(pos,vel,t0,mass,charge,field) # Store for referencing in setpa and setKE
        self.p.check_adiabaticity = True
        if self.p.isadiabatic():
            g = GuidingCenter()
            g.check_adiabaticity = True
            g.init(self.p)
            self.trajlist = [g]
        else:
            self.trajlist = [self.p]

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

    def setke(self, ke, unit="ev"):
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
        self.p.setke(ke, unit)
        self.p.check_adiabaticity = True
        if self.p.isadiabatic():
            g = GuidingCenter()
            g.init(self.p)
            g.check_adiabaticity = True
            self.trajlist = [g]
        else:
            self.trajlist = [self.p]
    
    def setpa(self, pa):
        """
        Reinitialize the object with the given pitch angle (in degrees).
        

        Modifies the velocity vector while keeping the energy constant so that
        the particle's pitch angle (angle between the velocity and magnetic field
        vectors) is `pa` degrees. Previous data is lost.
        
        Parameters
        -----------
        pa : float
            The new pitch angle in degrees.
        """
        self.p.setpa(pa)
        self.p.check_adiabaticity = True
        if self.p.isadiabatic():
            g = GuidingCenter()
            g.init(self.p)
            g.check_adiabaticity = True
            self.trajlist = [g]
        else:
            self.trajlist = [self.p]

    def advance(self,delta):
        """
        Advance the tracer position and the relevant momentum for a given duration.
        
        The trajectory is initialized at the latest state of the current tracer
        and integrated for an additional `delta` seconds. Uses the settings
        specific to the current tracer.
        
        This method can be called many times.
        
        Parameters
        ----------
        delta : float
            The number of seconds to advance the trajectory.
        """
        t = 0
        current = self.trajlist[-1]
        assert current.check_adiabaticity == True
        while t < delta:
            try:
                current.advance(delta-t)
            except NonAdiabatic:
                p = Particle()
                p.init(current)
                p.check_adiabaticity = True
                self.trajlist.append(p)
                current = self.trajlist[-1]
                print("Switched to particle mode at time", current.tcur,flush=True)
            except Adiabatic:
                g = GuidingCenter()
                g.init(current)
                g.check_adiabaticity = True
                self.trajlist.append(g)
                current = self.trajlist[-1]
                print("Switched to guiding center mode at time", current.tcur,flush=True)
            t = current.tcur

    def gett(self):
        """Return a 1-d array of time values along the trajectory."""
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.gett() ) )
        return res
    def getx(self):
        """Return a 1-d array of x-coordinate values along the trajectory."""
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getx()) )
        return res
    def gety(self):
        """Return a 1-d array of y-coordinate values along the trajectory."""
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.gety()) )
        return res
    def getz(self):
        """Return a 1-d array of z-coordinate values along the trajectory."""
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getz()) )
        return res
    def getr(self):
        """Return a 1-d array of radial distance along the trajectory."""
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getr()) )
        return res
    def getphi(self):
        """Return a 1-d array of polar angle coordinate (radians) along the
        trajectory."""
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getphi()) )
        return res
    def gettheta(self):
        """Return a 1-d array of azimuthal angle coordinate (radians) along the
        trajectory."""
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.gettheta()) )
        return res    
    def getke(self):
        """Return a 1-d array of kinetic energy values (Joule) along the
        trajectory."""
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getke()) )
        return res
