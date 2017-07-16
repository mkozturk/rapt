# -*- coding: utf-8 -*-
"""
GuidingCenter class definition

AUTHOR:
    Kaan Ozturk <mkozturk@yahoo.com>
    
    2016, Rice University
"""
import numpy as np
from scipy.integrate import ode
import pickle

import rapt.utils as ru
import rapt.flutils as rfu
from rapt import c, params, NonAdiabatic

class GuidingCenter:
    """
    A GuidingCenter of a charged particle, moving under the influence of given 
    electric and magnetic fields.        
    
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
    trajectory : n-by-5 array
        The trajectory array, each row in the form (t, x, y, z, ppar).
    check_adiabaticity : bool
        If True, checks for the adiabaticity condition after every integration step.
                    
    See Also
    --------
    rapt.Particle
    rapt.Adaptive

    Notes
    -----

    Examples
    --------
    
    """
    def __init__(self, pos=[], v=0, pa=None, ppar=None, t0=0, mass=None, charge=None, field=None):
        """
        Object constructor.
        
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
        
        See Also
        --------
        init
        
        Notes
        -----
        All parameters above are optional. The object can be initialized with
        an empty parameter set if it is going to be initialized differently
        (e.g. with `init` method)
        
        
        """
        # pos: initial position (array of 3)
        # v: initial speed (scalar)
        # pa: initial pitch angle
        # ppar = initial parallel momentum
        #     (N.B. Either pa or ppar should be provided. If both given, pa is used.)
        # t0: initial time
        # mass: particle mass in kg
        # charge: particle charge in C
        # field: The field object
    
        self.pos = pos  # initial position array
        self.v = v

        self.t0 = t0 # initial time
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field
        self.trajectory = np.zeros((1,5))
        self.check_adiabaticity = False
        # The object can be initialized two ways:
        # Either by specifying the initial conditions,
        # or by an empty call, to be initialized later by another object.
        # We consider the call empty when pos and v are not specified.
        
        if not (pos==[] or v==0): # if initial state is given explicitly
            gamma = 1/np.sqrt(1-(v/c)**2)
            if pa != None:
                vpar = 0 if pa==90 else v * np.cos(pa*np.pi/180)
                ppar = gamma*mass*vpar

            self.mu = ru.magnetic_moment(self.tcur, self.pos, ppar/(mass*gamma),
                                         self.v, self.field, self.mass)
            self.trajectory[0,0] = t0
            self.trajectory[0,1:4] = pos[:]
            self.trajectory[0,4] = ppar
    
    def init(self, p):
        """
        Initialize a Particle using the state of another Particle or GuidingCenter object.
        
        Parameters
        -----------
        p : Particle or GuidingCenter object.
            The object that is used to initialize the current GuidingCenter.
        
        Notes
        -----
        Takes the last state (position and momentum) of the given `Particle`
        or `GuidingCenter`, sets the initial conditions of self to match them,
        and runs `__init__` again. Therefore, all existing data is erased.
        
        """
        from rapt import Particle # Import here to avoid circular imports. 
        if isinstance(p, GuidingCenter):
            B = p.field.magB(p.trajectory[-1,:4])            
            gamma = np.sqrt(1 + 2*p.mu*B/(p.mass*c*c) + (p.trajectory[-1,4]/p.mass/c)**2)
            if gamma-1 < 1e-6: # nonrelativistic
                v = np.sqrt(2*p.mu*B/p.mass + (p.trajectory[-1,4]/p.mass)**2)
            else:
                v = c * np.sqrt(1-1/gamma**2)
            self.__init__(pos = p.trajectory[-1,1:4],
                          v = v,
                          ppar = p.trajectory[-1,4],
                          t0 = p.trajectory[-1,0],
                          mass = p.mass, 
                          charge = p.charge,
                          field = p.field)
            self.check_adiabaticity = p.check_adiabaticity
                
        elif isinstance(p, Particle):
            mom = p.trajectory[-1,4:]
            gm = np.sqrt(p.mass**2 + np.dot(mom,mom)/c**2) # gamma * m
            vel = mom/gm # velocity
            pos, vp, v = ru.guidingcenter(p.trajectory[-1,0],
                                          p.trajectory[-1,1:4],
                                          vel, 
                                          p.field, 
                                          p.mass,
                                          p.charge)
            gamma = 1 / np.sqrt(1-(v/c)**2)
            self.__init__(pos = pos,
                          v = v,
                          ppar = p.mass*gamma*vp,
                          t0 = p.trajectory[-1,0],
                          mass = p.mass, 
                          charge = p.charge,
                          field = p.field)
            self.check_adiabaticity = p.check_adiabaticity
        else:
            raise(ValueError, "Particle or GuidingCenter objects required.")

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

        # Calculate the current pitch angle
        mc = self.mass*c
        t,x,y,z,ppar = self.trajectory[-1]
        B = self.field.magB(self.trajectory[-1,:4])          
        gammasq_minus_1 = 2*self.mu*B/(mc*c) + (ppar/mc)**2
        if np.sqrt(gammasq_minus_1 + 1) - 1 < 1e-6:  # nonrelativistic
            ptot = np.sqrt(2*self.mass*self.mu*B + ppar**2) # total momentum
        else:  # relativistic
            ptot = np.sqrt(gammasq_minus_1)*mc  # total momentum
        pa_old = np.arccos(ppar/ptot)
        
        # Find the new speed corresponding to the given relativistic energy
        v_new = ru.speedfromKE(ke, self.mass, unit)  # new speed
        
        # Reinitialize the guiding center
        self.__init__(pos=[x,y,z], v=v_new, pa=pa_old, t0=t,
                      mass=self.mass, charge=self.charge, field=self.field )
    
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
        
        # Calculate the current speed        
        mc = self.mass*c
        t,x,y,z,ppar = self.trajectory[-1]
        B = self.field.magB(self.trajectory[-1,:4])          
        gammasq = 1 + 2*self.mu*B/(mc*c) + (ppar/mc)**2
        if np.sqrt(gammasq)-1 < 1e-6:  # nonrelativistic
            v = np.sqrt(2*self.mass*self.mu*B + ppar**2) / self.mass
        else:  # relativistic
            v = c * np.sqrt(1-1/gammasq)
        
        # Reinitialize the guiding center
        self.__init__(pos = [x,y,z], v=v, pa=pa, t0=t, 
                      mass=self.mass, charge=self.charge, field=self.field)

    def isadiabatic(self):
        """
        Check if the motion is adiabatic at the current location.
        
        The adiabaticity condition is defined as
        
        .. math::

            \rho_c / L < \epsilon_s

            \tau_c / T < \epsilon_t
        
        where :math: `\rho_c` is the cyclotron radius, :math: `\tau_c` is the 
        cyclotron period, L is the field length scale, T is the field time scale,
        and :math: `\epsilon_s, \epsilon_t` are thresholds for adiabaticity.
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
    
    def _TaoChanBrizardEOM(self, t, Y):
        # Phase-space preserving guiding-center EOM,
        # valid with nonzero electric fields and and/or time-varying fields.
        # Also suitable for use with static, purely magnetic fields.
        # Reference: Tao, X., A. A. Chan, and A. J. Brizard (2007),
        # Hamiltonian theory of adiabatic motion of relativistic charged particles,
        # Phys. Plasmas, 14, 092107, doi:10.1063/1.2773702
        tpos = np.concatenate(([t],Y[:3]))
        ppar = Y[3]
        B = self.field.B(tpos)
        Bmag = np.sqrt(np.dot(B,B))
        unitb = B / Bmag
        gamma = np.sqrt(1 + 2*self.mu*Bmag/(self.mass*c*c) + (ppar/(self.mass*c))**2)
        cb = self.field.curlb(tpos)
        Bstar = B + ppar * cb / self.charge
        Bstarpar = np.dot(Bstar,unitb)
        E = self.field.E(tpos)
        dbdt = self.field.dbdt(tpos)
        gB = self.field.gradB(tpos)
        Estar = E - (ppar*dbdt + self.mu * gB / gamma)/self.charge
        
        retval = np.ones(4) 
        retval[:3] = (ppar * Bstar / (gamma*self.mass) + np.cross(Estar,unitb) ) / Bstarpar
        retval[3] = self.charge*np.dot(Estar,Bstar) / Bstarpar
        if params["enforce equatorial"]:
            retval[2] = retval[3] = 0
        return retval
    
    def _BrizardChanEOM(self, t, Y):
        # Phase-space preserving guiding-center EOM.
        # Valid only under static magnetic fields.
        # Special case of TaoChanBrizardEOM when E=0, dB/dt=0.
        # Reference: A. J. Brizard and A. A. Chan, 
        # Nonlinear relativistic gyrokinetic Vlasov-Maxwell equations, 
        # Phys. Plasmas 6, 4548 (1999)
        gamma = 1.0/np.sqrt(1 - (self.v/c)**2)
        tpos = np.concatenate(([t],Y[:3]))
        ppar = Y[3]
        B = self.field.B(tpos)
        Bmag = np.sqrt(np.dot(B,B))
        unitb = B / Bmag
        gB = self.field.gradB(tpos)
        cb = self.field.curlb(tpos)
        Bstar = B + ppar * cb / self.charge
        Bstarpar = np.dot(Bstar, unitb)
        retval = np.ones(4) 
        retval[:3] = (ppar * Bstar / (gamma*self.mass)  + self.mu * np.cross(unitb, gB) / (self.charge * gamma) ) / Bstarpar
        retval[3] = -self.mu * np.dot(Bstar, gB) / (gamma * Bstarpar)
        if params["enforce equatorial"]:
            retval[2] = retval[3] = 0
        return retval
        
    def _NorthropTellerEOM(self,t,Y):
        gamma = 1.0/np.sqrt(1 - (self.v/c)**2)
        gm = gamma * self.mass
        tpos = np.concatenate(([t],Y[:3]))
        ppar = Y[3]
        Bvec = self.field.B(tpos)
        B = np.sqrt(np.dot(Bvec,Bvec))
        bdir = Bvec / B
        gB = self.field.gradB(tpos)
        retval = np.ones(4)
        retval[:3] = (gm*self.v**2 + ppar**2/gm)/(2*self.charge*B**2) * np.cross(bdir,gB) + ppar*bdir/gm
        retval[3] = -self.mu * np.dot(bdir, gB) / gamma
        if params["enforce equatorial"]:
            retval[2] = retval[3] = 0
        return retval
        
    def advance(self, delta, eom="TaoChanBrizardEOM"):
        """
        Advance the particle position and parallel momentum for a given duration.
        
        The trajectory is initialized at the latest state of the `GuidingCenter`
        and integrated for an additional `delta` seconds. Uses the 
        `scipy.integrate.ode` class with `"dopri5"` solver.
        
        This method can be called many times.
        
        Parameters
        ----------
        delta : float
            The number of seconds to advance the trajectory.
        eom : {'TaoChanBrizardEOM', 'BrizardChanEOM, 'NorthropTellerEOM'}
            The equation of motion that is solved.
        
        Raises
        -------
        NonAdiabatic
            Only if the `check_adiabaticity` attribute is set to True.
            
        Notes
        -----
        Assuming that the magnetic moment is a constant of motion, and averaging
        out the gyrophase, reduces the number of dynamic variables to 4.
        
        Despite their prevalence in textbooks, Northrop-Teller equations are
        not accurate enough. The equations by Tao, Chan & Brizard are designed
        to preserve the phase-space volume, and give more accurate results than
        Northrop-Teller equations.
        
        Neither Northrop-Teller (as implemented here), nor the Brizard-Chan
        equations are suitable to use with electric fields or with time-varying
        magnetic fields. Tao-Chan-Brizard equations must be used.
        
        Tao-Chan-Brizard EOM (the default setting) reduces to Brizard-Chan EOM 
        under static conditions, with little computational overhead. 
        
        The explicit runge-kutta method of order 4(5) due to Dormand & Prince
        with stepsize control is used to solve for the motion. The relative 
        tolerance `rtol` and the absolute tolerance `atol` of the solver can be
        set with `rapt.params["solvertolerances"] = (rtol, atol)`

        """
        # Set resolution of the trajectory
        if params["GCtimestep"] != 0:
            dt = params["GCtimestep"]
        else:
            dt = self.bounceperiod()/params["bounceresolution"]
        t0 = self.trajectory[-1,0]
        rtol, atol = params["solvertolerances"]
        deriv = eval("self._"+eom)
        r = ode(deriv).set_integrator("dopri5",rtol=rtol, atol=atol)
        r.set_initial_value(self.trajectory[-1,1:], self.trajectory[-1,0])
        while r.successful() and r.t < t0+delta:
            r.integrate(r.t+dt)        
            nextpt = np.hstack(([r.t],r.y))
            self.trajectory = np.vstack((self.trajectory,nextpt))
            self.tcur = r.t
            if self.check_adiabaticity and not self.isadiabatic():
                raise NonAdiabatic

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
    def getpp(self):
        """Return a 1-d array of parallel momentum values along the trajectory."""
        return self.trajectory[:,4]
    def getr(self):
        """Return a 1-d array of radial distance along the trajectory."""
        return np.sqrt( self.getx()**2 + self.gety()**2 + self.getz()**2 )
    def gettheta(self):
        """Return a 1-d array of azimuthal angle coordinate (radians) along the
        trajectory."""
        return np.arctan2( self.gety(), self.getx() )
    def getphi(self):  # Azimuthal angle
        """Return a 1-d array of polar angle coordinate (radians) along the 
        trajectory."""
        return np.arccos( self.getz() / self.getr() )
    def getB(self):
        """Return a 1-d array of total magnetic field strength values along the 
        trajectory."""
        out = np.zeros(len(self.trajectory))
        for i, row in enumerate(self.trajectory):
            out[i] = self.field.magB(row[:4])
        return out
    def getp(self):
        """Return a 1-d array of total momentum values along the trajectory."""
        two_m_mu = 2*self.mass*self.mu
        mc = self.mass * c        
        gamma = self.getgamma()
        B = self.getB()
        pp = self.trajectory[:,4]
        assert B.shape[0] == pp.shape[0] == gamma.shape[0]
        res = np.zeros(gamma.shape[0])
        # Try to vectorize this later.
        for i,g in enumerate(gamma):
            if g-1 < 1e-6: #nonrelativistic
                res[i] = np.sqrt(two_m_mu*B[i] + pp[i]**2)
            else:  # relativistic
                res[i] = mc*(g**2-1)
        return res
        
    def getv(self):
        """Return a 1-d array of particle speed values along the guiding-center trajectories."""
        mom = self.getp()
        gamma = self.gamma()
        assert mom.shape[0] == gamma.shape[0]
        return mom/gamma
        
    def cycrad(self):
        """Return the cyclotron radius value at the final position."""
        t, r, pp = self.trajectory[-1, 0], self.trajectory[-1, 1:4], self.trajectory[-1, 4]
        # The advancer stores parallel momentum at last element. Convert to parallel speed.
        Bmag = self.field.magB(self.trajectory[-1,:4])
        gamma = np.sqrt(1 + 2*self.mu*Bmag/(self.mass*c*c) + (pp/self.mass/c)**2)
        if gamma-1 < 1e-6:  # nonrelativistic
            vp = pp/self.mass
            v = np.sqrt(2*self.mu*Bmag/self.mass + vp**2)
        else:
            vp = pp / self.mass / gamma # parallel speed
            v = c * np.sqrt(1-1/gamma**2)
        return ru.cyclotron_radius2(t, r, vp, v, self.field, self.mass, self.charge)

    def cycper(self):
        """Return the cyclotron period value at the final position."""
        t, r, pp = self.trajectory[-1, 0], self.trajectory[-1, 1:4], self.trajectory[-1, 4]
        Bmag = self.field.magB(self.trajectory[-1,:4])
        gamma = np.sqrt(1 + 2*self.mu*Bmag/(self.mass*c*c) + pp**2)
        if gamma - 1 < 1e-6:  # nonrelativistic
            vp = pp/self.mass
            v = np.sqrt(2*self.mu*Bmag/self.mass + vp**2)
        else:
            v = c * np.sqrt(1-1/gamma**2)
        return ru.cyclotron_period2(t, r, v, self.field, self.mass, self.charge)

    def getgamma(self):
        """Return a 1-d array of relativistic factor values along the trajectory.
        
        The relativistic factor is defined as :math:`\gamma = \sqrt{1 + 2\mu B / (mc^2) + (p_{||}/mc)^2}`
        """
        mc = self.mass*c
        mc2 = mc*c
        pp = self.trajectory[:,4]
        B = self.getB()
        return np.sqrt(1 + 2*self.mu*B/mc2 + (pp/mc)**2)
    
    def getBm(self):
        """Returns an array of mirror field values at each trajectory point.
        
        Defined using the conservation of magnetic moment.
        """
        mc = self.mass*c        
        n = self.trajectory.shape[0]
        gamma = self.getgamma()
        B = self.getB()
        pp = self.trajectory[:,4]
        assert B.shape[0] == pp.shape[0] == gamma.shape[0] == n
        res = np.zeros(n)
        # Try to vectorize this later.
        for i,g in enumerate(gamma):
            if g-1 < 1e-6: #nonrelativistic
                res[i] = (B[i] + 0.5*pp[i]**2/(self.mu*self.mass))
            else:  # relativistic
                res[i] = B[i]/(1 - (pp[i]/mc)**2/((g-1)*(g+1)))
        return res
        
    def getke(self):
        """Return a 1-d array of kinetic energy values (Joule) along the
        trajectory."""
        mc = self.mass*c
        mc2 = mc*c
        n = self.trajectory.shape[0]
        gamma = self.getgamma()
        B = self.getB()
        pp = self.trajectory[:,4]
        assert gamma.shape[0] == B.shape[0] == pp.shape[0] == n
        res = np.zeros(n)
        # Try to vectorize this later.
        for i,g in enumerate(gamma):
            if g-1 < 1e-6: #nonrelativistic
                res[i] = self.mu*B[i] + 0.5*pp[i]**2/self.mass
            else:  # relativistic
                res[i] = (g-1)*mc2
        return res
    
    def bounceperiod(self):
        """Return the bounce period at the current position."""
        tpos, ppar = self.trajectory[-1, 0:4], self.trajectory[-1, 4]
        Bmag = self.field.magB(tpos)
        gamma = np.sqrt(1 + 2*self.mu*Bmag/(self.mass*c*c) + (ppar/(self.mass*c))**2)
        if gamma-1 < 1e-6: #nonrelativistic
            p = np.sqrt(2*self.mass*self.mu*Bmag + ppar**2)  # total momentum
            v = p/self.mass
            Bmirror = (p**2) / (2*self.mass*self.mu)
        else:  # relativistic
            p = self.mass * c *np.sqrt((gamma+1)*(gamma-1))  # momentum
            Bmirror = p**2 / ((p-ppar)*(p+ppar)) * Bmag
            v = p/self.mass/gamma
        return rfu.bounceperiod(tpos,self.field,Bmirror,v)
    
    def geteye(self,step=1):
        """Return a 1-d array of (time,second invariant) pairs along the 
        guiding-center trajectory.
        
        Parameters
        ----------
        step : int, optional
            Take every `step` values in the trajectory.
        """
        
        Bmirror = self.getBm()
        assert Bmirror.shape[0] == self.trajectory.shape[0]
        res = np.zeros(self.trajectory[::step].shape[0])
        for i,(row,Bm) in enumerate(zip(self.trajectory[::step], Bmirror[::step])):    
            res[i] = rfu.eye(row[:4], self.field, Bm)
        # zip together with time values
        return np.array(list(zip(self.trajectory[::step,0], res)))
