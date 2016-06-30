# -*- coding: utf-8 -*-
"""
RAPT utility functions

AUTHOR:
    Kaan Ozturk <mkozturk@yahoo.com>
    
    2016, Rice University
"""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from rapt import c,e,B0,Re

def gamma(mass=0, **kwargs):
    """Return the relativistic factor for given velocity or momentum.
    
    Usage: gamma(velocity=[1,1,2]), or gamma(momentum=[5,5,10], mass=5)."""
    if "velocity" in kwargs:
        return 1/np.sqrt(1-np.dot(kwargs["velocity"],kwargs["velocity"])/c**2)
    elif "momentum" in kwargs:
        if mass==0:
            raise ValueError("Particle mass not given.")
        else:
            return np.sqrt(1 + np.dot(kwargs["momentum"], kwargs["momentum"])/(mass*c)**2)
    else:
        raise ValueError("Either velocity of momentum vectors should be given.")

def cyclotron_period(t, pos, vel, field, mass, charge):
    """
    Return the period of the cyclotron motion of a particle.
    
    Parameters
    ----------
    t : float
        Time (seconds)
    pos : array-like
        Position array (x,y,z) (meters)
    vel : array-like
        Velocity array (vx, vy, vz) (m/s)
    field: Field object
        The field object that provides electric and magnetic field vectors and 
        related quantities.
    mass: float
        The mass of the particle, in kg.
    charge: float
        The charge of the particle, in Coulombs.
    
    Returns
    -------
    float
        The cyclotron period :math: `\frac{2\pi\gamma m}{qB}`
    
    See Also
    --------
    rapt.utils.cyclotron_period2

    Notes
    -----
    Works with the velocity vector. Suitable to use with `Particle` data.    
    """
    # Use with Particle.
    vel = np.array(vel)
    gamma = 1.0/np.sqrt(1 - np.dot(vel,vel)/c**2)
    B = field.magB(np.concatenate([[t], pos]))
    return 2*np.pi*gamma*mass/B/abs(charge)

def cyclotron_period2(t, pos, speed, field, mass, charge):
    """
    Return the period of the cyclotron motion of a guiding center.
    
    Parameters
    ----------
    t : float
        Time (seconds)
    pos : array-like
        Position array (x,y,z) (meters)
    speed : float
        The speed of the particle (m/s)
    field: Field object
        The field object that provides electric and magnetic field vectors and 
        related quantities.
    mass: float
        The mass of the particle, in kg.
    charge: float
        The charge of the particle, in Coulombs.
    
    Returns
    -------
    float
        The cyclotron period :math: `\frac{2\pi\gamma m}{qB}`
    
    See Also
    --------
    rapt.utils.cyclotron_period

    Notes
    -----    
    Works with the total speed. Suitable to call with `GuidingCenter` data.
    """
    # Use with GuidingCenter.
    gamma = 1.0/np.sqrt(1 - (speed/c)**2)
    B = field.magB(np.concatenate([[t], pos]))
    return 2*np.pi*gamma*mass/B/abs(charge)

def cyclotron_radius(t,pos,vel,field,mass,charge):
    """
    Return the radius of the cyclotron motion of a particle.
    
    Parameters
    ----------
    t : float
        Time (seconds)
    pos : array-like
        Position array (x,y,z) (meters)
    vel : array-like
        Velocity array (vx, vy, vz) (m/s)
    field: Field object
        The field object that provides electric and magnetic field vectors and 
        related quantities.
    mass: float
        The mass of the particle, in kg.
    charge: float
        The charge of the particle, in Coulombs.
    
    Returns
    -------
    float
        The cyclotron radius :math: `\frac{\gamma m v_\perp}{qB}`
    
    See Also
    --------
    rapt.utils.cyclotron_radius2

    Notes
    -----
    Works with the velocity vector. Suitable to use with `Particle` data.    
    """
    vsq = np.dot(vel, vel)
    gamma = 1.0/np.sqrt(1 - vsq/c**2)
    B = field.B(np.concatenate([[t], pos]))
    Bmag = np.sqrt(np.dot(B,B))
    vpar = np.dot(vel,B)/Bmag
    vperp = np.sqrt(vsq-vpar**2)
    return gamma*mass*vperp/(abs(charge)*Bmag)

def cyclotron_radius2(t, pos, vpar, v, field, mass, charge):
    """
    Return the radius of the cyclotron motion of a particle.
    
    Parameters
    ----------
    t : float
        Time (seconds)
    pos : array-like
        Position array (x,y,z) (meters)
    vpar : float
        Parallel speed (m/s)
    v : float
        Total speed (m/s)
    field: Field object
        The field object that provides electric and magnetic field vectors and 
        related quantities.
    mass: float
        The mass of the particle, in kg.
    charge: float
        The charge of the particle, in Coulombs.
    
    Returns
    -------
    float
        The cyclotron radius :math: `\frac{\gamma m v_\perp}{qB}`
    
    See Also
    --------
    rapt.utils.cyclotron_radius
    
    Notes
    -----    
    Works with the total speed and the parallel speed. Suitable to call with 
    `GuidingCenter` data.
    """
    gamma = 1.0/np.sqrt(1 - (v/c)**2)
    B = field.B(np.concatenate([[t], pos]))
    Bmag = np.sqrt(np.dot(B,B))
    vperp = np.sqrt((v-vpar)*(v+vpar))
    return gamma*mass*vperp/(abs(charge)*Bmag)

def magnetic_moment(t, pos, vpar, v, field, mass):
    """Return the magnetic moment of the given guiding center.

    Parameters
    ----------
    t : float
        Time (seconds)
    pos : array-like
        Position array (x,y,z) (meters)
    vpar : float
        Parallel speed (m/s)
    v : float
        Total speed (m/s)
    field: Field object
        The field object that provides electric and magnetic field vectors and 
        related quantities.
    mass: float
        The mass of the particle, in kg.
    
    Returns
    -------
    float
        The magnetic moment :math: `\frac{\gamma^2 m v_\perp^2}{2B}`

    """
    gamma = 1.0/np.sqrt(1 - (v/c)**2)
    Bmag = field.magB(np.concatenate([[t], pos]))
    return gamma**2*mass*(v-vpar)*(v+vpar)/(2*Bmag) # magnetic moment

def speedfromKE(KE, mass, unit="ev"):
    """
    Return the speed of a particle with given relativistic kinetic energy and mass.

    Parameters
    ----------
    KE : float
        The kinetic energy of the particle.
    mass : float
        The mass of the particle (kg).
    unit : str (optional)
        The unit of the energy. If "ev" (default), energy is in electron volts,
        otherwise in Joules
    
    Returns
    -------
    float
        The speed (m/s) corresponding to the given kinetic energy.
    
    Notes
    -----
    If the energy is less than a millionth of the rest energy (:math: `mc^2`),
    uses the nonrelativistic formula :math: `K=mv^2/2`; otherwise uses the
    relativistic formula `K = (\gamma - 1)mc^2`
    """
    mc2 = mass * c**2    
    if unit.lower()=="ev":
        KE = KE*e # convert to Joules
    if KE/mc2 < 1e-6: # nonrelativistic
        return np.sqrt(2*KE/mass)
    else:  # relativistic
        return c * np.sqrt(1-(mc2/(mc2+KE))**2)

def guidingcenter(t, r, v, field, mass, charge, tol=1e-3, maxiter=20, debug=False):
    """
    Return the guiding center for the given particle state.
    
    Parameters
    ----------
    t : float
        Time (seconds)
    r : array-like
        Position array (x,y,z) (meters)
    v : array-like
        Velocity array (vx, vy, vz) (m/s)
    field: Field object
        The field object that provides electric and magnetic field vectors and 
        related quantities.
    mass: float
        The mass of the particle, in kg.
    charge: float
        The charge of the particle, in Coulombs.
    tol : float, optional
        The error tolerance (relative change in the last iteration).
    maxiter : int, optional
        Maximum number of iterations.
    debug : bool
        If True, list of intermediate iteration positions is returned.
    
    Returns
    -------
    array
        Position (x,y,z) of the guiding center.
    float
        Parallel speed
    float
        Total speed
    
    Notes
    -----
    Given the particle position r and velocity v, the guiding center position R
    is defined as :math: `R = r - \rho(R)`. This function solves this vector
    equation iteratively for R.
    
    This equation is actually a truncation of the series at first order, so the
    result will not be exactly equal to the actual guiding center, even if the 
    tolerance is set to small values. A more accurate expression is available,
    but given other error sources, that level of accuracy is not needed.
    """ 

    def gyrovector(r):
        vsq = np.dot(v,v)
        gamma = 1/np.sqrt(1-vsq/c**2)
        B = field.B(np.concatenate([[t], r]))
        Bsq = np.dot(B,B)
        return gamma*mass/(charge*Bsq) * np.cross(B,v)
        
    def norm(v):
        return np.sqrt(np.dot(v,v))
    
    # Solve the vector equation R = r - rho(R) for R by iteration
    GC_old = np.array(r) - gyrovector(r)  # Initial estimate for R is r
    if debug:
        GC_list = [GC_old]
    iteration = 1
    while iteration <= maxiter:
        GC = r - gyrovector(GC_old)
        if debug:
            GC_list.append(GC)
        if norm(GC-GC_old)/norm(GC) < tol:  # converged to the solution
            if debug:
                return GC_list
            else:
                B = field.B(np.concatenate([[t], GC]))
                vp = np.dot(v,B)/norm(B)
                return GC, vp, norm(v)
        GC_old = GC
        iteration += 1
    print("Could not reach the specified tolerance after ", iteration, " iterations.")

def getperp(v):
    """Return a unit vector perpendicular to v.

    Parameters
    ----------
    v : array-like
        The reference vector in 3d-space.
    
    Returns
    -------
    array
        A vector perpendicular to the given parameter.
    
    Raises
    ------
    ValueError
        If the input vector is zero.
    
    Notes
    -----
    There are infinitely many vectors (a,b,c), perpendicular to a given vector 
    (x,y,z), satisfying :math: `a x + b y + c z = 0`. Here, the choices are as 
    follows:
    
     - if x == 0, return [1,0,0]
     - else if y == 0, return [0,1,0]
     - else if z == 0, return [0,0,1]
     - else return [1, 1, -(x+y)/z] (arbitrarily set a=b=1)
    """
    # For the input vector (x,y,z) and output vector (a,b,c)
    # the equation ax + by + cz = 0 must be satisfied.
    assert(len(v)==3)
    if v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0:
        raise ValueError('Zero vector')

    if v[0]==0:
        return np.array([1,0,0])
    if v[1]==0:
        return np.array([0,1,0])
    if v[2]==0:
        return np.array([0,0,1])
    
    # arbitrarily set a = b = 1
    # then the equation simplifies to
    # c = -(x + y)/z
    cc = -1.0 * (v[0] + v[1]) / v[2]
    return np.array([1, 1, cc])/np.sqrt(2+cc**2)
        
def GCtoFP(t, R, vp, speed, field, mass, charge, gyrophase=0):
    """
    Return a Particle position for the given guiding center position.

    Parameters
    ----------
    t : float
        The time at which to evaluate the field.
    R : array-like
        The guiding-center coordinate vector.
    vp : float
        The parallel speed.
    speed : float
        The total speed of the particle.
    field: Field object
        The field object that provides electric and magnetic field vectors and 
        related quantities.
    mass: float
        The mass of the particle, in kg.
    charge: float
        The charge of the particle, in Coulombs.
    gyrophase : float, optional
        The arbitrary cyclotron motion phase, in radians.
    
    Returns
    -------
    pos : array-like
        The position vector of the particle
    vel : array-like
        The velocity vector of the particle
    
    See Also
    --------
    getperp
    
    Notes
    -----
    Determines the particle position by constructing a vector that is
    perpendicular to the magnetic field vector at the guiding-center position.
    The angle is arbitrary for practical purposes, but deterministic. See the
    Notes on the `getperp` function.
    
    The user can set an angle with respect to that fixed direction using the 
    `gyrophase` parameter, to experiment with the effects of different phase
    angles, if necessary.
    """
    B = field.B(np.concatenate([[t], R]))
    Bsq = np.dot(B,B)
    b = B / np.sqrt(Bsq)  # unit vector in the field direction
    pa = np.arccos(vp/speed)  # pitch angle
    rc = cyclotron_radius2(t,R,vp,speed,field,mass,charge)
    u = getperp(B) # first reference vector
    u = u / np.sqrt(np.dot(u,u))   # make unit vector
    w = np.cross(b,u)  # second reference vector
    s = np.sign(charge)
    pos = R + rc*(np.cos(gyrophase)*u + np.sin(gyrophase)*w)
    vel = speed * (np.cos(pa)*b + s*np.sin(pa)*np.sin(gyrophase)*u - s*np.sin(pa)*np.cos(gyrophase)*w)
    return pos, vel

def dipole_drift_period_p(L, v, eqpa):
    """
    Return the approximate drift period of a proton in dipole field.
    
    Parameters
    ----------
    L : float
        Initial equatorial distance in units of Re.
    v: float
        Particle speed.
    eqpa : float
        Equatorial pitch angle in degrees.
    
    Returns
    -------
    float
        The drift period of a proton under the dipole field.
    
    References
    ----------
    Martin Walt, 'Introduction to Geomagnetically Trapped Radiation', p.49
    """
    b = v/c
    gamma = 1/np.sqrt(1-b**2)
    eqpa = eqpa *np.pi/180
    return 8.481 / (L*gamma*b**2) * (1 - (1/3)*(np.sin(eqpa)**0.62))

def dipole_drift_period_e(L, v, eqpa):
    """
    Return the approximate drift period of a electron in dipole field.
    
    Parameters
    ----------
    L : float
        Initial equatorial distance in units of Re.
    v: float
        Particle speed.
    eqpa : float
        Equatorial pitch angle in degrees.
    
    Returns
    -------
    float
        The drift period of a electron under the dipole field.
    
    References
    ----------
    Martin Walt, 'Introduction to Geomagnetically Trapped Radiation', p.49
    """
    b = v/c
    gamma = 1/np.sqrt(1-b**2)
    eqpa = eqpa *np.pi/180
    return 1.557e4 /(L*gamma*b**2) * (1 - 1/3 * (np.sin(eqpa)**0.62))

def dipole_bounce_period(L, v, eqpa):
    """
    Return the approximate bounce period of a particle in dipole field.
    
    Parameters
    ----------
    L : float
        Initial equatorial distance in units of Re.
    v: float
        Particle speed.
    eqpa : float
        Equatorial pitch angle in degrees.
    
    Returns
    -------
    float
        The bounce period of a particle (proton or electron) under dipole.
    
    References
    ----------
    Martin Walt, 'Introduction to Geomagnetically Trapped Radiation', p.49
    """
    b = v/c
    eqpa = eqpa *np.pi/180
    return 0.117 * L/b * (1 - 0.4635 * (np.sin(eqpa)**0.75))

# yanlış bu, düzelt
def Bm(tpos, pa, field):
    """Returns the mirror field for a particle. Assumes conservation of the first invariant.
    tpos: [time,x,y,z]
    pa: pitch angle in degrees
    field: the field object."""
    B = field.magB(tpos)
    return B/np.sin(pa * np.pi/180)**2

def dipoleBm(L,eqpa):
    """Returns the mirror field for a particle in dipole field. Assumes conservation of the first invariant.
    L: Equatorial distance in Re.
    eqpa: Equatorial pitch angle."""
    B = B0/L**3
    return B/np.sin(eqpa * np.pi/180)**2

def dipoleI(R0, Bm):
    """
    Return the second invariant value in dipole field.
    
    Parameters
    ----------
    R0 : float
        Equatorial distance to the guiding line (meters)
    Bm : float
        Mirror field strength. (Tesla)
    
    Returns
    -------
    float
        The second invariant :math: `I = \int_{s_{m1}}^{s_{m2}\sqrt{1 - B(s)/B_m}ds`
    
    """
    # find the northern mirror point
    C = 1 / ((Bm/B0)**2 * (R0/Re)**6)
    y_m = brentq(lambda y: y**6 - C*(4-3*y), 0, 1)
    theta_m1 = np.arcsin(np.sqrt(y_m))
    theta_m2 = np.pi - theta_m1
    
    # Evaluate the integral with the substitution x = cos(theta)
    K = np.sqrt(C)
    integrand = lambda x: np.sqrt( (1-K*np.sqrt(1+3*x**2)/(1-x**2)**3) * (1+3*x**2) )
    return R0 * quad( integrand, np.cos(theta_m2), np.cos(theta_m1) )[0] # N.B: integration limits are switched on purpose, due to a minus sign.

def dipole_gradI(x,y, Bm, h):
    """
    Return the gradient of the second invariant in dipole field.
    
    Parameters
    ----------
    x,y : float
        Equatorial position coordinates (meters)
    Bm : float
        Mirror field strength. (Tesla)
    h : float
        Step size for numerical differencing
    
    Returns
    -------
    array
        The gradient of the second invariant.
    
    """
    r0 = np.sqrt(x**2+y**2)
    f = lambda x: dipoleI(r0+x, Bm)
    mag = (f(h) - f(-h))/(2*h)
    return mag * np.array((x,y,0))/r0

def drift_error(b):
    """
    Return a vector of relative error of the bounce-averaged drift path
    compared to a circle (the expected drift path under a dipole)
    
    Parameters
    ----------
    b : BounceCenter object
    
    Returns
    -------
    array
        Array of relative error values at each trajectory point.
    """
    
    r0 = np.sqrt(np.dot(b.trajectory[0,1:],b.trajectory[0,1:]))
    e = (np.sqrt(
    np.diag(
    np.dot(
    b.trajectory[:,1:],np.transpose(b.trajectory[:,1:])
    )
    )
    )
    - r0) / r0
    return e