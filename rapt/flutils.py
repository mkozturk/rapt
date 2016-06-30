# -*- coding: utf-8 -*-
"""
RAPT field line utility functions

AUTHOR:
    Kaan Ozturk <mkozturk@yahoo.com>
    
    2016, Rice University
"""
import numpy as np

from rapt import params
from rapt.utils import dipoleI, dipole_gradI
from rapt.fieldline import Fieldline

from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize
from scipy.integrate import quad,simps
from scipy.misc import derivative

def criticalpoints(tpos, field, Bm):
    """
    Return a list of critical points where the field strength is a maximum or a minimum along the field line.
    
    Parameters
    ----------
    tpos : array-like
        Starting position and time for the field line.
    field : Field
        The field object that provides electric and magnetic field vectors and 
        related quantities.
    Bm : float
        The mirror field strength in Tesla (defines the endpoints of the field line).
    
    Returns
    -------
    array
        Array of (s, B, B'') triplets for each critical point, where s is the
        location along the field line (with respect to the initial position),
        B is the field strength at s, and B'' is the second derivative along
        the field line.
    
    """
    fl = Fieldline(tpos, field, Bmax = Bm)
    fl.trace()
    s = fl.gets()
    b = fl.getB()
    n = len(b)
    slist = []
    # first, make a list of approximate locations of minima or maxima.
    for i in range(1,n-1):
        if (b[i]-b[i-1])*(b[i+1]-b[i]) <= 0:
            slist.append(s[i])
    # Create an interpolated function B(s)
    B = interp1d(s, b, kind='cubic', assume_sorted=True)
    # Find the optimal point for each value in slist
    res = []    
    for s0 in slist:
        m = minimize(B, s0)
        sm = m.x[0]
        B2s = derivative(B, sm, dx=0.1*(s[1]-s[0]), n=2)
        res.append((sm, float(B(sm)), B2s))
    return res

def eye(tpos, field, Bm, usedipole=False):
    """
    Return the I integral (second invariant) value.
    
    Parameters
    ----------
    tpos : array-like
        4-element vector of time (s) and starting position of the field line
        (meters).
    Bm : float
        Mirror field value in Tesla.
    field: Field object
        The field object under which the integral is evaluated.
    
    Returns
    -------
    float
        The second invariant I, evaluated over the field line passing through
        the starting point at the given time.
        
    Other Parameters
    ----------------
    usedipole : bool, optional
        If True, uses the formula for dipole, without following a field line.
    """
    eqpa_threshold = 70 # Below this value, use Simpson's rule. Otherwise use interpolation.
    
    # Return a more accurate form for I if field is a dipole.    
    if usedipole:
        R0 = np.sqrt(np.dot(tpos[1:3],tpos[1:3]))
        return dipoleI(R0,Bm)
    
    # Follow a field line from the starting point up to Bm.
    # The trace will slightly overshoot the mirror point.
    fl = Fieldline(tpos, field, Bmax = Bm)
    fl.trace()
    s = fl.gets()
    b = fl.getB()
    n = len(b)

    Bmin = np.min(b)   
    
    if Bmin > Bm:
        return 0
    if abs(Bmin-Bm)/Bm < 1e-12:  # equatorial particle
        return 0
    # Determine the indices of points that satisfy B<Bm
    inside = np.where(b<Bm)[0]
    # Discard points so that only one point lies beyond the mirror locations.
    b = np.delete(b, list(range(0,inside[0]-1)) + list(range(inside[-1]+2, n)))
    s = np.delete(s, list(range(0,inside[0]-1)) + list(range(inside[-1]+2, n)))
    assert b[0]>Bm and b[1]<Bm and b[-2]<Bm and b[-1]>Bm

    eqpa = np.arcsin(np.sqrt(Bmin/Bm)) * 180 / np.pi   # equatorial pitch angle (estimated)
    if eqpa < eqpa_threshold: # use Simpson's rule to evaluate the integral
        # Determine the lower (antiparallel direction) mirror point location
        sm1 = (Bm-b[0]) * (s[1]-s[0]) / (b[1]-b[0]) + s[0]
        # Determine the upper (parallel direction) mirror point location
        sm2 = (Bm-b[-2]) * (s[-1]-s[-2]) / (b[-1]-b[-2]) + s[-2]
                    
        s[0], s[-1] = sm1, sm2
        b[0], b[-1] = Bm, Bm

        #Evaluate all but the end intervals with Simpson's rule
        bi = np.sqrt(1-b[1:-1]/Bm)
        eye = simps(bi,s[1:-1])
        # Add the end intervals with special formula
        d = s[-1]-s[-2]
        eye += (2/3)*d*np.sqrt((Bm-b[-2])/Bm)
        d = s[1]-s[0]
        eye += (2/3)*d*np.sqrt((Bm-b[1])/Bm)

    else:  # use interpolation and adaptive quadrature for integration
        # Generate an interpolating function for B(s)
        B = interp1d(s, b, kind='quadratic', assume_sorted=True)
        # Determine the lower (antiparallel direction) mirror point location
        # with root finding.
        sm1 = brentq(lambda x: B(x)-Bm, s[0],s[1])
          
        # Determine the upper (parallel direction) mirror point location
        if B(s[-2])==Bm:
            sm2 = s[-2]
        else:  
            sm2 = brentq(lambda x: B(x)-Bm, s[-2],s[-1])        
        eye = quad(lambda x: np.sqrt(1-B(x)/Bm), sm1, sm2, epsrel=1e-4)[0]

    return eye

def gradI(tpos, field, Bm, usedipole=False):
    """
    Return the gradient of the second invariant at the given location.
    
    Parameters
    ----------
    tpos : array-like
        4-element vector of time (s) and starting position of the field line
        (meters).
    field: Field object
        The field object under which the integral is evaluated.
    Bm : float
        Mirror field value in Tesla.
    
    Returns
    -------
    array
        The gradient vector of the second invariant I.
        
    Other Parameters
    ----------------
    usedipole : bool, optional
        If True, uses the formula for dipole, without following a field line.
    
    Notes
    -----
    The derivatives are calculated using central differences in two directions
    that are perpendicular to the field vector. The step size is set with the
    ``rapt.params["eyegradientstep"]`` parameter. If this value is too small,
    grad I becomes to jagged and the `BounceCenter` solver does not converge.
    
    If the particle has a very small bounce amplitude, one of the points in the
    central difference may have a field stronger than the mirror field, giving
    I = 0. In this case, forward differencing is used.
    """

    tpos = np.array(tpos)
    x,y = tpos[1],tpos[2]
    d = params['eyegradientstep']

    r = np.array((x,y,0))/max((abs(x),abs(y))) # scale down to reduce arithmetic errors. Will be normalized later.
    b = field.unitb(tpos)
    r = r - np.dot(r,b)*b # make sure r is perpendicular to b.
    r = r / np.sqrt(np.dot(r,r)) # make r a unit vector
    
    v1 = np.zeros(4)
    v1[1:4] = np.copy(r[:])
    v2 = np.zeros(4)
    v2[1:4] = np.cross(b,r)
    
    if usedipole:
        return dipole_gradI(x,y,Bm,d)

    I = lambda p: eye(p, field, Bm)
    
    # Use central difference to estimate the derivative,
    # except when one I-value is zero. Use forward difference there.
    
    # Partial derivative in v1-direction
    I1 = I(tpos + d*v1)
    I2 = I(tpos - d*v1)
    if I1 == 0:
        dIdv1 = (I(tpos) - I2)/d
    elif I2 == 0:
        dIdv1 = (I1 - I(tpos))/d
    else:
        dIdv1 = (I1 - I2)/(2*d)
    # Partial derivative in v2-direction
    I1 = I(tpos + d*v2)
    I2 = I(tpos - d*v2)
    if I1 == 0:
        dIdv2 = (I(tpos) - I2)/d
    elif I2 == 0:
        dIdv2 = (I1 - I(tpos))/d
    else:
        dIdv2 = (I1 - I2)/(2*d)
    return (dIdv1*v1[1:] + dIdv2*v2[1:])
        

def bounceperiod(tpos,field,Bm,v):
    """Return the bounce period.
    
    Parameters
    ----------
    tpos : array-like
        4-element array of time and position.
    field: Field object
        The field object under which the integral is evaluated.
    Bm : float
        Mirror field value (T).
    v : float
        Particle speed (m/s).
    
    Returns
    -------
    float
        The bounce period of a particle with the specified conditions.
    """
    
    return (2/v)*halfbouncepath(tpos,field,Bm)

def halfbouncepath(tpos,field,Bm):
    """Return the half-bounce path length S_b.
    
    Parameters
    ----------
    tpos : array-like
        4-element array of time and position.
    field: Field object
        The field object under which the integral is evaluated.
    Bm : float
        Mirror field value (T).

    Returns
    -------
    float
        The half-bounce path length (m) : The total helical distance traveled by a 
        particle around the guiding line passing through the given location and
        mirroring at B=Bm.
    """
    # Follow a field line from the starting point up to Bm.
    # The trace will slightly overshoot the mirror points.
    fl = Fieldline(tpos, field, Bmax = Bm)
    fl.trace()
    
    s = fl.gets()
    b = fl.getB()
    n = len(b)
    # Determine the indices of points that satisfy B<Bm
    inside = np.where(b<=Bm)[0]
    # there are 0 or 1 such points, we have an equatorial particle.
    # Discard points so that only one point lies beyond each mirror location.
    # i1,i2 are the indices of points to keep. 
    if len(inside) == 0:
        i1 = int((n-3)/2)
        i2 = int((n+1)/2)
    else:
        i1,i2 = inside[0]-1,inside[-1]+1
    b = np.delete(b, list(range(0,i1)) + list(range(i2+1, n)))
    s = np.delete(s, list(range(0,i1)) + list(range(i2+1, n)))
    n = len(b)
    
    if n == 3: # Equatorial particle
        # Evaluate Sb = pi * sqrt(2Bm/B2s)
        # where B2s is the second derivative at the minimum.
        # Estimate it using Lagrange polynomials of degree 2.
        s1, s2, s3 = s[0], s[1], s[2]
        B1, B2, B3 = b[0], b[1], b[2]
        s12 = s1-s2
        s23 = s2-s3
        s13 = s1-s3
        B2s = 2*(B1*s23 - B2*s13 + B3*s12) / (s12*s13*s23)
        res = np.pi * np.sqrt(2*Bm/B2s)
    else:
        # Get an interpolating function for B(s)
        B = interp1d(s, b, kind='quadratic', assume_sorted=True)
        # Determine the lower (antiparallel direction) mirror point location
        # with root finding.
        sm1 = brentq(lambda x: B(x)-Bm, s[0],s[1])
        # Determine the upper (parallel direction) mirror point location
        sm2 = brentq(lambda x: B(x)-Bm, s[-2],s[-1])        
        res = quad(lambda x: 1/np.sqrt(1-B(x)/Bm), sm1, sm2, epsrel=1e-4)[0]
        
    return res

def fl_error(tpos, field, Bm):
    """
    Return the max and mean relative error of calculated dipole field line 
    points, compared to the analytic result. For debugging.

    Parameters
    ----------
    tpos : array-like
        4-element array of time and position.
    field: Field object
        The field object under which the integral is evaluated.
    Bm : float
        Mirror field value (T).
    
    Returns
    -------
    e_max : float
        The maximum error
    e_avg : float
        Th average error
    """
    fl = Fieldline(tpos, field, Bmax=Bm)
    fl.trace()
    r0 = np.sqrt(np.dot(tpos[1:],tpos[1:]))
    e_max = 0
    e_avg = 0
    for p in fl.curve:
        r = np.sqrt(np.dot(p[1:],p[1:]))
        l = np.arcsin(p[3]/r)
        e = abs(1 - r/(r0*np.cos(l)**2))
        if e > e_max:
            e_max = e
        e_avg += e
    e_avg /= (len(fl.curve)-1)
    return (e_max, e_avg)

def eye_error(tpos, field, Bm):
    """
    Return the error in I calculated from the field line compared to the
    analytic dipole form. For debugging.
    
    Parameters
    ----------
    tpos : array-like
        4-element array of time and position.
    field: Field object
        The field object under which the integral is evaluated.
    Bm : float
        Mirror field value (T).
    
    Returns
    -------
    float
        The difference between I values computed over field line and using the
        dipole formula.
    """
    
    Icalc = eye(tpos, field, Bm, usedipole=False)
    Idip = eye(tpos, field, Bm, usedipole=True)
    return Icalc-Idip