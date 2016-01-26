import numpy as np
from scipy.optimize import newton,brentq
from scipy.integrate import quad
from rapt import c,e,B0,Re,params

def cyclotron_period(t, pos, vel, field, mass, charge):
    """Returns the period of the cyclotron motion of a particle."""
    # Use with Particle.
    gamma = 1.0/np.sqrt(1 - np.dot(vel,vel)/c**2)
    B = field.magB(np.concatenate([[t], pos]))
    return 2*np.pi*gamma*mass/B/abs(charge)

def cyclotron_period2(t, pos, speed, field, mass, charge):
    """Returns the period of the cyclotron motion of a particle."""
    # Use with GuidingCenter.
    gamma = 1.0/np.sqrt(1 - (speed/c)**2)
    B = field.magB(np.concatenate([[t], pos]))
    return 2*np.pi*gamma*mass/B/abs(charge)

def cyclotron_radius(t, pos,vel,field,mass,charge):
    """Returns the period of a particle at given position and velocity."""
    vsq = np.dot(vel, vel)
    gamma = 1.0/np.sqrt(1 - vsq/c**2)
    B = field.B(np.concatenate([[t], pos]))
    Bmag = np.sqrt(np.dot(B,B))
    vpar = np.dot(vel,B)/Bmag
    vperp = np.sqrt(vsq-vpar**2)
    return gamma*mass*vperp/(abs(charge)*Bmag)

def cyclotron_radius2(t, pos, vpar, v, field, mass, charge):
    """Returns the period of a particle at given position, parallel speed, and total speed."""
    gamma = 1.0/np.sqrt(1 - (v/c)**2)
    B = field.B(np.concatenate([[t], pos]))
    Bmag = np.sqrt(np.dot(B,B))
    vperp = np.sqrt((v-vpar)*(v+vpar))
    return gamma*mass*vperp/(abs(charge)*Bmag)

def magnetic_moment(t, pos, vpar, v, field, mass):
    """Returns the magnetic moment of the given guiding center."""
    gamma = 1.0/np.sqrt(1 - (v/c)**2)
    Bmag = field.magB(np.concatenate([[t], pos]))
    return gamma**2*mass*(v-vpar)*(v+vpar)/(2*Bmag) # magnetic moment

def speedfromKE(KE, mass, unit="ev"):
    """Returns the speed of a particle with given relativistic kinetic energy and mass."""
    mc2 = mass * c**2    
    if unit.lower()=="ev":
        KE = KE*e # convert to Joules
    if KE/mc2 < 1e-6: # nonrelativistic
        return np.sqrt(2*KE/mass)
    else:  # relativistic
        return c * np.sqrt(1-(mc2/(mc2+KE))**2)

def guidingcenter(t, r, v, field, mass, charge, tol=1e-3, maxiter=20, debug=False):
    """Returns the guiding center for the given particle.""" 

    def gyrovector(r):
        vsq = np.dot(v,v)
        gamma = 1/np.sqrt(1-vsq/c**2)
        B = field.B(np.concatenate([[t], r]))
        Bsq = np.dot(B,B)
        return gamma*mass/(abs(charge)*Bsq) * np.cross(B,v)
        
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
    """Returns a unit vector perpendicular to v."""
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
    #     c = -(x + y)/z
    cc = -1.0 * (v[0] + v[1]) / v[2]
    return np.array([1, 1, cc])/np.sqrt(2+cc**2)
        
def GCtoFP(t, R, vp, speed, field, mass, charge, gyrophase):
    """Returns a particle position for the given guiding center."""
#    def getperp(v):
#        """Returns a unit vector perpendicular to v."""
#        # For the input vector (x,y,z) and output vector (a,b,c)
#        # the equation ax + by + cz = 0 must be satisfied.
#        assert(len(v)==3)
#        if v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0:
#            raise ValueError('Zero vector')
#    
#        if v[0]==0:
#            return np.array([1,0,0])
#        if v[1]==0:
#            return np.array([0,1,0])
#        if v[2]==0:
#            return np.array([0,0,1])
#        
#        # arbitrarily set a = b = 1
#        # then the equation simplifies to
#        #     c = -(x + y)/z
#        cc = -1.0 * (v[0] + v[1]) / v[2]
#        return np.array([1, 1, cc])/np.sqrt(2+cc**2)

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
    """Returns the approximate drift period of a proton in dipole field.
    L: Initial equatorial distance in units of Re
    v: speed
    eqpa: equatorial pitch angle in degrees
    Ref.: Martin Walt, 'Introduction to Geomagnetically Trapped Radiation', p.49"""
    b = v/c
    gamma = 1/np.sqrt(1-b**2)
    eqpa = eqpa *np.pi/180
    return 8.481 / (L*gamma*b**2) * (1 - (1/3)*(np.sin(eqpa)**0.62))

def dipole_drift_period_e(L, v, eqpa):
    """Returns the approximate drift period of an electron in dipole field.
    L: Initial equatorial distance in units of Re
    v: speed
    eqpa: equatorial pitch angle
    Ref.: Martin Walt, 'Introduction to Geomagnetically Trapped Radiation', p.49"""
    b = v/c
    gamma = 1/np.sqrt(1-b**2)
    eqpa = eqpa *np.pi/180
    return 1.557e4 /(L*gamma*b**2) * (1 - 1/3 * (np.sin(eqpa)**0.62))

def dipole_bounce_period(L, v, eqpa):
    """Returns the approximate bounce period of a particle in dipole field.
    L: Initial equatorial distance in units of Re
    v: speed
    eqpa: equatorial pitch angle
    Ref.: Martin Walt, 'Introduction to Geomagnetically Trapped Radiation', p.49"""
    b = v/c
    eqpa = eqpa *np.pi/180
    return 0.117 * L/b * (1 - 0.4635 * (np.sin(eqpa)**0.75))

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
    """Returns the second invariant I in dipole field.
    R0: Equatorial distance.
    Bm: Mirror field strength."""
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
    """Gradient of the second invariant in dipole.
    Valid only for equatorial points (x,y)"""
    r0 = np.sqrt(x**2+y**2)
    f = lambda x: dipoleI(r0+x, Bm)
    mag = (f(h) - f(-h))/(2*h)
    return mag * np.array((x,y,0))/r0

def drift_error(b):
    """Returns a vector of relative error of the bounce-averaged drift path
    compared to a circle (expected rift path under a dipole)"""
    
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