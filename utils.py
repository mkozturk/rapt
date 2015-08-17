import numpy as np
from numpy.linalg import norm
from rapt import c,e,params

__M = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                ]) # for the curl calculation with central differences
    
def B(pos, field):
    """Strength of magnetic field at position pos."""
    Bvec = field(pos)
    return np.sqrt(np.dot(Bvec, Bvec))

def unitb(pos, field):
    Bvec = field(pos)
    return Bvec / np.sqrt(np.dot(Bvec, Bvec))

def gradB(pos, field):
    """Returns the gradient of field strength at position pos."""
    d=params["gradientstepsize"]
    x,y,z = pos
    return [(B((x+d,y,z), field) - B((x-d,y,z), field))/(2*d),
            (B((x,y+d,z), field) - B((x,y-d,z), field))/(2*d),
            (B((x,y,z+d), field) - B((x,y,z-d), field))/(2*d)]

def curlb(pos, field):
    """Returns the curl of the field direction vector."""
    d=params["gradientstepsize"]
    pos = np.array(pos)
    beta = np.concatenate((
        unitb(pos + (d,0,0), field),
        unitb(pos - (d,0,0), field),
        unitb(pos + (0,d,0), field),
        unitb(pos - (0,d,0), field),
        unitb(pos + (0,0,d), field),
        unitb(pos - (0,0,d), field)
        ))
    return np.dot(__M, beta) / (2*d)
    
def cyclotron_period(pos, vel, field, mass, charge):
    """Returns the period of the cyclotron motion of a particle."""
    gamma = 1.0/np.sqrt(1 - np.dot(vel,vel)/c**2)
    B = np.array(field(pos))
    return 2*np.pi*gamma*mass/(abs(charge)*np.sqrt(np.dot(B,B)))

def cyclotron_radius(pos,vel,field,mass,charge):
    """Returns the period of a particle at given position and velocity."""
    vsq = np.dot(vel, vel)
    gamma = 1.0/np.sqrt(1 - vsq/c**2)
    B = np.array(field(pos))
    Bmag = np.sqrt(np.dot(B,B))
    vpar = np.dot(vel,B)/Bmag
    vperp = np.sqrt(vsq-vpar**2)
    return gamma*mass*vperp/(abs(charge)*Bmag)

def cyclotron_radius2(pos,vpar,v,field,mass,charge):
    """Returns the period of a particle at given position, parallel speed, and total speed."""
    gamma = 1.0/np.sqrt(1 - (v/c)**2)
    B = np.array(field(pos))
    Bmag = np.sqrt(np.dot(B,B))
    vperp = np.sqrt((v-vpar)*(v+vpar))
    return gamma*mass*vperp/(abs(charge)*Bmag)

def magnetic_moment(pos, vpar, v, field, mass):
    """Returns the magnetic moment of the given guiding center."""
    gamma = 1.0/np.sqrt(1 - (v/c)**2)
    return gamma**2*mass*(v-vpar)*(v+vpar)/(2*B(pos, field)) # magnetic moment

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

def speedfromKE(KE, mass, unit="ev"):
    """Returns the speed of a particle with given relativistic kinetic energy and mass."""
    mc2 = mass * c**2    
    if unit.lower()=="ev":
        KE = KE*e # convert to Joules
    if KE/mc2 < 1e-6: # nonrelativistic
        return np.sqrt(2*KE/mass)
    else:  # relativistic
        return c * np.sqrt(1-(mc2/(mc2+KE))**2)

def GuidingCenter(r, v, field, mass, charge, tol=1e-3, maxiter=20, debug=False):
    """Returns the guiding center for the given particle.""" 

    def gyrovector(r,v=v,field=field, mass=mass, charge=charge):
        vsq = np.dot(v,v)
        gamma = 1/np.sqrt(1-vsq/c**2)
        B = np.array(field(r))
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
                B = np.array(field(GC))
                vp = np.dot(v,B)/norm(B)
                return GC, vp, norm(v)
        GC_old = GC
        iteration += 1
    print("Could not reach the specified tolerance after ", iteration, " iterations.")

def GCtoFP(R,vp,speed,field, mass, charge):
    """Returns a particle position for the given guiding center."""
    R = np.array(R)
    B = field(R)
    Bsq = np.dot(B,B)
    b = B / np.sqrt(Bsq)  # unit vector in the field direction
    pa = np.arccos(vp/speed)  # pitch angle
    rc = cyclotron_radius2(R,vp,speed,field,mass,charge)
    u = getperp(B) # first reference vector
    u = u / np.sqrt(np.dot(u,u))   # make unit vector
    w = np.cross(b,u)  # second reference vector
    phase = 0 # set the phase arbitrarily
    s = np.sign(charge)
    pos = R + rc*(np.cos(phase)*u + np.sin(phase)*w)
    vel = speed * (np.cos(pa)*b + s*np.sin(pa)*np.sin(phase)*u - s*np.sin(pa)*np.cos(phase)*w)
    return pos, vel

# Dynamical quantities over trajectories

def trajKE(traj,mass):
    """Returns the array of relativistic kinetic energy along the particle trajectory."""
    v = norm(traj[:,3:], axis=1)
    gamma = 1/np.sqrt(1-(v/c)**2)
    return (gamma-1)*mass*c**2

def GCKE(traj, field, mu, speed, mass):
    """Returns the array of relativistic kinetic energy along the guiding-center trajectory."""
    mc2 = mass*c**2
    gammasq = 1/(1 - (speed/c)**2)
    out = np.zeros(len(traj))
    for i, row in enumerate(traj):
        pos, vp = row[:3], row[3]
        B = norm( field(pos) )
        out[i] = np.sqrt(2*mu*B*mc2 + gammasq*mass*mc2*vp**2 + mc2**2) - mc2
    return out

def trajGC(traj,field, mass, charge):
    """Returns the guiding-center position, parallel speed, total speed from given particle trajectory."""
    out = []
    for row in traj:
        r,v = row[:3], row[3:]
        rgc, vp, spd = GuidingCenter(r, v, field, mass, charge)
        out.append(list(rgc)+[vp,spd])
    return np.array(out)

def trajmu(traj,field,mass,charge):
    '''Returns the magnetic moment (first invariant) from particle trajectory.'''
    out = []
    for row in traj:
        r,v = row[:3], row[3:]
        rgc, vp, spd = GuidingCenter(r, v, field, mass, charge)
        out.append(magnetic_moment(rgc, vp, spd, field, mass))
    return np.array(out)
