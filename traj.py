import numpy as np
from scipy.integrate import odeint

import rapt.utils as ru
from rapt import c, params

class Particle:
    def __init__(self, pos=None, vel=None, t0=0, mass=None, charge=None, field=None):
        self.pos = pos  # initial position array
        self.vel = vel  # initial velocity array
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  #  the field object
        if not (pos == None or vel == None): # if initial state is given explicitly
            self.trajectory = np.concatenate(([self.tcur], self.pos, self.vel))
            self.trajectory = np.reshape(self.trajectory, (1,7))
            
    def init(self, p, gyrophase=0):  # Initialization with another Particle or GuidingCenter instance
        if isinstance(p, Particle):
            self.mass = p.mass
            self.charge = p.charge
            self.field = p.field
            self.tcur = p.trajectory[-1,0]            
            self.pos = p.trajectory[-1,1:4]
            self.vel = p.trajectory[-1,4:]
            self.trajectory = np.concatenate(([self.tcur], self.pos, self.vel))
            self.trajectory = np.reshape(self.trajectory, (1,7))
        elif isinstance(p, GuidingCenter):
            self.mass = p.mass
            self.charge = p.charge
            self.field = p.field
            self.tcur = p.trajectory[-1,0]
            self.pos, self.vel = ru.GCtoFP(self.tcur, p.trajectory[-1,1:4], p.trajectory[-1,4], p.v, self.field, self.mass, self.charge, gyrophase)
            self.trajectory = np.concatenate(([self.tcur], self.pos, self.vel))
            self.trajectory = np.reshape(self.trajectory, (1,7))
        else:
            raise(ValueError, "Particle or GuidingCenter objects required.")
            
    def advance(self, delta):
        """Advance the particle position and velocity for time 'delta' starting at the current time, position, and velocity."""
        t = self.trajectory[-1,0]        
        pos = self.trajectory[-1,1:4]
        vel = self.trajectory[-1,4:]
        # set resolution of cyclotron motion
        res = ru.cyclotron_period(t, pos, vel, self.field, self.mass, self.charge) / params['cyclotronresolution']
     
        def deriv(Y, t=0):
            gamma = 1.0/np.sqrt(1 - np.dot(Y[4:], Y[4:]) / c**2)
            # later optimization: put gamma outside if the E-field is always zero.
            out = np.zeros(7) 
            out[0] = 1        # dt/dt = 1
            out[1:4] = Y[4:]  # d(pos)/dt = vel
            out[4:] = (self.field.E(Y[:4]) + 
            self.charge * np.cross(Y[4:], self.field.B(Y[:4]))
            ) / (self.mass*gamma) # d(vel)/dt = E(t, pos) / m + (q/m)vel x B(t, pos)
            return out
        
        times = np.arange(self.tcur, self.tcur+delta, res)
        rtol, atol = params["solvertolerances"]
        traj = odeint(deriv, self.trajectory[-1,:], times, rtol=rtol, atol=atol)
        self.trajectory = np.concatenate((self.trajectory, traj[1:,:]))
        self.tcur = self.trajectory[-1,0]
    
    def isadiabatic(self):
        """Returns True if the particle's motion satisfies the adiabaticity conditions
        at the present location."""
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
        return self.trajectory[:,0]
    def getx(self):
        return self.trajectory[:,1]
    def gety(self):
        return self.trajectory[:,2]
    def getz(self):
        return self.trajectory[:,3]
    def getvx(self):
        return self.trajectory[:,4]
    def getvy(self):
        return self.trajectory[:,5]
    def getvz(self):
        return self.trajectory[:,6]
    def getv(self):  # total speed of the particle
        return np.sqrt( self.getvx()**2 + self.getvy()**2 + self.getvz()**2 )
    def getr(self):  # Distance from the origin
        return np.sqrt( self.getx()**2 + self.gety()**2 + self.getz()**2 )
    def gettheta(self): # Polar angle
        return np.arctan2( self.gety(), self.getx() )
    def getphi(self):  # Azimuthal angle
        return np.arccos( self.getz() / self.getr() )
    def ke(self):      # Kinetic energy
        gamma = 1/np.sqrt(1 - self.getv()**2/c**2)
        return (gamma-1)*self.mass*c**2
    def kenr(self):    # Nonrelativistic kinetic energy
        return 0.5*self.mass*self.getv()**2
    def guidingcenter(self):
        out = []
        for row in self.trajectory:
            t,r,v = row[0], row[1:4], row[4:]
            rgc, vp, spd = ru.guidingcenter(t, r, v, self.field, self.mass, self.charge)
            out.append(list(rgc)+[vp,spd])
        return np.array(out)
    def mu(self):
        out = []
        for row in self.trajectory:
            t, r,v = row[0], row[1:4], row[4:]
            rgc, vp, spd = ru.GuidingCenter(t, r, v, self.field, self.mass, self.charge)
            out.append(ru.magnetic_moment(t, rgc, vp, spd, self.field, self.mass))
        return np.array(out)
    def cycrad(self):
        """The current cyclotron radius."""
        t, r, v = self.trajectory[-1, 0], self.trajectory[-1, 1:4], self.trajectory[-1, 4:]
        return ru.cyclotron_radius(t, r, v, self.field, self.mass, self.charge)
    def cycper(self):
        """The current cyclotron period."""
        t, r, v = self.trajectory[-1, 0], self.trajectory[-1, 1:4], self.trajectory[-1, 4:]
        return ru.cyclotron_period(t, r, v, self.field, self.mass, self.charge)

class GuidingCenter:
    # TO DO: Not compatible with electric fields and time-varying magnetic fields.
    # The advance() method needs to have:
    #   (1) The ExB term
    #   (2) The speed as a separate variable that's updated in time.
    #       Update the kinetic energy at each step and get speed from it.
    def __init__(self, pos=None, vpar=None, speed=None, t0=None, mass=None, charge=None, field=None, eq='brizardchan'):
        self.pos = pos  # initial position array
        self.vp = vpar  # initial parallel speed
        self.v = speed    # speed of the particle
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  # magnetic field function, taking position array and returning field array
        self.eq = eq     # Equation to solve. 'northropteller' or 'brizardchan'
        if not (pos==None or vpar==None or speed==None or t0==None): # if initial state is given explicitly
            self.mu = ru.magnetic_moment(self.tcur, self.pos, self.vp, self.v, self.field, self.mass)
            self.trajectory = np.concatenate(([t0], pos, [vpar]))
            self.trajectory = np.reshape(self.trajectory, (1,5))
    
    def init(self, p):  # Initialization with another Particle or GuidingCenter instance
        if isinstance(p, GuidingCenter):
            self.tcur = p.trajectory[-1,0]            
            self.pos = p.trajectory[-1,1:4]
            self.vp = p.trajectory[-1,4]
            self.v = p.v
            self.mass = p.mass
            self.charge = p.charge
            self.field = p.field
            self.eq = p.eq
            self.mu = ru.magnetic_moment(self.tcur, self.pos, self.vp, self.v, self.field, self.mass)
            self.trajectory = np.concatenate(([self.tcur], self.pos, [self.vp]))
            self.trajectory = np.reshape(self.trajectory, (1,5))
        elif isinstance(p, Particle):
            self.mass = p.mass
            self.charge = p.charge
            self.field = p.field
            self.tcur = p.trajectory[-1,0]
            self.pos, self.vp, self.v = ru.guidingcenter(self.tcur, p.trajectory[-1,1:4], p.trajectory[-1,4:], self.field, self.mass, self.charge)
            self.mu = ru.magnetic_moment(self.tcur, self.pos, self.vp, self.v, self.field, self.mass)
            self.trajectory = np.concatenate(([self.tcur], self.pos, [self.vp]))
            self.trajectory = np.reshape(self.trajectory, (1,5))
        else:
            raise(ValueError, "Particle or GuidingCenter objects required.")
        
    def isadiabatic(self):
        """Returns True if the particle's motion satisfies the adiabaticity conditions
        at the present location."""
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
    
    def advance(self, delta):
        if self.eq == "northropteller":
            self.NorthropTellerAdvance(delta)
        elif self.eq == "brizardchan":
            self.BrizardChanAdvance(delta)
    
    def NorthropTellerAdvance(self, delta):
        """Advance the GC position and parallel speed for time 'delta' starting at the current time, position, parallel speed."""
        dt = params["GCtimestep"]
        gamma = 1.0/np.sqrt(1 - (self.v/c)**2)
        
        def deriv(Y, t=0):
            Bvec = self.field.B(Y[:4])
            B = np.sqrt(np.dot(Bvec,Bvec))
            bdir = Bvec / B
            gB = self.field.B(Y[:4])
            return np.concatenate(
            (
            [1],
            gamma*self.mass/(2*self.charge*B**2) * (self.v**2 + Y[4]**2)*np.cross(bdir,gB) + Y[4]*bdir,
            [-self.mu/(self.mass*gamma**2) * np.dot(bdir, gB)]
            )
            )
        if delta <= dt:
            times = np.array([self.tcur, self.tcur+delta])
        else:
            times = np.arange(self.tcur, self.tcur+delta, dt)
        rtol, atol = params["solvertolerances"]
        traj = odeint(deriv, self.trajectory[-1,:], times, rtol=rtol, atol=atol)
        self.trajectory = np.concatenate((self.trajectory, traj[1:,:]))
        self.tcur = self.trajectory[-1,0]
    
    def BrizardChanAdvance(self, delta):
        dt = params["GCtimestep"]
        gamma = 1.0/np.sqrt(1 - (self.v/c)**2)

        def deriv(Y, t=0):
            B = self.field.B(Y[:4])
            Bmag = np.sqrt(np.dot(B,B))
            unitb = B / Bmag
            gB = self.field.gradB(Y[:4])
            cb = self.field.curlb(Y[:4])
            Bstar = B + gamma * self.mass * Y[4] * cb / self.charge
            Bstarpar = np.dot(B,Bstar) / Bmag
            retval = np.ones(5) 
            retval[1:4] = (Y[4] * Bstar  + self.mu * np.cross(unitb, gB) / (self.charge * gamma) ) / Bstarpar
            retval[4] = -self.mu * np.dot(Bstar, gB) / (self.mass * gamma*gamma * Bstarpar)
            return retval
        if delta <= dt:
            times = np.array([self.tcur, self.tcur+delta])
        else:
            times = np.arange(self.tcur, self.tcur+delta, dt)
        rtol, atol = params["solvertolerances"]
        traj = odeint(deriv, self.trajectory[-1,:], times, rtol=rtol, atol=atol)
        self.trajectory = np.concatenate((self.trajectory, traj[1:,:]))
        self.tcur = self.trajectory[-1,0]
        
    def gett(self):
        return self.trajectory[:,0]
    def getx(self):
        return self.trajectory[:,1]
    def gety(self):
        return self.trajectory[:,2]
    def getz(self):
        return self.trajectory[:,3]
    def getvp(self):
        return self.trajectory[:,4]
    def getr(self):  # Distance from the origin
        return np.sqrt( self.getx()**2 + self.gety()**2 + self.getz()**2 )
    def gettheta(self): # Polar angle
        return np.arctan2( self.gety(), self.getx() )
    def getphi(self):  # Azimuthal angle
        return np.arccos( self.getz() / self.getr() )
    def getB(self):
        out = np.zeros(len(self.trajectory))
        for i, row in enumerate(self.trajectory):
            out[i] = self.field.magB(row[:4])
        return out
    
    def cycrad(self):
        """The current cyclotron radius."""
        t, r, vp = self.trajectory[-1, 0], self.trajectory[-1, 1:4], self.trajectory[-1, 4]
        return ru.cyclotron_radius2(t, r, vp, self.v, self.field, self.mass, self.charge)

    def cycper(self):
        """The current cyclotron period."""
        t, r = self.trajectory[-1, 0], self.trajectory[-1, 1:4]
        return ru.cyclotron_period2(t, r, self.v, self.field, self.mass, self.charge)

    def ke(self):
        # BUG: Drift speed term not added.
        mc2 = self.mass*c**2
        gammasq = 1/(1 - (self.v/c)**2)
        return np.sqrt(2*self.mu*self.getB()*mc2 + gammasq*self.mass*mc2*self.trajectory[:,4]**2 + mc2**2) - mc2
    
    def kenr(self):
        # BUG: Drift speed term not added.
        return self.mu * self.getB() + 0.5*self.mass*self.trajectory[:,4]**2
        
class Adaptive:
    # The Adaptive class automatically decides whether to use Particle or GuidingCenter
    # by looking at local and instantaneous conditions and at the particle state.

    # This class is different from Particle or GuidingCenter in that it does not
    # store the trajectory data as a single 2D array. Instead, it stores a
    # list of Particle and GuidingCenter objects, which themselves store the
    # trajectory info. However, the interface is identical.

    def __init__(self, pos=None, vel=None, t0=0, mass=None, charge=None, field=None):
        self.pos = pos  # initial position array
        self.vel = vel  # initial velocity array
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  #  the field object
        p = Particle(pos,vel,t0,mass,charge,field)
        if p.isadiabatic():
            g = GuidingCenter()
            g.init(p)
            self.trajlist = [g]
        else:
            self.trajlist = [p]

    def advance(self, delta):
        t = 0
        gctimestep = 2   # follow the GC for that many seconds before checking for adiabaticity
        partimestep = 10 # follow the particle for that many periods before checking for adiabaticity
        while t < delta:
            # Check if the adiabaticity condition holds in the last state.
            # If adiabatic, proceed with GuidingCenter type. 
            # Otherwise proceed with Particle type.
            current = self.trajlist[-1] # a Particle or a GuidingCenter
            if current.isadiabatic():
                # If conditions are adiabatic, follow a GC.
                # If not already a GC, create a GuidingCenter and initialize with the latest particle.
                # If already a GC, just advance the solution.
                if not isinstance(current, GuidingCenter):
                    g = GuidingCenter()
                    g.init(current)
                    self.trajlist.append(g)
                    current = self.trajlist[-1]
                    print("Switched to guiding center at time", current.tcur)
                
                if t + gctimestep < delta:
                    current.advance(gctimestep)
                else: # the next time step would exceed final time "delta"
                    current.advance(delta-t)
                
            else: # conditions not adiabatic. Follow particle path instead of GC.
                # If not already a particle, create a Particle and initialize with the latest guiding center position.
                if not isinstance(current, Particle):
                    p = Particle()
                    p.init(current)
                    self.trajlist.append(p)
                    current = self.trajlist[-1]
                    print("Switched to particle at time", current.tcur)

                dt = partimestep*current.cycper()
                if t + dt < delta:
                    current.advance(dt)
                else: # the next time step would exceed final time "delta"
                    current.advance(delta-t)
                    break

            t = current.trajectory[-1,0]

    def gett(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.gett() ) )
        return res
    def getx(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getx()) )
        return res
    def gety(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.gety()) )
        return res
    def getz(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getz()) )
        return res
    def getr(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getr()) )
        return res
    def getphi(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.getphi()) )
        return res
    def gettheta(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.gettheta()) )
        return res
    def getv(self):
        res = np.array([])
        for p in self.trajlist:
            if isinstance(p, Particle):
                res = np.concatenate(res, p.getv())
            if isinstance(p, GuidingCenter):
                v = np.full( (len(p.trajectory),1), p.v )
                res = np.concatenate( (res, v) )
        return res
    
    def ke(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.ke()) )
        return res
    
    def kenr(self):
        res = np.array([])
        for p in self.trajlist:
            res = np.concatenate( (res, p.kenr()) )
        return res