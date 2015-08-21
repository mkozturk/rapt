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
        self.field = field  # magnetic field function, taking position array and returning field array
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
            self.pos, self.vel = ru.GCtoFP(p.trajectory[-1,1:4], p.trajectory[-1,4], p.v, self.field, self.mass, self.charge, gyrophase)
            self.trajectory = np.concatenate(([self.tcur], self.pos, self.vel))
            self.trajectory = np.reshape(self.trajectory, (1,7))
        else:
            raise(ValueError, "Particle or GuidingCenter objects required.")
            
    def advance(self, delta):
        """Advance the particle position and velocity for time 'delta' starting at the current time, position, and velocity."""
        pos = self.trajectory[-1,1:4]
        vel = self.trajectory[-1,4:]
        # set resolution of cyclotron motion
        res = ru.cyclotron_period(pos, vel, self.field, self.mass, self.charge) / params['cyclotronresolution']
        
        # gamma should be evaluated inside deriv() if the speed changes along the trajectory (e.g. due to electric fields)        
        gamma = 1.0/np.sqrt(1 - np.dot(vel, vel) / c**2)
     
        def deriv(Y, t=0):
            out = np.zeros(7) 
            out[0] = 1        # dt/dt = 1
            out[1:4] = Y[4:]  # d(pos)/dt = vel
            out[4:] = self.charge/(self.mass*gamma) * np.cross(Y[4:], self.field(Y[1:4]))  # d(vel)/dt = (q/m)vel x B(pos)
            return out
        
        times = np.arange(self.tcur, self.tcur+delta, res)
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
    def guidingcenter(self):
        out = []
        for row in self.trajectory:
            r,v = row[1:4], row[4:]
            rgc, vp, spd = ru.GuidingCenter(r, v, self.field, self.mass, self.charge)
            out.append(list(rgc)+[vp,spd])
        return np.array(out)
    def mu(self):
        out = []
        for row in self.trajectory:
            r,v = row[1:4], row[4:]
            rgc, vp, spd = ru.GuidingCenter(r, v, self.field, self.mass, self.charge)
            out.append(ru.magnetic_moment(rgc, vp, spd, self.field, self.mass))
        return np.array(out)
    def cycrad(self):
        """The current cyclotron radius."""
        return ru.cyclotron_radius(self.trajectory[-1, 1:4], self.trajectory[-1, 4:], self.field, self.mass, self.charge)
    def cycper(self):
        """The current cyclotron period."""
        return ru.cyclotron_period(self.trajectory[-1, 1:4], self.trajectory[-1, 4:], self.field, self.mass, self.charge)

class GuidingCenter:
    def __init__(self, pos=None, vpar=None, speed=None, t0=None, mass=None, charge=None, field=None, eq='northropteller'):
        self.pos = pos  # initial position array
        self.vp = vpar  # initial parallel speed
        self.v = speed    # speed of the particle
        self.tcur = t0    # current time
        self.mass = mass  # mass of the particle
        self.charge = charge  # charge of the particle
        self.field = field  # magnetic field function, taking position array and returning field array
        self.eq = eq     # Equation to solve. 'northropteller' or 'brizardchan'
        if not (pos==None or vpar==None or speed==None or t0==None): # if initial state is given explicitly
            self.mu = ru.magnetic_moment(self.pos, self.vp, self.v, self.field, self.mass)
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
            self.mu = ru.magnetic_moment(self.pos, self.vp, self.v, self.field, self.mass)
            self.trajectory = np.concatenate(([self.tcur], self.pos, [self.vp]))
            self.trajectory = np.reshape(self.trajectory, (1,5))
        elif isinstance(p, Particle):
            self.mass = p.mass
            self.charge = p.charge
            self.field = p.field
            self.tcur = p.trajectory[-1,0]
            self.pos, self.vp, self.v = ru.GuidingCenter(p.trajectory[-1,1:4], p.trajectory[-1,4:], self.field, self.mass, self.charge)
            self.mu = ru.magnetic_moment(self.pos, self.vp, self.v, self.field, self.mass)
            self.trajectory = np.concatenate(([self.tcur], self.pos, [self.vp]))
            self.trajectory = np.reshape(self.trajectory, (1,5))
        else:
            raise(ValueError, "Particle or GuidingCenter objects required.")

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
            Bvec = self.field(Y[1:4])
            B = np.sqrt(np.dot(Bvec,Bvec))
            bdir = Bvec / B
            gB = ru.gradB(Y[1:4],self.field)
            
            return np.concatenate(
            (
            [1],
            gamma*self.mass/(2*self.charge*B**2) * (self.v**2 + Y[4]**2)*np.cross(bdir,gB) + Y[4]*bdir,
            [-self.mu/(self.mass*gamma**2) * np.dot(bdir, gB)]
            )
            )
        times = np.arange(self.tcur, self.tcur+delta, dt)
        rtol, atol = params["solvertolerances"]
        traj = odeint(deriv, self.trajectory[-1,:], times, rtol=rtol, atol=atol)
        self.trajectory = np.concatenate((self.trajectory, traj[1:,:]))
        self.tcur = self.trajectory[-1,0]
    
    def BrizardChanAdvance(self, delta):
        dt = params["GCtimestep"]
        gamma = 1.0/np.sqrt(1 - (self.v/c)**2)

        def deriv(Y, t=0):
            B = self.field(Y[1:4])
            Bmag = np.sqrt(np.dot(B,B))
            unitb = B / Bmag
            gB = ru.gradB(Y[1:4], self.field)
            cb = ru.curlb(Y[1:4], self.field)
            Bstar = B + gamma * self.mass * Y[4] * cb / self.charge
            Bstarpar = np.dot(B,Bstar) / Bmag
            retval = np.ones(5) 
            retval[1:4] = (Y[4] * Bstar  + self.mu * np.cross(unitb, gB) / (self.charge * gamma) ) / Bstarpar
            retval[4] = -self.mu * np.dot(Bstar, gB) / (self.mass * gamma*gamma * Bstarpar)
            return retval
        
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
            out[i] = ru.B(row[1:4],self.field)
        return out
    
    def ke(self):
        mc2 = self.mass*c**2
        gammasq = 1/(1 - (self.v/c)**2)
        return np.sqrt(2*self.mu*self.getB()*mc2 + gammasq*self.mass*mc2*self.trajectory[:,4]**2 + mc2**2) - mc2