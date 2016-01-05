# Constants
e = 1.602176565e-19 # Elementary charge (Coulomb)
m_pr = 1.672621777e-27 # Proton mass (kg)
m_el = 9.10938291e-31 # Electron mass (kg)
c = 299792458 # speed of light (m/s)
B0 = 3.07e-5 # Tesla (Earth field strength at magnetic equator)
Re = 6378137 # meter (Earth radius)

# Parameters and defaults

params = {
    "cyclotronresolution": 10,
    "GCtimestep": 0.5,
    "BCtimestep": 0.1, # bounce center time step, in units of bounce period.
    "solvertolerances":(None, None), # rtol,atol values for scipy.integrate.odeint
    "fieldlineresolution": 50, # field line sampling size = length scale divided by this number
    "flsolver": "rkf", # 'rkf' or 'odeint'
    "eyegradientstep": 0.03*Re,
    "epss": 1e-2,     # spatial adiabaticity threshold
    "epst": 1e-2     # temporal adiabaticity threshold
}

from rapt import utils, fields, fieldline, rkf
from rapt.traj import Particle, GuidingCenter, Adaptive
from rapt.BounceCenter import BounceCenter
from rapt.flutils import eye, gradI, halfbouncepath
