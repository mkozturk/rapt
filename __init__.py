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
    "GCtimestep": 1,
    "gradientstepsize": 0.00001*Re,
    "solvertolerances":(None, None), # rtol,atol values for scipy.integrate.odeint
}

from rapt import utils
from rapt import fields
from rapt.traj import Particle, GuidingCenter