"""
This is the docstring for rapt/__init__.py
"""
# Constants
e = 1.602176565e-19 # Elementary charge (Coulomb)
m_pr = 1.672621777e-27 # Proton mass (kg)
m_el = 9.10938291e-31 # Electron mass (kg)
c = 299792458 # speed of light (m/s)
B0 = 3.07e-5 # Earth field strength at magnetic equator (Tesla)
re = Re = 6378137 # Earth radius (meter)

# Define exceptions that are raised when a Particle's motion is adiabatic
# or a GuidingCenter's motion is nonadiabatic.
class Adiabatic(Exception):
    pass
class NonAdiabatic(Exception):
    pass

# Parameters and defaults

params = {
    "cyclotronresolution": 10,
    "Ptimestep": 0,   # Time step for Particle steps. If zero, cyclotronresolution is used. 
    "bounceresolution": 10,
    "GCtimestep": 0,  # Time step for GuidingCenter steps. If zero, bounceresolution is used. 
    "BCtimestep": 0.1, # bounce center time step, in units of the initial bounce period.
    "solvertolerances":(1.49012e-8, 1.49012e-8), # rtol,atol values for scipy ode integrator
    "fieldlineresolution": 50, # field line sampling size = length scale divided by this number
    "flsolver": "rkf", # 'rkf' or 'odeint'
    "eyegradientstep": 0.03*Re, # spatial step size for evaluating the gradient of I.
    "epss": 5e-2,     # spatial adiabaticity threshold
    "epst": 5e-2,     # temporal adiabaticity threshold
    "enforce equatorial": False # If True, forces particle to stay on the z=0 plane.
}

from rapt import utils, fields, fieldline, rkf
from rapt.Particle import Particle
from rapt.GuidingCenter import GuidingCenter
from rapt.Adaptive import Adaptive
from rapt.BounceCenter import BounceCenter
from rapt.fieldline import Fieldline
from rapt.flutils import eye, gradI, halfbouncepath
