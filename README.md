# RAPT: A Python framework for tracing relativistic test particles in general electromagnetic fields

RAPT (Rice Adaptive Particle Tracer) is an object-oriented package for simulating trajectories of charged particles under electric and magnetic fields. It is intended to be used for plasma physics research and education.

Features:

* Relativistic equations of motion.
* Following the particle or guiding center trajectories.
* Automatic switching between particle and guiding-center trajectories for efficiency.
* Accurate, state-of-the-art equations for the guiding center.
* Support for time-dependent fields.
* Extensible with user-defined field models.
* Magnetic field-line tracing.
* Utilites to evaluate particle-related or field-line related quantities.
* Modular, object-oriented design for easy maintenance and extension.

# # Requirements
* numpy>=1.17
* scipy>=1.3.1

## Installation

## Examples
### Proton trajectory around the Earth dipole
```from numpy import sin, cos, pi
from rapt import m_pr, e, Re, Particle, params
from rapt.utils import speedfromKE
from rapt.fields import EarthDipole

params["cyclotronresolution"] = 20  # set step size to about 1/20 of a period
v = speedfromKE(1e6, m_pr, 'ev')    # speed of a 1MeV proton
pa = 30 * pi / 180                  # pitch angle

# Create an instance of Particle class and initialize
p = Particle(
        pos = (6*Re, 0, 0),                 # initial position
        vel = (0, - v*sin(pa), v*cos(pa)),  # initial velocity 
        t0 = 0,                             # starting time
        mass = m_pr,                        # proton mass
        charge = e,                         # proton charge
        field = EarthDipole()               # Earth's dipole field
)
# Follow the particle for 100s, store result internally
p.advance(10)

# Optional visualization with matplotlib:
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

pl.figure(1)
pl.plot(p.gett(), p.getz()/Re) # Plot the z coordinate vs. time
pl.xlabel('time [s]')
pl.ylabel('z [Re]')
pl.grid()

fig = pl.figure(2)
ax = fig.add_subplot(111, projection='3d') # Display the trajectory in 3D
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot(p.getx()/Re, p.gety()/Re, p.getz()/Re)

pl.show()

```
### Jupyter notebooks
* [Simulating a proton trajectory in the Earth's magnetic dipole field](https://github.com/mkozturk/rapt/blob/master/examples/Using%20the%20Particle%20class.ipynb): Illustrates the usage of the `Particle` class.

* [Electron guiding-center orbits under double-dipole field model](https://github.com/mkozturk/rapt/blob/master/examples/GuidingCenter%20Example%20-%20Electron%20under%20double%20dipole%20field.ipynb): Illustrates the usage of the `GuidingCenter` class.

* [Speiser orbits](https://github.com/mkozturk/rapt/blob/master/examples/Adaptive%20Example%20-%20Speiser%20orbits.ipynb): Illustrates the usage of the `Adaptive` class.

* [Drift-shell splitting: Bounce-averaged proton orbits](https://github.com/mkozturk/rapt/blob/master/examples/BounceCenter%20example%20-%20Drift-shell%20splitting.ipynb). Illustrates the usage of `BounceCenter` class.

* [Plotting field lines](https://github.com/mkozturk/rapt/blob/master/examples/Field%20line%20example.ipynb). How to visualize field lines in a given field model; plotting the field strength along a field line; getting the critical points (optima of field strength) on the field line. Illustrates the `Fieldline` class.

* [Creating new analytic field models](https://github.com/mkozturk/rapt/blob/master/examples/Creating%20new%20fields.ipynb) Illustrates generating new field models where the electric and magnetic fields have explicit mathematical formulas.
