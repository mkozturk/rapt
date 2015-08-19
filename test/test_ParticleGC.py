# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:36:48 2015

@author: kaan
"""

# Follow a proton under the dipolar field of the Earth.

from rapt import m_pr, e, Re, Particle, params
from rapt.traj import GuidingCenter
from rapt.utils import speedfromKE
from rapt.fields import dipole

from numpy import cos, pi
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

params['solvertolerances'] = (1e-8,1e-8)
v = speedfromKE(1e5, m_pr, 'ev')  # speed corresponding to 100keV proton
pa = 30                         # pitch angle in degrees
r0 = (6*Re, 0, 0)                 # initial position
# set initial parallel speed
if pa == 90:
    vp0 = 0 # explicitly set to zero to avoid nonzero vp due to roundoff error.
else:
    vp0 = v*cos(pa * pi / 180)
t0 = 0                            # starting time

# Create an instance of GuidingCenter class and initialize
g = GuidingCenter(r0, vp0, v, t0, m_pr, e, dipole)

# Create an instance of Particle, initialize with the same status as g
p = Particle()
p.init(g)

# Follow the particle and the guiding center for 1000s, store result internally
g.advance(500)
p.advance(500)
# Plot the z coordinate vs. time
pl.figure()
pl.plot( g.gett(), g.getz()/Re, p.gett(), p.getz()/Re )
pl.xlabel('time [s]')
pl.ylabel('z [Re]')

# Display the trajectory in 3D

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot(g.getx()/Re, g.gety()/Re, g.getz()/Re)
ax.plot(p.getx()/Re, p.gety()/Re, p.getz()/Re)

# Plot the kinetic energy vs. time
pl.figure()
pl.plot(g.gett(), g.ke()/e, p.gett(), p.ke()/e)
pl.title("Kinetic energy")
pl.xlabel('time [s]')
pl.ylabel('KE [MeV]')