# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:36:48 2015

@author: kaan
"""

# Follow a 1MeV proton under the dipolar field of the Earth.

from rapt import m_pr, e, Re
from rapt.traj import GuidingCenter
from rapt.utils import speedfromKE
from rapt.fields import dipole

from numpy import cos, pi
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

v = speedfromKE(1e6, m_pr, 'ev')  # speed
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

# Follow the particle for 10s, store result internally
g.advance(10)

# Plot the z coordinate vs. time
pl.plot( g.gett(), g.getz()/Re )
pl.xlabel('time [s]')
pl.ylabel('z [Re]')

# Follow for 10 more seconds, starting from the last point.
g.advance(10)

# Display the trajectory in 3D

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.plot(g.getx()/Re, g.gety()/Re, g.getz()/Re)

# Plot the kinetic energy vs. time
pl.figure()
pl.plot(g.gett(), g.ke()/e/1e6)
pl.title("Kinetic energy")
pl.xlabel('time [s]')
pl.ylabel('KE [MeV]')