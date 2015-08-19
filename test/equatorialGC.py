# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:36:48 2015

@author: kaan
"""

# Follow a proton under the dipolar field of the Earth.

from rapt import m_pr, e, Re, params
from rapt.traj import Particle, GuidingCenter
from rapt.utils import speedfromKE
from rapt.fields import dipole

from numpy import cos, pi
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D

v = speedfromKE(1e5, m_pr, 'ev')  # speed
pa = 90                           # pitch angle in degrees
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
params["cyclotronresolution"] = 30
params["GCtimestep"] = 0.2
p = Particle()
p.init(g)
# Follow the guiding center and the particle for 300s, store the result internally
g.advance(10)
p.advance(10)
pgc = p.guidingcenter() # Guiding center of the particle trajectory

# plot the x-y corordinates of g, and the guiding-center of the particle
pl.figure()
pl.plot( g.getx()/Re, g.gety()/Re, 'o', p.getx()/Re, p.gety()/Re, pgc[:,0]/Re, pgc[:,1]/Re )
pl.xlabel('x [Re]')
pl.ylabel('y [Re]')

# Plot the kinetic energy vs. time
pl.figure()
pl.plot(g.gett(), g.ke()/e-1e6, p.gett(), p.ke()/e-1e6)
pl.title("Kinetic energy")
pl.xlabel('time [s]')
pl.ylabel('KE [MeV]')