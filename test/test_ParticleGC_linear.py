# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:36:48 2015

@author: kaan
"""

# Follow a 100keV proton under the dipolar field of the Earth.

from rapt import Particle, params
from rapt.traj import GuidingCenter
from rapt.utils import speedfromKE
from rapt.fields import linear_inc

from numpy import cos, pi
import matplotlib.pyplot as pl

params['solvertolerances'] = (1e-10,1e-10)
params['cyclotronresolution'] = 20
params['gradientstepsize'] = 1e-4

p = Particle((4,0,0),(0,-0.1,0), 0, 1, 1, linear_inc)
g = GuidingCenter()
g.init(p)

# Follow the particle and the guiding center, store result internally
g.advance(20)
p.advance(20)
pgc = p.guidingcenter() # Guiding center of the particle trajectory

# plot the x-y corordinates of g, and the guiding-center of the particle
pl.figure()
pl.plot( g.getx(), g.gety(), 'o', p.getx(), p.gety(), pgc[:,0], pgc[:,1] )
pl.xlabel('x [Re]')
pl.ylabel('y [Re]')
