# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:59:27 2015

@author: kaan
"""
import numpy as np
from pylab import plot
from rapt import Re, m_pr, e
from rapt.utils import speedfromKE
from rapt.fields import dipole
from rapt.traj import Particle, GuidingCenter

r0 = np.array((6*Re,0,0))
v = speedfromKE(1e6, m_pr, "ev")
pa = 30 * np.pi / 180
v0 = v * np.array((0, -np.sin(pa), np.cos(pa)))

p = Particle(r0, v0, 0, m_pr, e, dipole)

g = GuidingCenter()
g.init(p)

p.advance(10)
g.advance(10)

plot(p.getx(), p.gety(), g.getx(), g.gety())
