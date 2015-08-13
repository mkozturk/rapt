import numpy as np
from rapt import Re, B0

def uniformz(r):
    """Uniform field B = [0,0,1]"""
    return np.array((0,0,1))

def linear_inc(r):
    """Linearly increasing field B = [0, 0, x]"""
    return np.array([0, 0, r[0]])
    
def dipole(r):
    """Returns the dipole field of the Earth at point r."""
    x,y,z = r[0], r[1], r[2]
    return -B0*Re**3 / pow(x*x+y*y+z*z, 5.0/2.0) * np.array([3*x*z, 3*y*z, (2*z*z -x*x- y*y)])