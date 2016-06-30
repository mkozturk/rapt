import numpy as np
import tricubic

n = 3
f = np.zeros((n,n,n), dtype='float')

#for i in range(n):
#  for j in range(n):
#    for k in range(n):
#      f[i][j][k] = i+j+k #some function f(x,y,z) is given on a cubic grid indexed by i,j,k
#ip = tricubic.tricubic(list(f), [n,n,n]) #initialize interpolator with input data on cubic grid
#for i in range(10):
#  res = ip.ip(list(np.random.rand(3)*(n-1))) #interpolate the function f at a random point in space
#  print (res)

x = np.linspace(0,10,5)
y = np.linspace(0,10,7)
z = np.linspace(0,10,20)
nx, ny, nz = len(x), len(y), len(z)
X, Y, Z = np.meshgrid(y,x,z)
#f = X + Y + Z
f = np.zeros((nx,ny,nz), dtype='float')

for i in range(nx):
  for j in range(ny):
    for k in range(nz):
      f[i][j][k] = i+j+k #some function f(x,y,z) is given on a cubic grid indexed by i,j,k

ip = tricubic.tricubic(list(f), [nx,ny,nz]) #initialize interpolator with input data on cubic grid
for i in range(10):
    res = ip.ip(list(np.random.rand(3)*10)) #interpolate the function f at a random point in space
    print (res)
