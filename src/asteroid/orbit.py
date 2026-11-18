
import pykep as pk
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


N = 1000 # number of objects simulated
# The following numpy array will hold all the initial conditions and object radius [x,y,z,vx,vy,vz,r]
ic_state = np.zeros((N, 7))
# We generate randomly orbital parameters and imensions and transform them to Cartesian state. Note we make use of nondimensional units as defined in the dynamics.
for i in range(N):
    a = np.random.uniform(1.02, 1.3)
    e = np.random.uniform(0, 0.02)
    inc = np.random.uniform(np.pi/3, np.pi/3 + 0.05)
    om = np.random.uniform(0, 2*np.pi)
    Om = np.random.uniform(np.pi/3, np.pi/3 + 0.05)
    nu = np.random.uniform(0, 2*np.pi)
    size = np.random.uniform(1e-3/2, 1e-4)
    r, v = pk.par2ic([a, e, inc, Om, om, nu], mu = 1.)
    ic_state[i][:3] = r
    ic_state[i][3:6] = v
    ic_state[i][6] = size


fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')

r = 0.5
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = r*np.cos(u)*np.sin(v)
y = r*np.sin(u)*np.sin(v)
z = r*np.cos(v)
ax.plot_wireframe(x, y, z, color="r", alpha=0.1)

ax.scatter3D(ic_state[:,0], ic_state[:,1], ic_state[:,2], alpha=0.2, s= 5, c='red')

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)

ax.view_init(45,15)


plt.show()