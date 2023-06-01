import numpy as np
from scipy.integrate import tplquad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define constants
mu0 = 4*np.pi*1e-7  # vacuum permeability

# Define the geometry of the solenoid
R1 = 0.02  # inner radius
R2 = 0.05  # outer radius
L = 0.06   # length

# Define the number of elements along each direction
nr = 20   # number of elements in radial direction
nz = 60   # number of elements in axial direction

# Define the current density
J = 1.0  # current density

# Define the basis functions
def basis(r, z):
    return np.sin(np.pi*r/R2) * np.sin(np.pi*z/L)

# Define the test function
def test(r, z):
    return basis(r, z)

# Define the integrand for the radial component of the magnetic field
def int_Br(r, phi, z):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    A = np.sum(np.outer(basis(r, z), basis(r, z)), axis=1)
    B = np.sum(np.outer(basis(r, z), np.gradient(basis(r, z), r)), axis=1)
    return mu0*J/2 * np.sum(B * np.cos(phi) - A * np.sin(phi))

# Define the integrand for the axial component of the magnetic field
def int_Bz(r, phi, z):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    A = np.sum(np.outer(basis(r, z), basis(r, z)), axis=1)
    B = np.sum(np.outer(basis(r, z), np.gradient(basis(r, z), z)), axis=1)
    return mu0*J/2 * np.sum(B)

# Define the mesh
r = np.concatenate((np.linspace(-R2, -R1, nr//2), np.linspace(R1, R2, nr//2)))
z = np.linspace(0, L, nz)
dr = (R2 - R1) / (nr - 1)
dz = L / (nz - 1)

# Compute the magnetic field
Br = np.zeros((nr, nz))
Bz = np.zeros((nr, nz))

for j in range(nz):
    R = r.reshape(-1, 1)
    Br[:, j] = tplquad(int_Br, 0, 2*np.pi, lambda x: R1, lambda x: R2, lambda x, y: z[j]-dz/2, lambda x, y: z[j]+dz/2)[0] / (R[:, 0] * dz)
    Bz[:, j] = tplquad(int_Bz, 0, 2*np.pi, lambda x: R1, lambda x: R2, lambda x, y: z[j]-dz/2, lambda x, y: z[j]+dz/2)[0] / (R[:, 0] * dr)

B = np.sqrt(Br**2 + Bz**2)
Bx = Br / B
By = np.zeros((nr, nz))
Bz = Bz / B

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Z = np.meshgrid(r, z)

ax.quiver(X, np.zeros((nz, nr)), Z, Bx.T, By.T, Bz.T, length=0.02, normalize=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

