import numpy as np
import matplotlib.pyplot as plt

# Parameters
# r = 5 # radius of sphere
N = 2000 # number of nucleons
# density at center of nucleus
# R = 5 # radius parameter of Fermi distribution
a = 0.5 # diffuseness parameter of Fermi distribution
A = 208             # total number of nucleons in nucleus
R = (A**(1/3))*1.2  # radius of nucleus in fm
a = 0.5             # parameter of Fermi distribution in fm
Na = 6.022e23

# generate samples from Fermi distribution
r = np.random.rand(N)          # sample r from 0 to 10 fm
rho_0 = 1/((-a**2) * ((-1/10) * np.pi**2 - (R / a)**2))

def density(r):
    rho_r = rho_0/(1 + np.exp((r - R)/ a))
    return rho_r

# Generate random values of r using Fermi distribution
# r_samples = np.random.rand(N)
# r_samples = R * np.log((N / r_samples - 1)) + R
r_samples = R*np.log((rho_0/density(r))-1) + R

# Generate random values of theta and phi
theta_samples = np.arccos(2 * np.random.rand(N) - 1)
phi_samples = 2 * np.pi * np.random.rand(N)

# Convert to Cartesian coordinates
x_samples = r_samples * np.sin(theta_samples) * np.cos(phi_samples)
y_samples = r_samples * np.sin(theta_samples) * np.sin(phi_samples)
z_samples = r_samples * np.cos(theta_samples)

# Check isotropy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.hist(theta_samples, bins=50, density=True)
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel('Density')
ax2.hist(phi_samples, bins=50, density=True)
ax2.set_xlabel(r'$\phi$')
ax2.set_ylabel('Density')
plt.show()

# Check radial distribution
plt.hist(r_samples, bins=50, density=True)
plt.xlabel('r')
plt.ylabel('Density')
plt.show()

# Plot 3D distribution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_samples, y_samples, z_samples, s=2)
ax.set_xlabel('x(fm)')
ax.set_ylabel('y(fm)')
ax.set_zlabel('z(fm)')
plt.show()



R = 5.0 # radius of sphere in fm
a = 0.5 # parameter of Fermi distribution in fm
rho0 = 1.0 # density at center of sphere in fm^-3
N = 1000 # total number of nucleons

# generate samples of r from given distribution
r_samples = np.random.rand(N) * 20 - 10 # sample r from -10 to 10 fm
p_samples = np.random.rand(N) # sample p from 0 to 1
rho_samples = rho0 / (1 + np.exp((r_samples - R) / a))
r_samples = np.where(p_samples < rho_samples / rho0, r_samples, 0)

# convert to spherical coordinates
theta_samples = np.arccos(2 * np.random.rand(N) - 1)
phi_samples = np.random.rand(N) * 2 * np.pi

# convert to cartesian coordinates
x_samples = r_samples * np.sin(theta_samples) * np.cos(phi_samples)
y_samples = r_samples * np.sin(theta_samples) * np.sin(phi_samples)
z_samples = r_samples * np.cos(theta_samples)

# plot histograms of angles and radius
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].hist(theta_samples, bins=30)
ax[0].set_xlabel(r'$\theta$ (rad)')
ax[0].set_ylabel('Counts')
ax[1].hist(phi_samples, bins=30)
ax[1].set_xlabel(r'$\phi$ (rad)')
ax[1].set_ylabel('Counts')
plt.show()

fig, ax = plt.subplots()
ax.hist(np.sqrt(x_samples**2 + y_samples**2 + z_samples**2), bins=30)
ax.set_xlabel('r (fm)')
ax.set_ylabel('Counts')
plt.show()

# Plot 3D distribution
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_samples, y_samples, z_samples, s=2)
ax.set_xlabel('x(fm)')
ax.set_ylabel('y(fm)')
ax.set_zlabel('z(fm)')
plt.show()

#################################################################################################
#2c) bx and by 
#################################################################################################

bmax = 10.0 # maximum impact parameter in fm
N = 1000 # total number of collisions

# generate samples of b and phi
b_samples = np.random.rand(N) * bmax
phi_samples = np.random.rand(N) * 2 * np.pi

# convert to cartesian coordinates
bx_samples = b_samples * np.cos(phi_samples)
by_samples = b_samples * np.sin(phi_samples)

# plot histogram of impact parameter magnitudes
fig, ax = plt.subplots()
ax.hist(b_samples, bins=30)
ax.set_xlabel('b (fm)')
ax.set_ylabel('Counts')
plt.show()

# plot scatter plot of impact parameter vectors
fig, ax = plt.subplots()
ax.scatter(bx_samples, by_samples, s=1)
ax.set_xlabel('b_x (fm)')
ax.set_ylabel('b_y (fm)')
plt.show()


