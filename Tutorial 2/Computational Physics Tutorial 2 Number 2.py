import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int
from scipy.integrate import quad

A = 208             # total number of nucleons in nucleus
R = (A**(1/3))*1.2  # radius of nucleus in fm
a = 0.5             # parameter of Fermi distribution in fm
Na = 6.022e23
N = 2000
sigma = 7.2         # fm^2

#################################################################################################
#2a) Getting the expected number of nucleons some distance r and plotting the result
#################################################################################################

# generate samples from Fermi distribution
# r = np.random.uniform(0,14, 10000)          # sample r from 0 to 14 fm
r = np.random.rand(N) * 14
# N = 208             

# calculate normalization constant Norm_Constant
def den_f(r,R,a):
    return (4 *np.pi* r**2) / (1 + np.exp((r - R)/ a))
rho_0 = A / (int.quad(den_f, 0, np.inf, args=(R,a))[0])
print(rho_0)

N_r = (rho_0 *4*np.pi*r**2 / (1 + np.exp((r - R)/ a))) * Na/A

plt.scatter(r,N_r, marker = '.', s = 0.8, color='r')
plt.xlabel('r(fm)')
plt.ylabel('N(r)')
plt.legend()
plt.show()

#################################################################################################
#2b) Checking the isotropy and radial distribution of the nucleons 
#################################################################################################

# generate samples from Fermi distribution
r1 = np.random.rand(N)      # sample r in fm
print(r1)

def density(r):
    rho_r = rho_0 / (1 + np.exp((r - R)/ a))
    return rho_r

# Generate random values of r using Fermi distribution
r_samples = R *np.log((rho_0 / density(r1)) - 1) + R

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

#################################################################################################
#2c) Sampling the impact parameters b, assuming bx and by are uniformly distributed  
#################################################################################################

# def density_func(r, R, a, rho_0):
#     return rho_0 / (1 + np.exp((r - R)/a))

# def max_impact_parameter(R, a, rho_0, A1, A2):
#     def integrand(z, b):
#         return density_func(np.sqrt(b**2 + z**2), R, a, rho_0)
#     integral, error = quad(integrand, -np.inf, np.inf, args=(0,))
#     return np.sqrt((A1 - A2) / integral)

# # R = 7  # lead nucleus radius in fm
# # a = 0.54  # thickness parameter in fm
# # rho_0 = 0.17  # normalization factor in fm^-3
# A1 = 208  # number of nucleons in lead-208 nucleus
# A2 = 208  # number of nucleons in another lead-208 nucleus

# b_max = max_impact_parameter(R, a, rho_0, A1, A2)

# print(f"Maximum impact parameter: {b_max:.2f} fm")


bmax = 2.5 * R # maximum impact parameter in fm
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
ax.set_xlabel('b(fm)')
ax.set_ylabel('Counts')
plt.show()

# plot scatter plot of impact parameter vectors
fig, ax = plt.subplots()
ax.scatter(bx_samples, by_samples, s=1)
ax.set_xlabel('$b_x$(fm)')
ax.set_ylabel('$b_y$(fm)')
plt.show()

#################################################################################################
#2d) Determining N_part and N_coll and plotting
#################################################################################################


# Define the thickness function
def thickness_function(r, R, a, rho_0):
    return rho_0 / (1 + np.exp((r - R)/a))

# Parameters for lead-lead collisions
A1 = 208
A2 = 208

# Generate 10000 random impact parameters in the range [0, b_max]
np.random.seed(42)
b_max = 2 * R
b = np.random.uniform(0, b_max, size=10000)

# Calculate Npart and Ncoll for each collision
Npart = np.zeros_like(b)
Ncoll = np.zeros_like(b)
d = []
for i in range(len(b)):
    # Calculate the maximum distance of closest approach
    # d.append(np.sqrt(b[i]**2 + (R**2 - (b[i]/2)**2)))
    
    # Calculate the thickness functions for the two nuclei
    TA = np.zeros_like(b)
    TB = np.zeros_like(b)
    for j in range(len(b)):
        TA[j] = quad(thickness_function, max(0, np.sqrt(b[j]**2 + (R**2 - (b[j]/2)**2)) - R), min(np.sqrt(b[j]**2 + (R**2 - (b[j]/2)**2))+ R, 2*R), args=(R, a, rho_0))[0]
        TB[j] = quad(thickness_function, max(0, np.sqrt(b[j]**2 + (R**2 - (b[j]/2)**2)) - R), min(np.sqrt(b[j]**2 + (R**2 - (b[j]/2)**2)) + R, 2*R), args=(R, a, rho_0))[0]
    
    # Calculate Npart and Ncoll
    Npart[i] = np.sum((TA > 0) & (TB > 0))
    Ncoll[i] = np.sum((TA > 0) | (TB > 0)) - Npart[i]
    
# Plot Npart vs b
plt.figure(figsize=(8, 6))
plt.scatter(b, Npart, s=10)
plt.xlabel('Impact parameter (fm)')
plt.ylabel('Number of participants')
plt.title('Number of participants vs impact parameter')
plt.show()

# Plot Ncoll vs b
plt.figure(figsize=(8, 6))
plt.scatter(b, Ncoll, s=10)
plt.xlabel('Impact parameter (fm)')
plt.ylabel('Number of binary collisions')
plt.title('Number of binary collisions vs impact parameter')
plt.show()

# Plot Ncoll vs Npart
plt.figure(figsize=(8, 6))
plt.scatter(Npart, Ncoll, s=10)
plt.xlabel('Number of participants')
plt.ylabel('Number of binary collisions')
plt.title('Number of binary collisions vs number of participants')
plt.show()

# Histogram Npart and Ncoll
plt.figure(figsize=(8, 6))
plt.hist(Npart, bins=30, log=True)
plt.xlabel('Number of participants')
plt.ylabel('Frequency')
plt.title('Distribution of number of participants')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(Ncoll, bins=30, log=True)
plt.xlabel('Number of binary collisions')
plt.ylabel('Frequency')
plt.title('Distribution of number of binary collisions')
plt.show()