import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as int
from scipy.integrate import quad


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
ax.set_xlabel('b (fm)')
ax.set_ylabel('Counts')
plt.show()

# plot scatter plot of impact parameter vectors
fig, ax = plt.subplots()
ax.scatter(bx_samples, by_samples, s=1)
ax.set_xlabel('b_x (fm)')
ax.set_ylabel('b_y (fm)')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Define the thickness function
def thickness_function(r, R, a, rho_0):
    return rho_0 / (1 + np.exp((r - R)/a))

# Parameters for lead-lead collisions
R = 7  # fm
a = 0.54  # fm
rho_0 = 0.17  # fm^-3
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