
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Define problem parameters
L = 1.0
T = 1.0
n = 1000  # number of grid points
m = 10  # number of eigenvalues to compute

# Define mass density function
def rho(x):
    return 1.3 - 0.5*np.sin(np.pi*x/L)

# Define grid
x = np.linspace(0, L, n+1)[:-1]
dx = x[1] - x[0]

# Compute differentiation matrix
D2 = np.zeros((n, n))
for i in range(1, n-1):
    D2[i,i-1:i+2] = [1, -2, 1]
D2[0,0:2] = [-2, 2]
D2[-1,-2:] = [2, -2]
D2 /= dx**2

# Compute mass density matrix
R = np.diag(rho(x))

# Compute eigenvalues and eigenvectors
A = D2 + np.diag((T/np.diag(R)) * np.ones(n))
eigenvalues, eigenvectors = np.linalg.eig(A)
sorted_indices = np.argsort(eigenvalues)[:m]
omega = np.sqrt(-eigenvalues[sorted_indices])
X = eigenvectors[:, sorted_indices]

print(omega[0:10])

# Plot eigenfunctions
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for i, ax in enumerate(axs):
    ax.plot(x, X[:, i])
    ax.set_xlabel('x')
    ax.set_ylabel(f'X_{i+1}(x)')
    ax.set_title(f'Eigenfunction {i+1} (omega={omega[i]:.3f})')
plt.tight_layout()

plt.show()

# Compute waveforms for n = 1, 5, 9
n_values = [1, 5, 9]
fig, axs = plt.subplots(len(n_values), 1, figsize=(8, 12))
for i, n in enumerate(n_values):
    waveform = X[:, i]
    axs[i].plot(x, waveform)
    axs[i].set_xlabel('x')
    axs[i].set_ylabel(f'X_{n}(x)')
    axs[i].set_title(f'Waveform for eigenfunction {n} (omega={omega[i]:.3f})')
plt.tight_layout()

plt.show()
