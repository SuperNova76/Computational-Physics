import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# Define the parameters and grid
L = 1.0  # Length of the domain
N = 100  # Number of grid points
x = np.linspace(0, L, N)
dx = L / (N - 1)  # Grid spacing
h = dx

# Define the material properties, T and rho(x)
T = 1  # Tension
rho = 1.3 - 0.5 * np.sin(np.pi * x / L)  # Density function

# Construct the finite difference matrix
A = np.zeros((N, N))

# Iterate over the grid points
# Set up matrix A
A = np.zeros((N, N))
for i in range(N):
    x = i * dx  # x-coordinate of the current grid point
    # rho_i = rho(x)  # Density at the current grid point

    A[i, i] = -2.0/ dx **2
    if i > 0:
        A[i, i-1] = 1.0/ dx **2
    if i < N-1:
        A[i, i+1] = 1.0/ dx**2

# Modify the boundary conditions
# A[0, 0] = 1.0
# A[N-1, N-1] = 1.0
# A[0, 1] = 0.0
# A[N-1, N-2] = 0.0

A[0, 0] = 1.0
A[0, 1] = 0.0
A[-1, -1] = 1.0
A[-1, -2] = 0.0

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(A)

# Sort the eigenvalues in ascending order
sorted_indices = np.argsort(eigenvalues)
eigenvalues = eigenvalues[sorted_indices]

# Print the first 10 eigenvalues
print("First 10 eigenvalues:")
for i in range(10):
    omega = np.sqrt((rho[i]/T) * np.real(-eigenvalues[i]))
    print(f"Eigenvalue {i+1}: {omega:.4f}")

# Plot the eigenfunctions for n=1, 5, and 9
plt.figure(figsize=(8, 6))
x_values = np.linspace(0, L, N)

for n in [1, 5, 9]:
    eigenvalue_index = n - 1
    omega = np.sqrt((rho[i]/T) * np.real(-eigenvalues[eigenvalue_index]))
    X = np.real(eigenvectors[:, eigenvalue_index])

    # Normalize the eigenfunction
    X /= np.max(np.abs(X))

    plt.plot(x_values, X, label=f"n={n}, omega={omega:.4f}")

plt.xlabel('x')
plt.ylabel('X(x)')
plt.title('Eigenfunctions')
plt.legend()
plt.grid(True)
plt.show()
