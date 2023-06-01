import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def solve_differential_equation(omega, L, T, num_points):
    h = L / (num_points - 1)  # Step size

    # Create the matrix for the finite difference approximation
    matrix_size = num_points
    matrix = np.zeros((matrix_size, matrix_size))
    rhs = np.zeros(matrix_size)

    # Fill the matrix and right-hand side
    for i in range(1, num_points - 1):
        x = i * h
        rho_val = 1.3 - 0.5 * np.sin(np.pi * x / L)
        matrix[i, i] = 2.0 + h**2 * omega**2 * rho_val / T
        matrix[i, i - 1] = -1.0
        matrix[i, i + 1] = -1.0

    # Set the boundary conditions
    matrix[0, 0] = 1.0
    matrix[-1, -1] = 1.0

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = la.eig(matrix)
    sorted_indices = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[sorted_indices].real
    eigenvectors = eigenvectors[:, sorted_indices].real

    return eigenvalues, eigenvectors

# Define the parameters
omega = 2.0  # Frequency
L = 10.0  # Length of the string
T = 1.0  # Tension
num_points = 100

# Solve the differential equation numerically
eigenvalues, eigenfunctions = solve_differential_equation(omega, L, T, num_points)


# Determine the first 10 eigenvalues for the given mass density function (1.3 - 0.5*sin(pi*x/L))
eigenvalues_general = solve_differential_equation(omega, L, T, num_points)
print("First 10 eigenvalues (with mass density 1.3 - 0.5*sin(pi*x/L):")
for i, eigenvalue in enumerate(eigenvalues_general):
    print(f"Eigenvalue {i+1}: {eigenvalue}")

# Determine the first 10 eigenvalues for the homogeneous string (mass density 1)
eigenvalues_homogeneous = solve_differential_equation(omega, L, T, num_points, homogeneous=True)
print("First 10 eigenvalues (homogeneous string with mass density 1):")
for i, eigenvalue in enumerate(eigenvalues_homogeneous):
    print(f"Eigenvalue {i+1}: {eigenvalue}")

# Define the indices of the eigenfunctions to plot
indices = [1, 5, 9]

# Plot the eigenfunctions for n = 1, 5, 9
x_values = np.linspace(0, L, num_points)
plt.figure(figsize=(8, 6))
for index in indices:
    eigenfunction = eigenfunctions[:, index]
    eigenfunction /= np.max(np.abs(eigenfunction))  # Normalize the eigenfunction
    plt.plot(x_values, eigenfunction, label=f"Eigenfunction {index}")

plt.xlabel('x')
plt.ylabel('X(x)')
plt.title('Eigenfunctions of the Differential Equation')
plt.legend()
plt.grid(True)
plt.show()
