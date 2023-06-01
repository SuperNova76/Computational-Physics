import numpy as np
import pandas as pd
import numpy.random as rd
import matplotlib.pyplot as plt

L = 1.0     # m
T = 1.0     # N
N = 500
n = 10      # No of eigenvalues to be returned.
const = (L**2)/(25*T)
b = np.zeros(N)
B = []
D2 = [] 
x = np.linspace(0, L, N+1) [:-1]
dx = x[1] - x[0]
print(dx)

# making the matrix A for discritization
for i in range(N):
    B.append(1)
    if i == 0:
        vect = np.zeros(N)
        vect[i] = -2/dx**2
        vect[i+1] = 1/dx**2
    elif i == N-1:
        vect = np.zeros(N)
        vect[i] = -2/dx**2
        vect[i-1] = 1/dx**2
    else:
        vect = np.zeros(N)
        vect[i] = -2/dx**2
        vect[i-1] = 1/dx**2
        vect[i+1] = 1/dx**2
    D2.append(vect)

# The density function 
rho_1 = 1
def rho(x):
    return 1.3 - 0.5*np.sin(np.pi*x/L)

rho_M = np.diag(rho(x))

# Solving the system and find the eigenvalues and eigenfunctions 
A = D2 + np.diag((T/np.diag(rho_M)) * np.ones(N))
eigenvalues, eigenvectors = np.linalg.eig(A)
sorted_indices = np.argsort(eigenvalues)[:n]
omega = np.sqrt(-eigenvalues[sorted_indices])
X = eigenvectors[:, sorted_indices]
print("The first 10 values for $\omega$")


n_v = [1, 5, 9]
n_va = [1, 5, 9]

colour = ['k', 'b', 'r']
for i, j, col in zip(n_v, n_v, colour):
    waveform = X[:, i-1]
    plt.figure(j)
    plt.plot(x, waveform, color = col)
    plt.xlabel('$x$')
    plt.ylabel(f'$X_{j}(x)$')
    plt.title(f'Waveform for eigenvalue {j}')
plt.show()
