# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 18:39:52 2023

@author: SuperNova_
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import magpylib as magpy

### Constants ###
I = 1
L = 60e-3                                           # Am^-2
A = L*(15e-3)                                       # m^2
V = (np.pi*(50e-3)**2 - np.pi*(20e-3)**2)*L         # m^3
J = I*A/V                                           # A
mu_0 = 4*np.pi*1e-7                                 # Permeability of free space
c = mu_0*J/4*np.pi

### The magnitudes of the positions we are evaluating the magnetic field ###
r1 = np.sqrt((0.000001)**2+(0.000001)**2)
r2 = np.sqrt((0)**2 + (10e-3)**2)
r3 = np.sqrt((40e-3)**2 + (10e-3)**2)
r4 = np.sqrt((40e-3)**2 + (25e-3)**2)
pos = np.array([r1, r2, r3, r4])

### We define a function that to calculate the magnetic field at position r ###
def Bfield(r):    
    f = lambda theta, r, z: (c*np.sin(theta))/r
    B = integrate.tplquad(f, 0, 2*np.pi, 0.0000001, r, 0, L)
    return B

# ==============================================================
# trapezoidal rule
# def integrad(r, theta):
#       return np.sin(theta)/r

# def trapezoid(a, b, n):
#         """
# 	    trapezoidal rule numerical integral implementation
# 	    a: interval start
# 	    b: interval end
# 	    n: number of steps
# 	    return: numerical integral evaluation
# 	    """

#         # initialize result variable
#         res = 0

#         # calculate number of steps
#         h = (b - a) / n

#         # start at a
#         x = a

#         # sum 2*yi (1 ≤ i ≤ n-1)
#         for i in range(1, n):
#                 x += h
#                 res += integrad(x)
#                 res *= 2
#                 # evaluate function at a and b and add to final result
#                 res += integrad(a)
#                 res += integrad(b)
#         # divide h by 2 and multiply by bracketed term
#         return (h / 2) * res
	    

# print("The Magnetic Field at the center of the solenoid is:\n", trapezoid(0.00001, pos[0], 100), "T")


print("The Magnetic Field at the center of the solenoid is:\n", Bfield(pos[0]), "T")
print("The Magnetic Field at the inner boundary in the middle of the solenoid:\n", Bfield(pos[1]), "T")
print("The Magnetic Field 10mm outside the inner boundary of the solenoid is:\n", Bfield(pos[2]), "T")
print("The Magnetic Field 10mm outside the outer boundary of the solenoid is:\n", Bfield(pos[3]), "T")

### To calculate and visualize the magnetic field at all positions ###

# We create our hallow cylinder/solenoid.
c1 = magpy.magnet.Cylinder(magnetization = (0,0,500), dimension = (50,60)) # We define the dimensions of the outer cyliner in mm.
c2 = magpy.magnet.Cylinder(magnetization = (0,0,-500), dimension = (20,60)) # We define the dimentions of the inner cylinder in mm.
cyl = magpy.Collection(c1,c2)
magpy.show(cyl, backend='plotly', animation=3)

# We create positions for all space.
xs = np.linspace(-130,130,100)
zs = np.linspace(-130,130,100)
posis = [[x,0,z] for z in zs for x in xs]

### calculate and plot the amplitude of the magnetic field and the field lines ###
B = [cyl.getB(pos) for pos in posis]
Bs = np.array(B).reshape([100,100,3])       
Bamp = np.linalg.norm(Bs,axis=2)            # magnitude of the magnetic field
X,Z = np.meshgrid(xs,zs)
plt.pcolor(xs,zs,Bamp,cmap='jet',vmin=-200)
U,V = Bs[:,:,0], Bs[:,:,2]
plt.streamplot(X,Z,U,V,color='k',density=2)
plt.ylabel("z (mm)")
plt.xlabel("y (mm)")
plt.show()

