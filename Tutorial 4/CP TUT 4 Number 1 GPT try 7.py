import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

# Set up the grid
xmin, xmax, ymin, ymax = 0, 10, 0, 10
x = np.linspace(xmin, xmax, num=101)
y = np.linspace(ymin, ymax, num=101)
xx, yy = np.meshgrid(x, y)
dx = 1/10

# Set up the point charge
q = 1
xq, yq = 6, 5

# Set up the electrode
xele = 4  # x-coordinate of the electrode
yele_min, yele_max = 2, 8  # y-coordinate range of the electrode
ele_thickness = 0.1  # Electrode thickness in cm

# Calculate the distance from each point on the grid to the point charge
r = np.sqrt((xx - xq) ** 2 + (yy - yq) ** 2)

# Set up the boundary conditions
V = np.zeros_like(xx)
V[0, :] = 0  # top boundary
V[-1, :] = 0  # bottom boundary
V[:, 0] = 0  # left boundary
V[:, -1] = 0  # right boundary

# Set up the matrix for the Poisson equation
N = len(x)
A = np.zeros((N ** 2, N ** 2))
b = np.zeros(N ** 2)

epsilon0 =  8.854e-10  # Vacuum permittivity in farads per centimeter

for i in range(N):
    for j in range(N):
        k = i * N + j
        if r[i, j] == 0:
            A[k, k] = 1
        else:
            if (
                yele_min <= yy[i, j] <= yele_max
                and xele - ele_thickness / 2 <= xx[i, j] <= xele + ele_thickness / 2
            ):
                # Electrode region
                A[k, k] = 1
                b[k] = 0  # Potential is zero on the electrode
            else:
                A[k, k] = -4
                if i > 0:
                    A[k, k - N] = 1
                if i < N - 1:
                    A[k, k + N] = 1
                if j > 0:
                    A[k, k - 1] = 1
                if j < N - 1:
                    A[k, k + 1] = 1
                rho = q / (r[i, j] ** 2)  # Charge density expression
                b[k] = -rho * (dx**2) / (4 * epsilon0)  # Right-hand side of the Poisson equation

# Solve for the potential
V_inner = np.linalg.solve(A, b)
V = V_inner.reshape((N, N))

# Calculate the electric field
Ex, Ey = np.gradient(-V, x, y)

# Calculate the magnitude and angle of the electric field
magnitude = np.sqrt(Ex**2 + Ey**2)
angle = np.arctan2(Ey, Ex)

# Coordinates of the points
points = [(5, 5), (9, 5), (9, 9), (4+0.05 ,2), (4-0.05 ,2), (4, 2), (4, 8)]

# Find the indices of the lattice points with minimum distances
indices = []
for point in points:
    x, y = point
    distances = np.sqrt((xx - x)**2 + (yy - y)**2)
    min_distance = np.min(distances)
    min_indices = np.where(distances == min_distance)
    indices.append((min_indices[0][0], min_indices[1][0]))

# Print the indices
for i, point in enumerate(points):
    print(f"Indices of the lattice point closest to {point}: {indices[i]}")


# Coordinates of the points
point1 = indices[0]  # (5, 5)
point2 = indices[1]  # (9, 5)
point3 = indices[2]  # (9, 9)
point4 = indices[3]  
point5 = indices[4]  

# Convert the coordinates to indices
x1, y1 = point1
x2, y2 = point2
x3, y3 = point3

# Print the potential at the points
print(f"Potential at (5, 5): {V[x1, y1]}")
print(f"Potential at (9, 5): {V[x2, y2]}")
print(f"Potential at (9, 9): {V[x3, y3]}")

for i, point in enumerate(points):
    x_index, y_index = indices[i]
    electric_field_magnitude = magnitude[x_index, y_index]
    electric_field_angle = np.degrees(angle[x_index, y_index])
    print(f"Electric Field at {point}: Magnitude = {electric_field_magnitude:.2f}, Angle = {electric_field_angle:.2f} degrees")


# Plot the potential
# plt.imshow(V, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="coolwarm")
plt.contourf(xx, yy, V, 50,cmap = 'coolwarm')
plt.colorbar()
plt.xlabel("$x(cm)$")
plt.ylabel("$y(cm)$")
plt.title("Potential")
plt.show()

# Visualize the vector field
plt.quiver(xx, yy, Ex, Ey, magnitude, cmap='viridis')
plt.xlabel('$x(cm)$')
plt.ylabel('$y(cm)$')
plt.title('Electric Field')
plt.colorbar(label='Electric Field Magnitude')
plt.show()

def plotE(psi, n):
    v, u = np.gradient(psi) #grad of E field
    
    x = np.linspace(0, 10, n) # x-axis
    fig, axarr = plt.subplots()

    im1 = axarr.contourf(x, x, psi, 50,cmap = 'coolwarm') #basically makes x,y grid
    axarr.streamplot(x, x, -u, -v, density=1.4, color="black") #plots the E field vectors
    fig.colorbar(im1, orientation='vertical', ax=axarr,label=(r"$\phi(V)$")) #adds colorbar
    axarr.set_xlabel("$x(cm)$")
    axarr.set_ylabel("$y(cm)$")

plotE(V,N)
plt.show()


lam = []
E_perp_above = Ex[21:80,40] #electric field to the right of the line
E_perp_below = Ex[21:80,38] #electric field to the left of the line


for i in range(len(E_perp_above)):
    a = (E_perp_above[i] - E_perp_below[i])*epsilon0 #applying boundary condition
    lam.append(a)
    
    

area = ele_thickness*6    
y = np.linspace(2,8,59) #the corresponding y points where the line charge density was calculated

# plt.figure(10)
plt.plot(y,np.array(lam),color = 'orangered')
plt.grid()
plt.xlabel(r'$y(cm)$')
plt.ylabel(r'$\lambda(C/cm)$')
plt.title('Charge Distribution')
plt.show()


dy = y[1] - y[0] #spacing of y
charge = simps(lam,y,dy,axis = -1) #using simpsons rule to calculate integral of line charge density 
print('The total charge is =', charge, 'C')
print('\n')
