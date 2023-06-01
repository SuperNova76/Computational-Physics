import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from matplotlib.patches import Polygon
import matplotlib.tri as tri

nodepattern = [[0.0,  1, 0.0], [0.1, 8, 0.2], [0.2, 12, 0.1], [0.3, 18, 0.2], 
               [0.4, 24, 0.3], [0.52, 30, 0.4], [0.65, 30, 0.3], [0.8, 25, 0.1], 
               [1.0, 46, 0.3] ]

# build the nodes
nodes = np.zeros( [0,2] )
for r,n,o in nodepattern:
    offset = 2*np.pi/n * o
    angles = np.linspace(offset, 2*np.pi+ offset, n, False)

    nodes = np.concatenate( (nodes, np.array((r*np.cos(angles), r*np.sin(angles))).T) )

plt.plot(nodes[:, 0], nodes[:, 1], "ro")

# number of inner nodes: all nodes minus nodes on (outer) boundary
n_inner_nodes = len(nodes) - nodepattern[-1][1]

#################################################################################################
# Mashgrid Triangles
#################################################################################################

triang = tri.Triangulation(nodes[:,0], nodes[:,1])

plt.figure("mesh")
plt.plot(nodes[:, 0], nodes[:, 1], "ro")
plt.triplot(triang)

triangle = nodes[triang.get_masked_triangles()[82]]
pprint(triangle)

plt.show()

#################################################################################################

#################################################################################################

def tri_grad(t, f):
    xa, ya, xb, yb, xc, yc = t.reshape(6)
    fa, fb, fc = f

    if (yc == ya):
        # avoid division by zero by reordering the corner points
        CAB = np.array([[xc, yc], [xa, ya], [xb, yb]])
        return tri_grad(CAB, np.array([fc, fa, fb]))

    c1 = (fc-fa)/(yc-ya) - (fb-fa)/(xb-xa)*(xc-xa)/(yc-ya)
    c2 = 1 - (yb-ya)/(xb-xa)*(xc-xa)/(yc-ya)

    b = c1 / c2
    a = (fb-fa)/(xb-xa) - b*(yb-ya)/(xb-xa)

    return np.array([a, b])


for t in [np.array(((0, 0), (0, 1), (1, 0))), triangle]:
    print(
        f"A({t[0,0]},{t[0,1]}) B({t[1,0]},{t[1,1]}) C({t[2,0]},{t[2,1]})")
    for i in range(3):
        f = np.array([int(i==j) for j in range(3)])
        print( f"  gradient for f(A,B,C) =", f, " --> ", tri_grad(t, f))

#################################################################################################

#################################################################################################

def tri_area(t):
    # get corners
    A = t[0, :]
    B = t[1, :]
    C = t[2, :]
        
    # calculate sides
    a = np.linalg.norm(B-C)
    b = np.linalg.norm(A-C)
    c = np.linalg.norm(A-B)

    # Heron's formula
    s = (a+b+c) / 2.0
    return np.sqrt( s * (s-a) * (s-b) * (s-c) )

print("area =", tri_area(triangle))

#################################################################################################

#################################################################################################

# create the empty matrix
A = np.zeros([n_inner_nodes, n_inner_nodes])

# calculate the integrals for the coefficients on the diagonal
for i in range(n_inner_nodes):
    Acoeff = 0
    for t in triang.get_masked_triangles():
        if (i in t):
            # print( [ 1 if t==i else 0 ] )
            gr = tri_grad(nodes[t], np.array(t == i).astype(float))
            ar = tri_area(nodes[t])
            Acoeff += np.dot(gr, gr) * ar

    A[i][i] = Acoeff

# add the integrals for neighboring u_i, u_j
for i, j in triang.edges:
    if i < n_inner_nodes and j < n_inner_nodes:

        Acoeff = 0
        for t in triang.get_masked_triangles():
            if (i in t) and (j in t):
                # print "   ", t, "   ",
                # print (t==i), "   ", (t==j)

                gr1 = tri_grad(nodes[t], np.array(t == i).astype(float))
                gr2 = tri_grad(nodes[t], np.array(t == j).astype(float))
                ar = tri_area(nodes[t])
                Acoeff += np.dot(gr1, gr2) * ar

        A[i, j] = Acoeff
        A[j, i] = Acoeff

plt.matshow(A)

#################################################################################################
# Vector Components
#################################################################################################

b = np.zeros(n_inner_nodes)
for i in range(n_inner_nodes):

    for t in triang.get_masked_triangles():
        if (i in t):
            ar = tri_area(nodes[t])
            rm = np.mean(np.hypot(nodes[t][:,0],nodes[t][:,1]))

b[12] = 1
b[22] = -1

# plt.figure("charge density")
plt.tripcolor(nodes[:, 0], nodes[:, 1], np.concatenate(
    (b,  np.zeros(len(nodes)-n_inner_nodes))), shading='gouraud')


# solve equation for the inner points
phi_inner = np.linalg.solve(A, b)

# build solution including boundary points
phi = np.concatenate( (phi_inner, np.zeros( len(nodes)-n_inner_nodes )))

# plt.figure("potential")
plt.tripcolor(nodes[:, 0], nodes[:, 1], phi, shading='gouraud')