import numpy as np

# Define the parameters
cell_n = 500
l = 5
dx = l/cell_n
C = 100e-12 * dx
Ri = 50
G = 1e9 * dx
R = 30e-6 * dx
L = 270e-9 * dx
Rl = 100
f = 100e6
U0 = 100
omega = 2*(np.pi)*f

###########################################################################
###         Voltages of the different components of the circuit         ###
###########################################################################
LI = L*omega*(1j)
CI = 1/(omega*C*(1j))

# Impendence for the components connected in series #
LR = LI + R

# Impendence for the components connected in parallel #
GC = 1/((1/G) + 1/(CI))

##Circuits
circuit = ['100ohm','short','open']
##storing stuff
HC = []
CC = []
CNV = []
CV = []

# Define the function to generate the matrix
def setMatrix(RL):
    # create an empty matrix
    V = np.zeros(cell_n, dtype = complex)
    V[0] = -U0
    matrix = np.zeros(((1*cell_n+1), (1*cell_n+1)), dtype=complex)

    # Fill in the matrix
    for i in range (0, (1*cell_n+1)):
        row = matrix[i]

        if i==0:
            row[i] = -Ri-R-(L*omega*(1j))
            row[i+1] = -G

        if i==(3*cell_n):
            row[i-1] = 1/(omega*C*(1j))

            if (RL==-1):
                row[i] = 0
            else:
                row[i] = -RL
        
        elif i%3 == 1:
            row[i-1] = 1
            row[i] = -1
            row[i+1] = -1
            row[1+2] = -1

        elif i%3 == 2:
            row[i-1] = G
            row[i] = -1/(omega*C*(1j))

        elif i%3 == 0:
            row[i-1] = 1/(omega*C*(1j))
            row[i] = -R-(L*omega*(1j))
            row[i+1] = -G

        matrix[i] = row

    return matrix


V = np.zeros(cell_n+1, dtype = complex) # the b in our equation Rx = V
V[0] = -U0
Res_open = setMatrix(10**40)
Res_short = setMatrix(0)
Res_100 = setMatrix(100)

Res = np.array([Res_open, Res_short, Res_100])

###    Determining the split of the current in proportion to Capacitor C and Resistor G     ###
current = []
for i in range(len(circuit)):
    current.append(np.linalg.solve(Res[i], V))

print(current)
I = [[],[],[]]
i = 0
ratio_G = G/(G + GC)
ratio_C = C/(G + GC)
for a in range(len(circuit)):
    for i in range(cell_n):
        if i != (cell_n - 1):
            Iv = current[a][i] - current[a][i + 1]
            IG = Iv*ratio_G
            IC = Iv*ratio_C
            I.append(current[a][i])
            I.append(IG)
            I.append(IC)
        else:
            I.append(current[a][i])
print(np.array(I).shape)
Nv = []
Av = [U0]
i = 0
for i in range(cell_n):
        if i == 0:
            V = U0 - current[i]*(Ri + LR)
            Nv.append(V)
            Av.append(V)
        elif i == (cell_n - 1):
            pass
        else:
            V = Nv[i-1] - current[i]*LR
            Nv.append(V)
            Av.append(V)
    
        HC.append(current)
        CC.append(I) 
        CNV.append(Nv)
        CV.append(Av)

