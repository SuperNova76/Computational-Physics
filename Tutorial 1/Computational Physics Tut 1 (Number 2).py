import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


cell_n = 500           # cells
l = 5                  # m
dx = l/cell_n          # segmenting the circuit 

C = 100e-12 * dx       # F
Ri =  50 + 0j          # Ohm
G = 1e9 * dx           # Ohm
R = 30e-6 * dx         # Ohm
L = 270e-9 * dx        # H
Rl = 100               # Ohm
U0 = 100               # V
tau = 5e-9             # Decay time

f = 100e6
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

# Types of curcuits #
circuit = ['Open','Short','100 Ohm']
short_R = 0
open_R = 10**50


## An empty list to append the voltages and currents for each curcuit configuration
Cable_I = []
CC = []
CNV = []
Cable_V = []


###########################################################################
### Matrix Generation to be generated to encode the system of equations ###
###########################################################################

for a in circuit:
    V = np.zeros(cell_n, dtype = complex) # the b in our equation Rx = V
    V[0] = U0
    Res = []
    for i in range(cell_n):         ##  creating an array with the currents(unknown) we looking at in the range, from 0 to 499
        if i == 0:                  ##  First cell working with current I_0
            row = np.zeros(cell_n, dtype = complex)
            LRcur = Ri + LR + GC
            GCcur = -GC
            row[i] = LRcur          ##  first entry going around loop in first cell with currents I_0
            row[i + 1] = GCcur      ##  Next with currents I_1

        elif i == cell_n - 1:
            row = np.zeros(cell_n, dtype = complex)
            if a == 'Open':         ##  End of circuit with open circuit imples Rl = really big
                LRcur = GC + open_R
                GCcur = -GC 
                row[i] = LRcur
                row[i - 1] = GCcur

            elif a == 'Short':      ##  End of circuit with short circuit imples Rl = 0
                LRcur = GC + short_R
                GCcur = -GC
                row[i] = LRcur
                row[i - 1] = GCcur
            
            elif a == '100 Ohm':    ##  End of circuit with terminal resistor with resistance Rl = 100
                LRcur = GC + Rl
                GCcur = -GC
                row[i] = LRcur
                row[i - 1] = GCcur

        else:
            row = np.zeros(cell_n, dtype = complex)
            LRcur = LR + 2*GC
            GCcur = -GC
            row[i] = LRcur
            row[i - 1] = GCcur
            row[i + 1] = GCcur
        Res.append(row)

    current = np.linalg.solve(Res,V)
    print(current.shape)

    Nv = []
    Av = [U0]
    i = 0
    for i in range(cell_n):
            if i == 0:
                V1 = U0 - current[i]*(Ri + LR)
                Nv.append(V1)
                Av.append(V1)
            elif i == (cell_n - 1):
                pass
            else:
                V1 = Nv[i-1] - current[i]*LR
                Nv.append(V1)
                Av.append(V1)
    
    Cable_I.append(current)
    Cable_V.append(Av)

### Calculating the Voltage between two points to determine the Voltage between wires in the
# middle and at the end of the cable ###                
i=0
mid_V = []
end_V = []
for i in range(len(Cable_V)):    ## Cable_V is the cable voltage for all 3 circuit configurations and 
    mV = Cable_V[i][250]         ## since we have 500 cells, the middle will have index 250 
    eV = Cable_V[i][499]         ## and the end will have index 499
    mid_V.append(np.abs(mV))
    end_V.append(np.abs(eV))

a = ['Open Circuit', 'Short Circuit', '100 Ohms']
data = {
    'Circuit Configuration': a,
    'middle voltage(V)': mid_V,
    'end voltage(V)': end_V}
   
df = pd.DataFrame(data)
print(df)

plt.plot([i for i in range(cell_n)], np.absolute(Cable_V[0]), label="Open Circuit")
plt.plot([i for i in range(cell_n)], np.absolute(Cable_V[1]), label="Short Circuit")
plt.plot([i for i in range(cell_n)], np.absolute(Cable_V[2]), label="$R_l = 100\Omega$")
plt.xlabel("length (mm)")
plt.ylabel("Voltage(V)")
plt.legend(loc = 'best')
plt.title("Voltage as a function of length")
plt.show()


V = 0
##Fourier Transform of input signal

t = np.linspace(0,80e-9,1000)      ## defining time for the input signal
V = U0*np.exp(-t/tau)               ## The equation describing the voltage as a function of time

ffts = fft(V)                       

Time = t[1] - t[0]                     ## sampling interval 
n = len(t)                             ## number of samples

f = np.linspace(0, 1 / Time, n)        ## frequency 

plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.title('Fourier transform of the exponential input voltage')
plt.plot(f[:n//2], np.abs(ffts)[:n//2]*1/n)  # 1/n is a normalization factor
plt.show()

## Calculating the measured voltage across the termination resistor
VRL = (np.abs(Cable_V[2][499])*Rl)*np.exp(-t[999]/tau)
print("The measured voltage across the termination resistor is: ", np.real(VRL), "V")

## Plotting the Exponential Voltage as a function of time ##
plt.plot(t,V)
plt.ylabel("Voltage(V)")
plt.xlabel("time(s)")
plt.title('Exponential Voltage as a Function of time')
plt.show()

# Calculating the signal speed
w = 1/np.sqrt(L*C)
Lambda = 0.01 #m 
wave_speed = Lambda*w/(2*np.pi)
print('The signal speed is: ', wave_speed, "meters per second")