# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:43:49 2023

@author: SuperNova_
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


### Constants ###
G = 6.67430e-11             #Gravitational Constant
Msun = 1988500e24           #kg
Mearth = 5.9724e24          #kg
Mmars = 0.64171e24          #kg
AU = 1.496e11               #m
rearth = 0.9832899*AU       #m
rmars = 1.3813334*AU        #m
vearth = 30287.4            #m/s
vmars = 26499.8             #m/s
a = 0.0
b = 1.0
rsun = 0.0

dt = 1000
tmax = 2*365.25*24*3600     #s
n_steps = int(tmax / dt)

# Define the equations of motion for the 2-body problem
m1 = Msun
m2 = Mearth
m3 = Mmars
x1 = rsun
x2 = rearth
x3 = rmars
y1 = 0.0
y2 = 0.0
y3 = 0.0
vx1 = 0.0
vx2 = 0.0 
vx3 = 0.0
vy1 = 0.0
vy2 = vearth
vy3 = vmars

N=int(n_steps)
alp = np.array([x1, x2, x3, y1, y2, y3, vx1, vx2, vx3, vy1, vy2, vy3])  ### an array that stores information about the initial state of the system
m=12

h=dt
t=a
### Empty arrays to be appended later during the implementation of Runge-Kutta
w=[0,0,0,0,0,0,0,0,0,0,0,0]
K1=[0,0,0,0,0,0,0,0,0,0,0,0]
K2=[0,0,0,0,0,0,0,0,0,0,0,0]
K3=[0,0,0,0,0,0,0,0,0,0,0,0]
K4=[0,0,0,0,0,0,0,0,0,0,0,0]

### Defining a function of the velocities and accelerations of the 3 bodies in out system.
def f(t,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12):
    ax1 = 0                                             # acceleration for the Sun, the Earth and Mars are too light to accelerate it.
    ay1 = 0

    ax2 = - G*m1*(u2-u1)/((u2-u1)**2+(u5-u4)**2)**(3/2) # acceleration for Earth
    ay2 = - G*m1*(u5-u4)/((u2-u1)**2+(u5-u4)**2)**(3/2)

    ax3 = -G*m1*(u3-u1)/((u3-u1)**2+(u6-u4)**2)**(3/2)  # acceleration for Mars
    ay3 = -G*m1*(u6-u4)/((u3-u1)**2+(u6-u4)**2)**(3/2) 
    
    return [u7, u8, u9, u10, u11, u12, ax1, ax2, ax3, ay1, ay2, ay3]

for j in range(m):
    w[j] = alp[j]

### arrays for the updated positions, velocities and accelerations for the 3 bodies
t_vals = [t]
x1_vals = [w[0]]
x2_vals = [w[1]]
x3_vals = [w[2]]
y1_vals = [w[3]]
y2_vals = [w[4]]
y3_vals = [w[5]]
vx1_vals = [w[6]]
vx2_vals = [w[7]]
vx3_vals = [w[8]]
vy1_vals = [w[9]]
vy2_vals = [w[10]]
vy3_vals = [w[11]]
    
### computing the k1,k2,k3 and k4 values that we'll use to update the positions and velocities of the system.
    
for i in range(N+1):
    for j in range(m):
        K1[j] = h*f(t,w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],w[9],w[10],w[11])[j]
    for j in range(m):
        K2[j] = h*f(t+h/2, w[0]+K1[0]/2, w[1]+K1[1]/2, w[2]+K1[2]/2, w[3]+K1[3]/2,
                    w[4]+K1[4]/2, w[5]+K1[5]/2, w[6]+K1[6]/2, w[7]+K1[7]/2, w[8]+K1[8]/2, 
                    w[9]+K1[9]/2, w[10]+K1[10]/2, w[11]+K1[11]/2)[j]
    for j in range(m):
        K3[j] = h*f(t+h/2, w[0]+K2[0]/2, w[1]+K2[1]/2, w[2]+K2[2]/2, w[3]+K2[3]/2,
                    w[4]+K2[4]/2, w[5]+K2[5]/2, w[6]+K2[6]/2, w[7]+K2[7]/2, w[8]+K2[8]/2, 
                    w[9]+K2[9]/2, w[10]+K2[10]/2, w[11]+K2[11]/2)[j]
    for j in range(m):
        K4[j] = h*f(t+h, w[0]+K3[0], w[1]+K3[1], w[2]+K3[2], w[3]+K3[3],
                    w[4]+K3[4], w[5]+K3[5], w[6]+K3[6], w[7]+K3[7], w[8]+K3[8], 
                    w[9]+K3[9], w[10]+K3[10], w[11]+K3[11])[j]
    for j in range(m):
        w[j] = w[j]+(K1[j]+2*K2[j]+2*K3[j]+K4[j])/6  ### Updating the positions and velocities for the 3 bodies
    t=a+(i+1)*h
    t_vals.append(t)
    x1_vals.append(w[0])
    x2_vals.append(w[1])
    x3_vals.append(w[2])
    y1_vals.append(w[3])
    y2_vals.append(w[4])
    y3_vals.append(w[5])
    vx1_vals.append(w[6])
    vx2_vals.append(w[7])
    vx3_vals.append(w[8])
    vy1_vals.append(w[9])
    vy2_vals.append(w[10])
    vy3_vals.append(w[11])

### çalculating the orbital eccentricities for both Earth and Mars ###
r_earth = np.sqrt(np.array(x2_vals)**2 + np.array(y2_vals)**2)
rp_earth = np.min(np.sqrt(np.array(x2_vals)**2 + np.array(y2_vals)**2))
ra_earth = np.max(np.sqrt(np.array(x2_vals)**2 + np.array(y2_vals)**2))

r_mars = np.sqrt(np.array(x3_vals)**2 + np.array(y3_vals)**2)
rp_mars = np.min(np.sqrt(np.array(x3_vals)**2 + np.array(y3_vals)**2))
ra_mars = np.max(np.sqrt(np.array(x3_vals)**2 + np.array(y3_vals)**2))

e_earth = (ra_earth - rp_earth) / (ra_earth + rp_earth)
e_mars = (ra_mars - rp_mars) / (ra_mars + rp_mars)

print('The orbital eccentricity for Earth is: ', e_earth)
print('The orbital eccentricity for Mars is: ', e_mars)

### çalculating the orbital Period for both Earth and Mars ###
a_earth = rp_earth / (1 - e_earth)
a_mars = rp_mars / (1 - e_mars)

T_earth = 2 * np.pi * np.sqrt((a_earth**3)/(G * Msun))
T_mars = 2 * np.pi * np.sqrt((a_mars**3)/(G * Msun))

print('The orbital Period for Earth is (in years): ', 2 * T_earth / (tmax))
print('The orbital Period for Mars is (in years): ', 2 * T_mars / (tmax))


#Plot the trajectories of the two bodies
plt.plot(x1_vals, y1_vals, label='The Sun')
plt.plot(x2_vals, y2_vals, label='Earth')
plt.plot(x3_vals, y3_vals, label='Mars')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Orbital Trajectories of Earth and Mars around the Sun')
plt.legend()
plt.show()

#Plot the trajectories of the two bodies
plt.figure()
# plt.plot(x1_vals, y1_vals, label='The Sun')
plt.plot(t_vals, r_earth, label='Earth')
plt.plot(t_vals, r_mars, label='Mars')
plt.xlabel('r')
plt.ylabel('t')
plt.title('Plot showing the Distance of the Earth and Mars for the Sun as a function of time')
plt.legend()
plt.show()