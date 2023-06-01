import numpy as np

mu_0 = 4*np.pi*1e-7
I =  1*(60e-3)*(15e-3)
L = 60e-3
r1 = 10e-3
r2 = 25e-3
x1 = 10e-3
x2 = 70e-3

def B(I, r1, r2, x1, x2):
    B_field = (mu_0*I/(2*(r2-r1)*L))*(x2*np.log((np.sqrt(r2**2+x2**2)+r2)/(np.sqrt(r1**2+x2**2)+r1)) - 
                                    x1*np.log((np.sqrt(r2**2+x1**2)+r2)/(np.sqrt(r1**2+x1**2)+r1)))
    return B_field

BField = B(I,r1,r2,x1,x2)
print(BField)