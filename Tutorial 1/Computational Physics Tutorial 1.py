import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter


# n = [5, 10, 20, 100, 1000]               ### The number of rows of the Matrix and the dimension of the vector
n = np.arange(20, 201, 20)
n1 = n


###########################################################################
###                     Generating the matrix M                         ###
###########################################################################
if __name__ == '__main__':              ### This creates an empty List of Lists with a number of elements 
                                        ### equal to the length of n
    n2 = len(n)
    M = [[] for x in range(n2)]
    x = [[] for x in range(n2)]
# M = [[],[],[],[],[]]
# x = [[],[],[],[],[]]

for t in range(len(n)):
    for i in range(n[t]):
        M[t].append([])
        x[t].append(i+1)

        for j in range(n1[t]):                  
            M[t][i].append(((37.1*(i+1) + 91.7*j**2) % 20.0) - 10.0)

###########################################################################
###          Defining a function that multiplies Two Matrices           ###
###########################################################################

def matrix_multiplication(A, B):
    # Get the number of rows and columns of A and B
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    # Check if the matrices can be multiplied
    if cols_A != rows_B:
        print("Matrices cannot be multiplied")
        return None

    # Create an empty result matrix with the correct dimensions
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]

    # Perform matrix multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]

    return C

###########################################################################
###           Defining a function that multiplies Two Vectors           ###
###########################################################################

def dot_product(row_vector, col_vector):
    """
    Multiplies a row vector and a column vector.
    
    Args:
    row_vector (list): A list representing the row vector.
    col_vector (list): A list representing the column vector.
    
    Returns:
    result (float): The product of the two vectors.
    """
    if len(row_vector) != len(col_vector):
        raise ValueError("Vectors must have the same length.")
    
    result = 0
    for i in range(len(row_vector)):
        result += row_vector[i] * col_vector[i]
    
    return result

###########################################################################
###      Defining a fuction that multiplies a Matrix and a Vector       ###
###########################################################################

def matrix_vector_mult(M,x):
    y = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            y[i] += M[i,j] * x[j]

    return y

###########################################################################
# xT dot x and xT dot Mx for the defined functions for n = 5,10,20,100,1000
###########################################################################

xTx1 = []                
Mx1 = []
xMx1 = []

for t in range(len(n)):
    xTx1.append(dot_product(np.transpose(np.array(x[t])), np.array(x[t])))
    Mx1.append(matrix_vector_mult(np.array(M[t]), np.array(x[t])))
    xMx1.append(dot_product(np.transpose(np.array(x[t])), Mx1[t]))

###########################################################################
# Tabulated results for xT dot x and xT dot Mx for the defined functions
###########################################################################

data = {
    'n': n,
    'xT dot x': xTx1,
    'xt dot Mx': xMx1
}

df = pd.DataFrame(data)
print(df)

n_ = np.arange(20, 201, 20)
n1_ = n_


###########################################################################
###                     Generating the matrix M                         ###
###########################################################################
if __name__ == '__main__':              ### This creates an empty List of Lists with a number of elements 
                                        ### equal to the length of n
    n2 = len(n_)
    M = [[] for x in range(n2)]
    x = [[] for x in range(n2)]
# M = [[],[],[],[],[]]
# x = [[],[],[],[],[]]

for t in range(len(n_)):
    for i in range(n_[t]):
        M[t].append([])
        x[t].append(i+1)

        for j in range(n1_[t]):                  
            M[t][i].append(((37.1*(i+1) + 91.7*j**2) % 20.0) - 10.0)

###########################################################################
# Tabulated results for xT dot x and xT dot Mx for the defined functions 
# and the execution times
###########################################################################

t1_start = []
t1_stop = []
t1_start_pkg = []
t1_stop_pkg = []
for t in range(len(n_)):
    t1_start.append(perf_counter())
    dot_product(np.transpose(np.array(x[t])), np.array(x[t]))
    t1_stop.append(perf_counter())
    ### Execution time for the Python Package ###
    t1_start_pkg.append(perf_counter())
    np.matmul(np.transpose(np.array(x[t])), np.array(x[t]))
    t1_stop_pkg.append(perf_counter())
 
ex_time1 = np.array(t1_stop)-np.array(t1_start)
ex_time1_pkg = np.array(t1_stop_pkg)-np.array(t1_start_pkg)
print("Elapsed time during the dot product of two Vectors in seconds:", ex_time1)
print("Elapsed time during the dot product of two Vectors using the Python Package in seconds:", ex_time1_pkg)


t2_start = []
t2_stop = []
t2_start_pkg = []
t2_stop_pkg = []
for t in range(len(n_)):
    t2_start.append(perf_counter())                 ### Starts the Stopwatch
    matrix_vector_mult(np.array(M[t]), np.array(x[t]))
    t2_stop.append(perf_counter())                  ### Stops the Stopwatch

    ### Execution time for the Python Package ###
    t2_start_pkg.append(perf_counter())
    np.matmul(np.array(M[t]), np.array(x[t]))
    t2_stop_pkg.append(perf_counter())
 
ex_time2 = np.array(t2_stop)-np.array(t2_start)
ex_time2_pkg = np.array(t2_stop_pkg)-np.array(t2_start_pkg)
# print("Elapsed time during the mutliplication of a Matrix and a Vector in seconds:", ex_time2)
# print("Elapsed time during the mutliplication of a Matrix and a Vector using the Python Package in seconds:", ex_time2_pkg)



t3_start = []
t3_stop = []
t3_start_pkg = []
t3_stop_pkg = []
for t in range(len(n_)):
    t3_start.append(perf_counter())                 ### Starts the Stopwatch     
    dot_product(np.transpose(np.array(x[t])), matrix_vector_mult(np.array(M[t]), np.array(x[t])))
    t3_stop.append(perf_counter())                  ### Stops the Stopwatch

    ### Execution time for the Python Package ###
    t3_start_pkg.append(perf_counter())
    np.matmul(np.transpose(np.array(x[t])), np.matmul(np.array(M[t]), np.array(x[t])))
    t3_stop_pkg.append(perf_counter())
 
 
ex_time3 = np.array(t3_stop)-np.array(t3_start)
ex_time3_pkg = np.array(t3_stop_pkg)-np.array(t3_start_pkg)
# print("Elapsed time during the multiplication of two Vectors and a Matrix in seconds:", ex_time3)
# print("Elapsed time during the multiplication of two Vectors and a Matrix using the Python Package in seconds:", ex_time3_pkg)
print(np.array(matrix_multiplication(np.array(M[0]), np.array(M[0]))))


t4_start = []
t4_stop = []
t4_start_pkg = []
t4_stop_pkg = []
for t in range(len(n_)):
    t4_start.append(perf_counter())                     ### Starts the Stopwatch
    matrix_multiplication(np.array(M[t]), np.array(M[t]))
    t4_stop.append(perf_counter())                      ### Stops the Stopwatch

    ## Execution time for the Python Package ###
    t4_start_pkg.append(perf_counter())
    np.matmul(np.array(M[t]), np.array(M[t]))
    t4_stop_pkg.append(perf_counter())
 
ex_time4 = np.array(t4_stop)-np.array(t4_start)
ex_time4_pkg = np.array(t4_stop_pkg)-np.array(t4_start_pkg)
# print("Elapsed time during the multiplication of two Matrices: in seconds:", ex_time4)
# print("Elapsed time during the multiplication of two Matrices using the Python Package in seconds:", ex_time4_pkg)

###########################################################################
###      Tabulating the executions times for both Implementations       ###
###########################################################################

data1 = {
    'n': n_,
    'Execution Times for Manual Operation of xT . x (s)': ex_time1,
    'Execution Times for Python Packages of xT . x (s)': ex_time1_pkg,
    'Execution Times for Manual Operation of M . x (s)': ex_time2,
    'Execution Times for Python Packages of M . x (s)': ex_time2_pkg,
    'Execution Times for Manual Operation of xT . Mx (s)': ex_time3,
    'Execution Times for Python Packages of xT . Mx (s)': ex_time3_pkg,
    'Execution Times for Manual Operation of M . M (s)': ex_time4,
    'Execution Times for Python Packages of M . M (s)': ex_time4_pkg,
}

df1 = pd.DataFrame(data1)
print(df1)


###########################################################################
###           Plotting the execution times as a function of n           ###
###########################################################################

plt.scatter(n_, ex_time1, label = "Custom Implementation", marker = 'o')
plt.scatter(n_, ex_time1_pkg, label = "Package implementation", marker = 'o')
plt.legend(loc = 'best')
plt.xlabel('Dimensions (n)')
plt.ylabel('Execution Times (s)')
plt.title(" Execution Times as a Function of n for Vector Multiplication")
plt.show()

plt.scatter(n_, ex_time2, label = "Custom Implementation", marker = 'o')
plt.scatter(n_, ex_time2_pkg, label = "Package implementation", marker = 'o')
plt.legend(loc = 'best')
plt.xlabel('Dimensions (n)')
plt.ylabel('Execution Times (s)')
plt.title(" Execution Times as a Function of n for Matrix-Vector Multiplication")
plt.show()

plt.scatter(n_, ex_time3, label = "Custom Implementation", marker = 'o')
plt.scatter(n_, ex_time3_pkg, label = "Package implementation", marker = 'o')
plt.legend(loc = 'best')
plt.xlabel('Dimensions (n)')
plt.ylabel('Execution Times (s)')
plt.title(" Execution Times as a Function of n for Vector-Matrix-Vector Multiplication")
plt.show()

plt.scatter(n_, ex_time4, label = "Custom Implementation", marker = 'o')
plt.scatter(n_, ex_time4_pkg, label = "Package implementation", marker = 'o')
plt.legend(loc = 'best')
plt.xlabel('Dimensions (n)')
plt.ylabel('Execution Times (s)')
plt.title(" Execution Times as a Function of n for Matrix Multiplication")
plt.show()

# plt.legend(loc = 'best')
# plt.xlabel('Dimensions (n)')
# plt.ylabel('Execution Times (s)')
# plt.title(" Execution Times as a Function of n for Manual Implementations")
# plt.show()


plt.plot(n_, ex_time1_pkg, label = "xTx")
plt.plot(n_, ex_time2_pkg, label = "Mx")
plt.plot(n_, ex_time3_pkg, label = "xTMx")
plt.plot(n_, ex_time4_pkg, label = "MM")
plt.legend(loc = 'upper right')
plt.xlabel('Dimensions (n)')
plt.ylabel('Execution Times (s)')
plt.title(" Execution Times as a Function of n for Built-in Package Implementations")
plt.show()

###########################################################################
###                 Estimating FLOPS for the operations                 ###
###########################################################################

def calculate_flops(results):
    xtx_flops = 2 * sum([result[0]**2 for result in results])
    xtmx_flops = 2 * sum([2*result[0]**2 + result[0]**3 for result in results])
    mm_flops = sum([2*result[0]**3 for result in results])
    mx_flops = 2 * sum([result[0]**2 for result in results])
    xtx_time = sum([result[5] for result in results])
    xtmx_time = sum([result[6] for result in results])
    mm_time = sum([result[7] for result in results])
    mx_time = sum([result[8] for result in results])
    total_time = xtx_time + xtmx_time + mm_time + mx_time
    total_flops = xtx_flops + xtmx_flops + mm_flops + mx_flops
    flops_per_sec = total_flops / total_time
    return flops_per_sec

results = [
    (100, "numpy.dot", 10, 0.001, 0.0001, 0.0005, 0.001, 0.001, 0.002),
    (200, "numpy.matmul", 20, 0.005, 0.0002, 0.002, 0.005, 0.005, 0.01),
    (300, "custom", 5, 0.01, 0.001, 0.01, 0.02, 0.03, 0.02)
]

flops_per_sec = calculate_flops(results)
print("FLOPS per second: {:.2e}".format(flops_per_sec))

elapsed_t = np.array([ex_time1, ex_time2, ex_time3, ex_time4])
elapsed_t_pkg = np.array([ex_time1_pkg, ex_time2_pkg, ex_time3_pkg, ex_time4_pkg])
flops_xtx = []
flops_mx = []
flops_xtmx = []
flops_mm = []
for t in range(len(n_)):
    flops_xtx.append((2*n_[t]**3)/ex_time1)
    flops_mx.append((2*n_[t]**3)/ex_time2)
    flops_xtmx.append((2*n_[t]**3)/ex_time3)
    flops_mm.append((2*n_[t]**3)/ex_time4)


data3 = {
    'xTx': flops_xtx,
    'Mx': flops_mx,
    'xTMx': flops_xtmx,
    'MM': flops_mm
}
df3 = pd.DataFrame(data3)
print(df3)

## Calculating FLOPS for one of the custom defined functions ##
FLOPS = 2*(n_[0]**3)/ex_time1[0]
print(FLOPS)