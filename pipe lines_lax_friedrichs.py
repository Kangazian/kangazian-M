'M.KANGAZIAN. SEPT 2023'
'solving pipe by LaX method'
import math
import numpy as np
#estimation H,V,P for a pipe
def lax_friedrichs(V0, H0, g, C, dx, dt, num_steps):

    num_points = len(V0)
    V = np.zeros((num_steps, num_points))
    H = np.zeros((num_steps, num_points))
    P = np.zeros((num_steps, num_points))

    V[0] = V0
    H[0] = H0

    for i in range(0, 10):
        for j in range(0, num_points - 1):
            V[i][j] = 0.5 * (V[i-1][j+1] + V[i-1][j-1]) - (g * dt) / (2 * dx) * (H[i-1][j+1] - H[i-1][j-1])
            H[i][j] = 0.5 * (H[i-1][j+1] + H[i-1][j-1]) - (g * C**2 * dt) / (2 * dx) * (V[i-1][j+1] - V[i-1][j-1])



        rho = 1000
        P[i] = rho * g * H[i]

    return V, H, P
#******cal_C*******
e=0.004
D=0.5
E=2*10**10
k=2*10**8
C=math.sqrt(1/(1000*((1/k)+(D/E*e))))
# ******in_conditions*******
V0 = np.ones(10) * 9.9
H0 = 5 * np.ones(10)
g = 9.81
dx =600
dt =0.1
num_steps = 10
#************Run function************
V, H, P = lax_friedrichs(V0, H0, g, C, dx, dt, num_steps)

np.set_printoptions(precision=3,suppress=True)
print("\n ***Head on the node for Time[0,1] ,10, dx=600,dt=0.1 Steps 10:****")
print(H)
print("\n**** velocity of the each dx for Time[0,1] ,10 Steps,dx=600,dt=0.1")
print(V)
print("\n**** pressure of the each dx for Time[0,1] ,10 Steps,dx=600,dt=0.1")
print(P)






