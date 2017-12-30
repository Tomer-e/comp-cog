import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.axes.Axes import semilogx
from mpl_toolkits.mplot3d import Axes3D


M = 2
SD = 0.1
ITERS = 10000
ETA = np.logspace(-5,-2,6,dtype=np.float64)
U_0 = 0
REPEATS = 200
my_range = range(ITERS)

def R(y) : return  (-(M-y)**2)

def Rb(y):
    if (y<=M):
        return -2*(M-y)**2
    return -(M-y)**2
ETAb = 0.001
SDb = [0.1,0.5]

def reinforce(eta,reward,u_0,sd, iters):

    u = u_0
    u_s =[u_0]
    for i in range (1,ITERS):
        y = u + np.random.normal(0,sd)
        r = reward (y)
        delta_u = eta * r * ((y - u)/(sd**2))
        u = u+delta_u
        u_s.append(u)

    return u , np.asarray(u_s)

def graph_per_eta(eta,r ,repeats, fig, sd, plotU):
    all_u_s = []
    plt.figure(fig)
    title = "eta = " + str(eta)
    plt.title (title)
    converge = 0
    for i in range (REPEATS):
        u,u_s= reinforce(eta,r,U_0,sd,ITERS)
        if (plotU) :
            if i in [8,71, 100,144, 188]:
                plt.plot(my_range,u_s)
        all_u_s.append(u_s)
        if (u>=(M-0.1) and u<=(M+0.1)):
            converge+=1
    all_u_s = np.asarray(all_u_s)
    median_u = np.median(all_u_s,axis=0)
    plt.plot(my_range,median_u,color = "black",linewidth = 3)
    plt.xlabel("time (iteration number)")
    plt.ylabel("mu")
    return converge

### A ###
convergence = []
for i in range (len(ETA)):
    plt.figure(i)
    convergence.append(graph_per_eta(ETA[i],R,REPEATS, i,SD,1))
print(convergence)
convergence = np.asarray(convergence)/REPEATS
plt.figure(6)
# plt.plot(ETA,convergence)
plt.semilogx(ETA,convergence)

### B ###

plt.figure(7)
range_y = np.linspace(-1,5,REPEATS)
rb = np.vectorize(Rb)
rb = rb(range_y)
plt.plot(range_y,rb, color = "black")


u1,u_s1 = reinforce(ETAb,Rb,U_0,SDb[0],REPEATS)
med_u1 = np.median(u_s1)
u2,u_s2 = reinforce(ETAb,Rb,U_0,SDb[1],REPEATS)
med_u2 = np.median(u_s2)

plt.plot (med_u1,Rb(med_u1),"r.")
plt.plot (med_u2,Rb(med_u2),"r.")
plt.show()
