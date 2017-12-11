import numpy as np
import matplotlib.pyplot as plt

K = 1000
NW  = 0.5
ITERS = 50000
ETA = 0.1
# A ; self organizing Feature Map implementation:

def gen_rand_inputA():
    return np.random.uniform(-np.pi,np.pi)

def gen_rand_inputB():
    a = np.random.uniform(-np.pi/4,np.pi/4)
    b = np.random.uniform(-np.pi,-np.pi/4)
    c = np.random.uniform(np.pi/4,np.pi)
    rand_n = np.random.uniform(0,1)
    if rand_n> 0.5:
        return a
    rand_n = np.random.uniform(0,1)
    if rand_n> 0.5:
        return b
    return c


def gen_rand_prototypes(k):
    return (np.random.uniform(-np.pi,np.pi,k))

def pi_func(C,v,nw):
    """

    :param C:
    :param v:
    :param nw: Noise Width
    :return:
    """
    return C *np.exp(-1/(nw**2)*np.cos((np.pi*v)/K))

def calc_C(k,nw):
    return 1/sum ([np.exp (-1/(nw**2)*np.cos(np.pi*v/k)) for v in range (-k,k)])

def l(xn,prototypes):
    return np.argmin(abs(xn-prototypes))

def update_prototypes (xn,old_prototypes,eta,k,c,nw):
    my_m = np.asarray(range(k))
    delta = (eta * (xn - old_prototypes)) * pi_func(c,my_m-l(xn,old_prototypes),nw)
    new_prototypes = old_prototypes + delta
    # new_prototypes = [old_prototypes[m]  +  eta*(xn - old_prototypes[m]) * pi_func(c,m-l(xn,old_prototypes),nw)for m in range(k)]
    # new_prototypes = np.asarray(new_prototypes)
    return new_prototypes

def show_graphs (prototypes,time,nw,eta,k):
    title = "time = " + str(time)+", noise width = "+ str(nw)+ ", eta = " + str (eta) +", k = "+ str(k)
    plt.figure(1).suptitle(title)
    plt.title (title)
    plt.subplot(1, 3, 1)
    plt.plot (range(K), prototypes, ".")
    plt.xlabel("neuron index")
    plt.ylabel("orientation")
    plt.subplot(1, 3, 2)
    plt.plot (np.cos(prototypes), np.sin (prototypes), "r")
    plt.xlabel("cos(theta)")
    plt.ylabel("sin (theta)")
    plt.subplot(1, 3, 3)
    hist= np.histogram(prototypes,bins = 360, range = (-np.pi, np.pi))[0]
    plt.plot (hist)

    plt.show()


def SOM (gen_rand_input,iters,eta,nw,k,times):
    prototypes = gen_rand_prototypes(k)
    C = calc_C(k,nw)
    for i in range (iters):
        input_neuron = gen_rand_input()
        prototypes = update_prototypes(input_neuron, prototypes,eta,k,C,nw)
        if (i+1 in times):
            show_graphs(prototypes, i+1,nw,eta,k)
    return prototypes

# SOM (gen_rand_inputA,ITERS,ETA,NW,K,[1,5000,10000,50000])
#
# SOM (gen_rand_inputB,ITERS,ETA,NW,K,[50000])

for n in [0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9]:
    SOM (gen_rand_inputA,50000,ETA,n,K,[50000])





