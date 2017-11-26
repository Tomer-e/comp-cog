import numpy as np
import matplotlib.pyplot as plt

def rand_samples(size):
    samples = np.random.uniform(-1,1,(size,2))
    # print(samples[:,1])
    samples[:, 1] = np.ones(size)
    return samples

def calculate_correlation_matrix(samples):
    c = np.asarray([[0,0],[0,0]],np.float64)
    c[0][0] = sum (samples[:, 0]**2)
    c[1][1] = sum (samples[:, 1]**2)
    c[0][1] =  sum(samples[:, 0]*samples[:, 1])
    c[1][0] =  sum(samples[:, 0]*samples[:, 1])
    return c/len(samples)

def y0(x):
    return x**3-x**2+x-1

def calc_u (samples):
    y_s = np.asarray([y0(x) for x in samples[:,0]])
    u1 = sum(np.multiply(samples[:,0],y_s))
    u2 = sum(y_s)

    return np.asarray([u1,u2]).T/len(samples)

def calc_w(c,u):
    c_inv = np.linalg.inv(c)
    return np.dot (c_inv,u)

samples = rand_samples(500)
# print (a)
c = calculate_correlation_matrix(samples)
u = calc_u(samples)
print ("u :\n", u)
w = calc_w(c,u)
print ("w :\n", w)

y0_s = np.asarray([y0(x) for x in samples[:,0]])
y_s = np.asarray([np.dot(w,x) for x in samples])
plt.plot (samples[:,0], y_s, "r.")
plt.plot (samples[:,0], y0_s, ".")


plt.show()