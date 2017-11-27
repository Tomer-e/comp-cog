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
def calc_y(w,x_s):
    return np.asarray([np.dot(w,x) for x in x_s])

def calc_error (y0_s,y_s):
    return (sum((y0_s-y_s)**2))/(2*len(y0_s))

def simulate (start, end ,step,  iters):
    training_errors = []
    for size in range(start,end,step):
        training_errors.append(0)
    training_errors = np.asarray(training_errors,np.float64)
    # print(training_errors)
    # training_errors = np.zeros(len (training_errors))
    # print(training_errors)

    generalization_errors = np.zeros(len (training_errors))
    for size in range(start,end,step):
        for i in range (iters):
            samples = rand_samples(size)
            c = calculate_correlation_matrix(samples)
            u = calc_u(samples)
            w = calc_w(c, u)
            y0_s = np.asarray([y0(x) for x in samples[:, 0]])
            y_s = calc_y(w, samples)
            training_error = calc_error(y0_s, y_s)
            training_errors[size//step-1]+= training_error
            all_range_samples = np.asarray([[x / 50, 1] for x in range(-50, 50, 1)])
            y0_s = np.asarray([y0(x) for x in all_range_samples[:, 0]])
            y_s = calc_y(w, all_range_samples)
            generalization_error = calc_error(y0_s, y_s)
            generalization_errors[size//step-1]+=generalization_error

    return range(start,end,step),training_errors/iters,generalization_errors/iters



samples = rand_samples(500)
# print (a)
c = calculate_correlation_matrix(samples)
print ("c : \n",c)
u = calc_u(samples)
print ("u :\n", u)
w = calc_w(c,u)
print ("w :\n", w)

y0_s = np.asarray([y0(x) for x in samples[:,0]])
y_s = calc_y(w,samples)

plt.title("y0 vs y")
plt.plot (samples[:,0], y_s, "r.", label = "y")
plt.plot (samples[:,0], y0_s, ".",label = "y0")
plt.legend()
plt.xlabel("x (-1,1)")
plt.ylabel("y = x^3-x^2+x-1")
plt.show()
print ()

# training error:
training_error = calc_error(y0_s,y_s)
print ("training error = " ,training_error)

# generalization error
all_range_samples = np.asarray([[x/50,1] for x in range (-50,50,1)])
y0_s = np.asarray([y0(x) for x in all_range_samples[:,0]])
y_s  = calc_y(w,all_range_samples)
generalization_error = calc_error(y0_s,y_s)
print ("generalization_error = ", generalization_error)

samples_num ,training_errors,generalization_errors =  simulate(5,100,5,100)
plt.title("<training error> vs <generalization error>" )
plt.plot(samples_num,training_errors,"r",label = "training error")
plt.plot(samples_num,generalization_errors,"b", label = "generalization error")
plt.legend()
plt.xlabel("num of samples")
plt.ylabel("<error>")
plt.show()


