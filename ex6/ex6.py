
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NUM_OF_EXAMPLES  = 10000
ETA = 10**(-5)
DELTA_LIM = 10**(-5)
########### --EX6-- ###########

def gen_examples ():
    var = np.asarray([1,1,1])
    u1_avg = np.asarray([20,0,20])
    u2_avg = np.asarray([-10,10,-10])
    u3_avg = np.asarray([-10,-10,-10])
    u = [u1_avg,u2_avg,u3_avg]
    names = ["u1","u2","u3"]
    examples1 = np.random.normal(u[0], var, size = (NUM_OF_EXAMPLES,3))
    examples2 = np.random.normal(u[1], var, size = (NUM_OF_EXAMPLES,3))
    examples3 = np.random.normal(u[2], var, size = (NUM_OF_EXAMPLES,3))

    return np.asarray([examples1,examples2,examples3])

def initial_w():
    return np.asarray([1,2,3])

def gen_rand_examples(examples):
    rand_dist = np.random.randint(0,3)
    rand_ex = np.random.randint(0,10000)
    return examples[rand_dist][rand_ex]

def net_training(w0,eta):
    examples = gen_examples()
    deltaW = np.inf
    w = w0
    w_s = [w.copy()]
    norms = [np.linalg.norm(w)]
    while (np.linalg.norm(deltaW) >= DELTA_LIM):
        x = gen_rand_examples(examples)
        y = np.dot(w,x)
        # print(x.shape, w.shape)
        deltaW = (eta*y)*(x-(y*w))
        # print(deltaW)
        w = w + deltaW
        w_s.append(w.copy())
        norms.append(np.linalg.norm(w))

    return examples,w,norms,np.asarray(w_s)

def calc_correlation_matrix(examples):
    return np.dot(examples.T,examples)/(3*NUM_OF_EXAMPLES)

def get_max_eigen_vector(matrix):
    values, vectors = (np.linalg.eig(matrix))


    return vectors[:,np.argmax(values)]
####### A #######
w = initial_w()

examples,w,norms,w_s = net_training(w,ETA)

####### B #######

plt.figure(1)
plt.plot (range(len(norms)), norms)
plt.xlabel("Num of iterations")
plt.ylabel("Weights vector norm")
plt.title ("B")

####### C #######

fig = plt.figure(2)
ax = fig.add_subplot(111,projection = '3d')
examples = examples.reshape(30000,3)
plt.title("C - Examples & W")

####### D #######

cor = calc_correlation_matrix(examples)
print ("D. Correlation Matrix = ")
print (cor)
max_eigen_vec = get_max_eigen_vector(cor)
to_show_w = w*20## just for visualization
ax.scatter(examples[:,0],examples[:,1],examples[:,2],".")
ax.plot([0,to_show_w[0]],[1,to_show_w[1]],[2,to_show_w[2]], "r-")

####### E #######

plt.figure(3)
vec_w_diff = w_s - max_eigen_vec
vec_w_diff_norms = np.linalg.norm(vec_w_diff,axis=1)
plt.plot (range(len(vec_w_diff_norms)), vec_w_diff_norms)
plt.xlabel("Num of iterations")
plt.ylabel("eigenvector -weights norm")

plt.title ("E")

####### F #######

print("F. The vector with the max eigenvalue = ",max_eigen_vec)
print("W =",w)

plt.show()

