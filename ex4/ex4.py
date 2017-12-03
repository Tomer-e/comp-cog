import numpy as np
import matplotlib.pyplot as plt
from prprob import prprob
## Back-Propagation net

NUM_OF_INPUTS = 35
NET_SIZE = 10
ETA = 0.02

def trans_func (x):
    return np.tanh(x)
def trans_dev(x):
    return 1-trans_func(x)**2
def rand_weights (num_of_inputs, num_of_neurons):
    return np.random.normal(0,0.1,(num_of_inputs,num_of_neurons))
def rand_letter(alphabet, mode = 0):
    rand_int = np.random.randint(0,26)
    print(alphabet.shape)
    if (mode == 1):
        return alphabet.T[0], 1
    return alphabet.T[rand_int], rand_int+1

def calc_outputs (weights,inputs):
    return np.dot(weights.T,inputs.reshape ((len(inputs),1)))

def vec_calc_func (func, x):
    return np.asarray([func(a) for a in x])
def calc_a (b,j):
    return np.dot(b.T,j)[0][0]


alphabet, targets = prprob(mode = 1)

p, letter_num = rand_letter(alphabet,1)
v = rand_weights(NUM_OF_INPUTS,NET_SIZE)
x = calc_outputs(v,p)
b = vec_calc_func(trans_func, x)
j = rand_weights(NET_SIZE,1)
z = calc_outputs(b,j)

a = vec_calc_func(trans_func,z)
delta_a = ((1 - a**2) * (letter_num/26 - a))[0][0]


print ("y0 = ", letter_num/26)

print ("X = \n", x)
print ("a = ", a)
print (b.shape)
print (j.shape)
print ("delta a", delta_a)

# print(rand_v().shape)

# p_s =

# print (trans_func(0))



# print (alphabet.shape)
# for i in alphabet:
    # print(i.shape)