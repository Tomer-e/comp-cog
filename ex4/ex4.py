import numpy as np
import matplotlib.pyplot as plt
from prprob import prprob
## Back-Propagation net

NUM_OF_LETTERS = 26
NUM_OF_INPUTS = 35
NET_SIZE = 10
ETA_a = 0.02
ETA_b = 0.2

def trans_func (x):
    return np.tanh(x)
def trans_dev(x):
    return 1-trans_func(x)**2
def rand_weights (num_of_inputs, num_of_neurons):
    return np.random.normal(0,0.1,(num_of_inputs,num_of_neurons))
def rand_letter(alphabet, mode = 0):
    rand_int = np.random.randint(0,26)
    if (mode == 1):
        return alphabet.T[0], 1
    return alphabet.T[rand_int], rand_int+1

def calc_outputs (weights,inputs):
    return np.dot(weights.T,inputs.reshape ((len(inputs),1)))

def vec_calc_func (func, x):
    return np.asarray([func(a) for a in x])
def calc_a (b,j):
    return np.dot(b.T,j)[0][0]

def net(ETA,num_of_neurons,iters):
    alphabet, targets = prprob(mode = 1)

    v = rand_weights(NUM_OF_INPUTS,num_of_neurons)
    j = rand_weights(num_of_neurons,1)

    error = []
    for i in range (iters):
        # -------- StepA --------
        p, letter_num = rand_letter(alphabet)
        x = calc_outputs(v,p)
        b = vec_calc_func(trans_func, x)
        z = calc_outputs(b,j)
        a = vec_calc_func(trans_func,z)

        # -------- StepB --------

        delta_a = ((1 - a**2) * (letter_num/NUM_OF_LETTERS - a))[0][0]
        # print ("b shape = ", b.shape, "delta a shape =" ,delta_a.shape)
        # print ("j shape =", j.shape)
        delta_b_s = (1 - b**2) * (delta_a * j)

        # -------- StepC --------
        j = j+(ETA * b * delta_a)
        # print ("p shape = ", p.shape, "delta bs shape =" ,delta_b_s.shape)
        v = v+(ETA * p * delta_b_s).T

        # -------- Generalization Error --------
        iter_err = 0
        for i in range (NUM_OF_LETTERS):
            wanted = i/NUM_OF_LETTERS
            x = calc_outputs(v, alphabet.T[i])
            b = vec_calc_func(trans_func, x)
            z = calc_outputs(b, j)
            a = vec_calc_func(trans_func, z)
            iter_err+= (0.5*(1/NUM_OF_LETTERS)*(wanted-a[0][0])**2)
        error.append(iter_err)

    # -------- print results --------
    print ("ETA = ",ETA, ", Number of neurons = ",num_of_neurons, ", Number of iterations", iters)
    for i in range (NUM_OF_LETTERS):
        x = calc_outputs(v, alphabet.T[i])
        b = vec_calc_func(trans_func, x)
        z = calc_outputs(b, j)
        a = vec_calc_func(trans_func, z)
        print ("Network output = ", (a*NUM_OF_LETTERS)[0][0].round(), "actual letter = ", i+1)
    return error

def show_err(error, eta, n_neurons, figure):
    plt.figure(figure)
    plt.plot(range(2000),error[0:2000])
    title = "ETA = "+ str(eta)+ ", number of neurons = "+ str(n_neurons)
    plt.title(title)
    plt.xlabel("Iter")
    plt.ylabel("Error")


error1 = net(ETA_a,10,150000)
error2 = net (ETA_b,10,150000)
error3 = net(ETA_a,2,150000)

show_err(error1, ETA_a,10,1)
show_err(error2,ETA_b,10,2)
show_err(error3, ETA_a, 2,3)

plt.show()
# print (a)
# print("delta_b_s: \n",delta_b_s)
#
# print ("y0 = ", letter_num)
# print ("a  *26 = " ,(a*26)[0][0])
# print ("X = \n", x)
# print (b.shape)
# print (j.shape)
# print ("delta a", delta_a)


# print(rand_v().shape)

# p_s =

# print (trans_func(0))



# print (alphabet.shape)
# for i in alphabet:
    # print(i.shape)