import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread, imsave as imsave
from skimage.color import rgb2gray

def read_image_and_flat(filename):
    """
    :param fileame: string containing the image filename to read.
    :return grayscale image
    """

    im = imread(filename)
    # if (255 in im):
    #
    if (im.ndim > 2):
        im = (rgb2gray(im) * 2) - 1
    if(255 in im):
        im = im /255
        im = (rgb2gray(im) * 2) - 1

    return (im.reshape(3600,))

def calc_con_matrix(memories):
    if (memories.ndim == 1):
        memories = memories.reshape((3600,1))
    con_mat = (np.dot(memories,memories.T))
    # print (con_mat)
    np.fill_diagonal(con_mat, 0)
    return con_mat*(1/con_mat[0].size)

def asynchronously_update(con_mat, s):
    s_vec = s.copy()
    s_vec_old = np.zeros(s_vec.shape)
    while(not np.array_equal(s_vec_old, s_vec)):
        s_vec_old = s_vec.copy()
        for i in range(len(s_vec)):
            s_vec[i]= np.sign(np.dot(con_mat[i],s_vec))
            if (s_vec[i] == 0):
                s_vec[i] = -1
        # print ("state: ", s_vec)

    # print ("state: ", s_vec)
    # print ("state: ", s_vec_old,"old")
    # print (sum(s_vec-s_vec_old))
    # s_n = [np.sign(np.dot(con_mat[i],s_vec)) for i in range((s.size))]
    return s_vec

def is_stable(s,memory):
    return np.array_equal(s,memory)

def rand_flip(memory,rate):
    rand = np.zeros(memory.size)+1
    rand[0:int(memory.size*rate)] = [-1]* int(memory.size*rate)
    np.random.shuffle(rand)
    return (np.asarray([rand[i]*memory[i] for i in range(len(memory))]))

def stimulate(rate, n_iters, animal):
    suc = 0
    fail = 0
    for i in range (n_iters):
        animal_with_err = rand_flip(animal, rate)
        s_with_err = asynchronously_update(con_mat, animal_with_err)
        if (is_stable(s_with_err, animal)):
            suc+=1
        else:
            fail+=1
    print ("err rate = ", rate,", converge: ", suc, ", failed to converge:",fail)
    return rate, suc
#A
lion = read_image_and_flat("lion.png")
monkey = read_image_and_flat("monkey.png")
mystery = read_image_and_flat("mystery.png")
tiger = read_image_and_flat("tiger.png")
zebra = read_image_and_flat("zebra.png")

#B
con_mat = calc_con_matrix(lion)

#C
s = asynchronously_update(con_mat,lion)

#D
print ("was the net converge to lion? ",is_stable(s,lion))

#E
lion_with_err =rand_flip(lion,0.2)
s_with_err = asynchronously_update(con_mat,lion_with_err)
print ("was the net converge to lion? (lion with 20% noise)",is_stable(s_with_err,lion))

#F,G
print ("F,G")
x = [0]*5
y = [0]*5
x[0],y[0] = stimulate(0.2, 100, lion)
x[1],y[1] = stimulate(0.3, 100, lion)
x[2],y[2] = stimulate(0.4, 100, lion)
x[3],y[3] = stimulate(0.5, 100, lion)
x[4],y[4] = stimulate(0.6, 100, lion)

plt.plot(x,y)
plt.show()

#H
con_mat = calc_con_matrix(np.asarray([lion,monkey,zebra]).T)

print ("H")
x = [0]*5
y = [0]*5
x[0],y[0] = stimulate(0.2, 100, lion)
x[1],y[1] = stimulate(0.3, 100, lion)
x[2],y[2] = stimulate(0.4, 100, lion)
x[3],y[3] = stimulate(0.5, 100, lion)
x[4],y[4] = stimulate(0.6, 100, lion)

plt.plot(x,y)

plt.show()
#I
print ("I")
con_mat = calc_con_matrix(np.asarray([lion,monkey,tiger,zebra]).T)
my = asynchronously_update(con_mat,mystery)

my = my.reshape(60,60)
plt.imshow(my,cmap = "gray")
plt.show()
# simulate(0.2,100,mystery)
