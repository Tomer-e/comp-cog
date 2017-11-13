import numpy as np
import matplotlib.pyplot as plt

def generate_points_n_tags (amount):
    points = np.random.normal(0,5,(amount,2))
    tags = points[:,0] -points[:,1]
    tags = np.sign(tags)
    tags[tags==0] = -1
    return points.T,tags

def show_points (points,tags):
    plt.plot(points[0, :][tags == -1], points[1, :][tags == -1], "ro", zorder = 0)
    plt.plot(points[0, :][tags == 1], points[1, :][tags == 1], "bo", zorder = 0)

def graph(formula, x_range):
    x = np.array(x_range)
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y,color = "g")


def binary_per(x,y0):
    w = np.array([1,1])
    x1 = x[0]
    x2 = x[1]
    signs = np.zeros(len(y0))
    iter_counter = 0
    while (not np.array_equal(signs, y0)):
        for i in range (len (x1)):
            signs[i] = np.sign (w[0]*x1[i] + w[1]*x2[i])
            if (signs[i] != y0[i] or (iter_counter == 0 and i == 0)):
                w[0] = w[0] + y0[i]*x1[i]
                w[1] = w[1] + y0[i] * x2[i]
        iter_counter += 1
    print ("iter_counter", iter_counter)
    return  (w)
    # print (x1[i],x2[i], y0[i])

points, tags = generate_points_n_tags(1000)
my_points = np.array([2,1,-1,-1,
                      3,3,-1,2]).reshape(2,4)
my_y = np.array([1,1,-1,-1])

show_points(points,tags)
w = binary_per(points,tags)
print(w)
formula = lambda x: (-w[0]*x)/w[1]
graph(formula, range(int(points[0].min()),int(points[1].max())))
plt.plot ()
plt.quiver(w[0],w[1],color='black', units = "y", zorder = 1)

axes = plt.axis()
plt.show()
print(my_points)
# print(points)
# print(tags)

# print (points)
