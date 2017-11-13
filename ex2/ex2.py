import numpy as np
import matplotlib.pyplot as plt

def generate_points_n_tags (amount):
    points = np.random.normal(0,5,(amount,2))
    tags = points[:,0] -points[:,1]
    tags = np.sign(tags)
    tags[tags==0] = -1
    return points.T,tags

def binary_per(x,y0):
    w = np.array([1,1])
    return
points, tags = generate_points_n_tags(1000)
# print(points)
# print(tags)
plt.plot(points[0,:][tags==-1], points[1,:][tags==-1], "ro")
plt.plot(points[0,:][tags==1], points[1,:][tags==1], "bo")

print (points)
binary_per(points,tags)
plt.show()
