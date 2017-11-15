import numpy as np
import matplotlib.pyplot as plt


def generate_points_n_tags(amount):
    points = np.random.normal(0, 5, (amount, 2))
    tags = points[:, 0] - points[:, 1]
    tags = np.sign(tags)
    tags[tags == 0] = -1
    # print(sum (tags))
    return points.T, tags


def show_points(points, tags):
    plt.plot(points[0, :][tags == -1], points[1, :][tags == -1], "ro", zorder=0)
    plt.plot(points[0, :][tags == 1], points[1, :][tags == 1], "bo", zorder=0)


def graph(formula, x_range, color):
    x = np.array(x_range)
    y = formula(x)
    plt.plot(x, y, color=color)


def binary_per(x, y0):
    w = np.array([1, 1], np.float64)
    x1 = x[0]
    x2 = x[1]
    signs = np.zeros(len(y0))
    iter_counter = 0
    while (not np.array_equal(signs, y0)):
        # print("signs =", signs)
        # print ("yo = ", y0)
        for i in range(len(x1)):
            signs[i] = np.sign(w[0] * x1[i] + w[1] * x2[i])
            if (signs[i] != y0[i] or (iter_counter == 0 and i == 0)):
                w[0] = w[0] + y0[i] * x1[i]
                w[1] = w[1] + y0[i] * x2[i]
        iter_counter += 1
    return (w / np.linalg.norm(w))


def stimulate(to_repeat, n_points):
    w_s = []
    ideal = np.array([1, -1])
    plt.quiver(ideal[0], ideal[1], color="black", units="xy", zorder=1)

    for i in range(to_repeat):
        points, tags = generate_points_n_tags(n_points)
        # if (n_points <1000):
        #
        #     show_points(points,tags)
        w = binary_per(points, tags)
        w_s.append(w.copy())
        # error = 1/to_repeat
        calc_line = lambda x: (-w[0] * x) / w[1]
        color = np.random.rand(3, 1)
        graph(calc_line, range(int(points[0].min()), int(points[0].max() + 1)), color)
        plt.quiver(w[0], w[1], color=color, units="xy", zorder=1)
    w_s = np.asarray(w_s)
    err = 0
    for i in range(len(w_s)):
        x = np.dot(w_s[i], ideal)
        y = np.linalg.norm(ideal) * np.linalg.norm(w_s[i])
        err += np.fabs(np.arccos(x / y))
    err = err / to_repeat
    return err


# A
points, tags = generate_points_n_tags(1000)
show_points(points, tags)

# B
w = binary_per(points, tags)

# C
calc_line = lambda x: (-w[0] * x) / w[1]
color = np.random.rand(3, 1)
graph(calc_line, range(int(points[0].min()), int(points[1].max())), color)
plt.quiver(w[0], w[1], color=color, units="y", zorder=1, width=0.2)
plt.title("1000 points, w, separate line")

plt.show()

# D
plt.figure(0)
error = stimulate(10, 1000)
plt.title("1000 points, 10 repetitions")
plt.figure(1)
plt.title("25 points, 10 repetitions")
error = stimulate(10, 25)

plt.show()

# E
errors = []
p = [25, 35, 55, 100, 150, 200, 500]
for points_n in p:
    errors.append(stimulate(100, points_n))

plt.clf()  # clear plot
plt.title("errors")
plt.plot(p, errors, "--bo")
plt.xlabel("p")
plt.ylabel("error")
plt.show()
