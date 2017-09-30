import matplotlib.pyplot as plt
import numpy as np

h = lambda b1, b0, x: x * b1 + b0

# Number of samples
_np = 100

x = np.transpose(np.matrix(np.array(np.random.normal(0, np.sqrt(0.7), _np))))
error = np.random.normal(0, np.sqrt(0.5), _np)
y = h(0.5, 2, x)
X = np.concatenate((x, np.ones(shape=[len(x), 1])), 1)

plt.plot(X[:, 0], y, 'k-')

i = 0

# Training
b1 = 0
b0 = 0
alpha = 0.001

while 1:
    pre_b1 = b1
    pre_b0 = b0
    db1 = float(np.transpose(y - (X@np.transpose(np.matrix([b1, b0]))))@X[:, 0])
    b1 = pre_b1 + db1 * alpha
    db0 = float(np.transpose(y - (X@np.transpose(np.matrix([b1, b0]))))@X[:, 1])
    b0 = pre_b0 + db0 * alpha

    if abs(pre_b1 - b1) < 0.001 and abs(pre_b0 - b0) < 0.001:
        break
    i += 1
    print(i)
    print(b1)
    print(b0)
    _x = np.linspace(start=0, stop=10, num=100).flatten()
    _y = np.array(list(map(lambda elem: elem * b0 + b1, x))).flatten()

    plt.plot(_x, _y, 'k-')
_x = np.linspace(start=0, stop=10, num=100).flatten()
_y = np.array(list(map(lambda elem: elem * b0 + b1, x))).flatten()
plt.plot(_x, _y, 'r-')
plt.show()