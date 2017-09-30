import numpy as np
import matplotlib.pyplot as plt

LR1D = lambda tx, ty:  np.linalg.inv(np.transpose(tx)@tx)@(np.transpose(tx)@ty)

trainX = np.transpose(np.array([[2, 4, 6, 5, 8, 1, 3, 6, 5, 8, 4, 3, 9, 3, 4], [1]*15]))
trainy = np.array([1, 6, 5, 5, 5, 3, 6, 4, 2, 7, 7, 1, 8, 3, 4])

# Plot points
b = LR1D(trainX, trainy)
plt.plot(trainX[:, 0], trainy, 'ko')

# Plot line
x = np.linspace(start=0, stop=10, num=10)
y = list(map(lambda elem: elem * b[0] + b[1], x))
plt.plot(x, y, 'r-')

plt.show()
