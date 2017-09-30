import numpy as np


def data_gaussian2(_np, var):

    # two center points
    a1 = 0.33
    a2 = 0.5
    b1 = 0.66
    b2 = 0.5

    # three classes
    xa1 = np.transpose(np.matrix(np.random.normal(a1, np.sqrt(var),_np,)))
    xa2 = np.transpose(np.matrix(np.random.normal(a2, np.sqrt(var), _np, )))
    xb1 = np.transpose(np.matrix(np.random.normal(b1, np.sqrt(var), _np, )))
    xb2 = np.transpose(np.matrix(np.random.normal(b2, np.sqrt(var),_np,)))

    # class label
    ya = np.ones(shape=[_np, 1])

    # combine three data points
    return np.concatenate((np.concatenate((xa1, xa2, ya), 1), np.concatenate((xb1, xb2, ya - 2), 1)))



def LC(tx, ty, x_test):
    return np.sign(x_test@(np.linalg.inv(np.transpose(tx)@tx)@(np.transpose(tx)@ty)))

data = data_gaussian2(100,0.1)
train = np.concatenate((data[0:90, :], data[100:190, :]))  # 100 data points with 0.1 variation
test = np.concatenate((data[90:100, :], data[190:200, :]))  # please check the function, Data_gaussian2()
trainy = train[:, train.shape[1]-1]  # last column is class label
trainX = np.concatenate((train[:, 0:train.shape[1]-1], np.ones(shape=[len(train), 1])), 1)
testy = test[:, train.shape[1]-1]  # last column is class label
testX = np.concatenate((test[:, 0:train.shape[1]-1], np.ones(shape=[len(test), 1])), 1)
est_y = LC(trainX, trainy, testX)
a = np.transpose(testy).tolist()[0]
b = np.transpose(est_y).tolist()[0]
acc = len(list(filter(lambda value: value == 0, map(lambda pair: pair[0] - pair[1], zip(a, b))))) / len(test)
print(acc)