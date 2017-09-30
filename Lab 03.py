import numpy as np

# Extract data from file
data = list(map(lambda line:  line.strip().split(','), open('iris.data', 'r').readlines()))

# 10 fold
K, N = 10, len(data)

X = np.concatenate((np.vectorize(lambda n: float(n))(np.array(data)[:, :-1]), np.ones(shape=[N, 1])), 1)
y = np.array([[1, 0, 0]]*50 + [[0, 1, 0]]*50 + [[0, 0, 1]]*50)

k = int(N/K)
result = np.zeros(shape=[100, 1])

# 100 times CVs
for j in range(100):

    ridx = np.arange(N)
    np.random.shuffle(ridx)
    local_result = np.zeros(shape=[K, 1])

    for i in range(K):
        s1, s2 = (1+k*((i+1)-1)), ((i+1)*k)
        testy = y[ridx[s1:s2], :]
        testX = X[ridx[s1:s2], :]
        trainy = y[-ridx[s1:s2], :]
        trainX = X[-ridx[s1:s2], :]
        bs = np.linalg.inv(np.transpose(trainX)@trainX) @ (np.transpose(trainX)@trainy)
        t = testX@bs
        local_result[i] = len(np.where(t.argmax(axis=1) == testy.argmax(axis=1))[0]) / k

    result[j] = np.mean(local_result)

print(np.mean(result))
print(np.std(result))
