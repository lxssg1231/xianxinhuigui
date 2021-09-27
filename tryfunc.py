import numpy as np


def Prediction(x, theta):
    return x * theta.T


def Cost(x, y, theta):
    temp = np.power((x * theta.T), 2)
    return np.sum(temp)/(2*len(x))


# x(72,2) y(72,1) theta(1,2) temp(1,2)
def GradintDescent(x, y, theta, alpha):
    temp = np.matrix(np.zeros(theta.shape[1]))
    for i in range(1000):
        temp1 = (x * theta.T) - y
        for j in range(theta.shape[1]):
            temp2 = np.multiply(temp1, x[:, j])
            temp[0, j] = theta[0, j] - (np.sum(temp2) / (len(x)) * alpha)
        theta = temp
    return theta
