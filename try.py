import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tryfunc

data = pd.read_csv('data/data1.txt', names=['Population', 'Profit'])  # 读取数据集的数据，格式为DataFrame
data.insert(0, 'x_0', 1)  # 加入x_0特征

X = data.iloc[:, :-1]  # 分离X，Y
Y = data.iloc[:, -1:]

X = np.matrix(X)  # 将DataFrame转化成矩阵形式
Y = np.matrix(Y)

theta = np.matrix(np.zeros(X.shape[1]))  # 根据X维数生成相对于维数的theta，初始值设为0

theta = tryfunc.GradintDescent(X, Y, theta, 0.01)  # 对theta执行梯度下降法，得到最优theta
print(theta)
print(tryfunc.Cost(X, Y, theta))

x = np.linspace(data.Population.min(), data.Population.max(), 100)
y = theta[0, 0] + theta[0, 1] * x

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, y, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
# ax.set_title('Predicted Profit vs. Population Size')
# plt.show()
