import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

digits = datasets.load_digits()
#print(digits.images[0])
#print(digits.target[0])
plt.figure()
#plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation = 'nearest')
x_train = digits.images[0:1000]
y_train = digits.target[0:1000]
x_test = [digits.images[346]]
y_test = []
num = len(x_train)
num1 = len(x_test)
distance = np.zeros(num)
for j in range(num1):
    for i in range(num):
        distance[i] = dist(x_train[i], x_test[j])
    min_index = np.argmin(distance)
    y_test.append(y_train[min_index])
plt.imshow(digits.images[346], cmap=plt.cm.gray_r, interpolation='nearest')
print(y_test[0])
plt.show()
