"""原理分析"""
import numpy as np


m, j, k = 512, 512, 976

K = np.random.randint(0, 2, size=[m,j+k])
a = np.random.randint(0, 2, size=[j])
b = np.random.randint(0, 2, size=[k])

r1 = np.matmul(K, np.concatenate((a, b)))
print(r1.shape)

K1, K2 = K[:,:j], K[:,j:]
r2 = np.matmul(K1, a) + np.matmul(K2, b)
print(r2.shape)

print(r1 == r2)