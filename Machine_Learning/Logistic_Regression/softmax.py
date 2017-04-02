# softmax function
# 2017-03-30 jkang
# ref: Machine Learning by Andrew Ng on Corsera
import numpy as np

'''
e.g.
In classification task,
z = W*x + b
x shape: (example) x (feature)
'''

def softmax(x):
    rowmax = np.max(x, axis=1)
    x -= rowmax.reshape((x.shape[0] ,1)) # for numerical stability
    x = np.exp(x)
    sum_x = np.sum(x, axis=1).reshape((x.shape[0],1))
    return x / sum_x

z = np.random.random((10,3)) # (10 examples) x (3 features)
pred = softmax(z)
print(pred)

