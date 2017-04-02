# 2017-03-11 jkang
# simple logistic regression
# Python3.5
# Tensorflow1.0.1
# ref:
# - http://web.stanford.edu/class/cs20si/
# - iris dataset from Matlab Neural Network example
#
# Input: iris data (4 features)
# Output: iris label (3 categories)

import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


learning_Rate = 0.01
batch_size = 10
max_epochs = 30

irisInputs_tmp = sio.loadmat('irisInputs.mat')
irisInputs = irisInputs_tmp['irisInputs'].T
irisTargets_tmp = sio.loadmat('irisTargets')
irisTargets = irisTargets_tmp['irisTargets'].T

X = tf.placeholder(tf.float32, [batch_size, 4], name='irisInputs')
Y = tf.placeholder(tf.float32, [batch_size, 3], name='irisTargets')

w = tf.Variable(np.zeros((4, 3)), name='weight', dtype=np.float32)
b = tf.Variable(np.zeros((1, 3)), name='bias', dtype=np.float32)

logits = tf.matmul(X, w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_Rate).minimize(loss)

def softmax(x):
    ex_val = np.exp(x - np.max(x))
    return ex_val / ex_val.sum()


with tf.Session() as sess:
    # training
    writer = tf.summary.FileWriter('./graph', sess.graph)
    sess.run(tf.global_variables_initializer())
    n_batches = int(irisTargets.shape[0] / batch_size)
    for i in range(max_epochs):
        total_loss = 0
        for ibatch in range(n_batches):
            x_batch = irisInputs[batch_size *
                                 ibatch: batch_size * ibatch + batch_size]
            y_batch = irisTargets[batch_size *
                                  ibatch: batch_size * ibatch + batch_size]
            _, loss_batch = sess.run([optimizer, loss], feed_dict={
                                     X: x_batch, Y: y_batch})
            total_loss += loss_batch
        print('Average loss at epoch {0}: {1}'.format(
            i, total_loss / n_batches))
    print('Optimization finished!')
    weights, bias = sess.run([w, b])
    writer.close()


# testing
rand_idx = np.random.permutation(irisInputs.shape[0])[0]
x_data = irisInputs[rand_idx]
y_data = irisTargets[rand_idx]
pred = softmax(np.dot(x_data, weights) + bias)
print('Y:', y_data)
print('pred:', np.argmax(pred) + 1, 'th element')

