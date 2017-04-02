# simple linear regression
# 2017-03-11 jkang
# Python3.5
# Tensorflow1.0.1
# ref: http://web.stanford.edu/class/cs20si/
#
# input: number of fire
# output: number of theft

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

data_file = 'fire_theft.xls'

book = xlrd.open_workbook(data_file, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

X = tf.placeholder(tf.float64, shape=(), name='NumFire')
Y = tf.placeholder(tf.float64, shape=(), name='NumTheft')

w = tf.Variable(np.zeros(1), name='Weight')
b = tf.Variable(np.zeros(1), name='Bias')

Y_predict = tf.add(tf.multiply(X, w), b)

def huber_loss(labels, predictions, delta=1.0):
    # Huber loss (outlier robust)
    delta = np.array(delta, dtype=np.float64)
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)
loss = huber_loss(Y, Y_predict, delta=1.0)

# loss = tf.square(tf.sub(Y, Y_predict), name='loss')

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graph', sess.graph)

    # online training
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print("Epoch {0}: {1}".format(i, total_loss / n_samples))

    w_value, b_value = sess.run([w, b])

writer.close()

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()

