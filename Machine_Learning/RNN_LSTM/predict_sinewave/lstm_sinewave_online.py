# Simple LSTM regression
# 2017-03-16 jkang
# Python3.5
# Tensorflow1.0.1
#
# input: sinewaves (varying frequency, amplitude and duration)
# output: sinewaves (shifted inputs)
#
# no mini-batch mode (online training)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Input, Ouput dataset
n_examples = 100
srate = 200  # Hz
sin_in = {}
sin_out = {}
for i in range(n_examples):
    freq = np.random.random(1) * 10 + 1  # 1 <= freq < 11 Hz
    amplitude = np.random.random(1) * 10
    duration = np.random.random(1) * 5 + 5  # sample from 5 ~ 10 sec
    t = np.linspace(0, duration, duration * srate + 1)
    sin = np.sin(2 * np.pi * freq * t) * amplitude
    key = 's' + str(i + 1)
    shift = int(srate/freq*1/4) # 1/4 phase shift to make input & output orthogonal
    sin_in[key] = sin[:-shift]
    sin_out[key] = sin[shift:]


# Hyper-Parameters
learning_rate = 0.01
max_iter = 20

# Network Parameters
n_input_dim = 1
# n_input_len = len(sin_in)
# n_output_len = len(sin_out)
n_hidden = 100
n_output_dim = 1

# TensorFlow graph
# (batch_size) x (time_step) x (input_dimension)
x_data = tf.placeholder(tf.float32, [1, None, n_input_dim])
# (batch_size) x (time_step) x (output_dimension)
y_data = tf.placeholder(tf.float32, [1, None, n_output_dim])

# Parameters
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_output_dim]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output_dim]))
}


def RNN(x, weights, biases):
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0) # Make RNNCell
    outputs, states = tf.nn.dynamic_rnn(cell, x, time_major=False, dtype=tf.float32)
    '''
    **Notes on tf.nn.dynamic_rnn**

    - 'x' can have shape (batch)x(time)x(input_dim), if time_major=False or 
                         (time)x(batch)x(input_dim), if time_major=True
    - 'outputs' can have the same shape as 'x'
                         (batch)x(time)x(input_dim), if time_major=False or 
                         (time)x(batch)x(input_dim), if time_major=True
    - 'states' is the final state, determined by batch and hidden_dim
    '''
    
    # outputs[-1] is outputs for the last example in the mini-batch
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x_data, weights, biases)
cost = tf.reduce_mean(tf.squared_difference(pred, y_data))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step <= max_iter:
        loss = 0
        for i in range(n_examples):
            key = 's' + str(i + 1)
            x_train = sin_in[key].reshape((1, len(sin_in[key]), n_input_dim))
            y_train = sin_out[key].reshape(
                (1, len(sin_out[key]), n_output_dim))
            c, _ = sess.run([cost, optimizer], feed_dict={
                            x_data: x_train, y_data: y_train})
            loss += c
        mean_mse = loss / n_examples

        print('Epoch =', str(step), '/', str(max_iter),
              'Cost = ', '{:.5f}'.format(mean_mse))
        step += 1


# Test
with tf.Session() as sess:
    sess.run(init)
    for i in np.arange(5)[1:]:
        idx = 's' + str(np.random.permutation(n_examples)[0])
        x_test = sin_in[idx].reshape((1, len(sin_in[idx]), n_input_dim))
        pred_out = sess.run(pred, feed_dict={x_data: x_test})
        f, axes = plt.subplots(2, sharey=True)
        axes[0].plot(sin_out[idx])
        axes[1].plot(pred_out)
        plt.show()
        #f.savefig('result_' + idx + '.png')

