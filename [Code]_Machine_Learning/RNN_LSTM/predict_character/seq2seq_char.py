# Simple character-level prediction using RNN
# 2017-03-30 jkang
# 
# 'hello_world_good_morning_see_you_hello_great'
#
# input:  'ello_world_good_morning_see_you_hello_great'
# output: 'hello_world_good_morning_see_you_hello_grea'
# 
# Python3.5
# Tensorflow1.0.1
# ref: https://hunkim.github.io/ml/

import tensorflow as tf
import numpy as np

# Make input data
char_raw = 'hello_world_good_morning_see_you_hello_great'
char_list = list(set(char_raw))
char_idx = {c: i for i, c in enumerate(char_list)} # character with index
char_data = [char_idx[c] for c in char_raw]
char_data_onehot = tf.one_hot(char_data, 
                              depth=len(char_list), 
                              on_value=1., 
                              off_value=0.,
                              axis=1, 
                              dtype=tf.float32)
char_input = char_data_onehot[:-1] # 'ello_world_good_morning_see_you_hello_great'
char_output = char_data_onehot[1:] # 'hello_world_good_morning_see_you_hello_grea'
print('char_data:', char_data)
print('char_data_onehot:', char_data_onehot.shape)
print('char_input:', char_input.shape)
print('char_output:', char_output.shape)

# Set configurations
n_char = len(char_list)
rnn_size = n_char # number of one-hot coding vectors == output size for each cell
n_timestep = char_input.shape.as_list()[0] # length of the input
batch_size = 1  # one example
max_iter = 300

# Set RNN
rnn_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
init_state = tf.zeros([batch_size, rnn_cell.state_size])
input_split = tf.split(value=char_input, num_or_size_splits=n_timestep, axis=0)
outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, input_split, init_state)

# logits: A 3D Tensor of shape [batch_size x sequence_length x num_decoder_symbols] and dtype float. 
# targets: A 2D Tensor of shape [batch_size x sequence_length] and dtype int.
# weights: A 2D Tensor of shape [batch_size x sequence_length] and dtype float.
# logits = tf.reshape(tf.concat(values=char_output, axis=1), [batch_size, n_timestep, rnn_size])
logits = tf.reshape(outputs, [batch_size, n_timestep, rnn_size])
targets = tf.reshape(char_data[1:], [batch_size, n_timestep]) # target as index
weights = tf.ones((batch_size, n_timestep))

loss = tf.contrib.seq2seq.sequence_loss(logits, targets, weights)
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(learning_rate = 0.01, decay = 0.9).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(max_iter):
        _, c = sess.run([train_op, cost])
        result = sess.run(tf.arg_max(logits, 2))
        print('Epoch: {:>4}'.format(i + 1), '/', str(max_iter),
              'Cost: {:4f}'.format(c), 'Predict:', ''.join([char_list[t] for t in result[0]]))

