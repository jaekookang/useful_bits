# # Simple Character-level Language Model using vanilla RNN
# 2017-04-11 jkang  
# Python3.5  
# TensorFlow1.0.1  
#   
# - input:  &nbsp;&nbsp;'hello_world_good_morning_see_you_hello_grea'  
# - output: 'ello_world_good_morning_see_you_hello_great'  
# 
# ### Reference:  
# - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py
# - https://github.com/aymericdamien/TensorFlow-Examples
# - https://hunkim.github.io/ml/
# 
# ### Comment:  
# - 단어 단위가 아닌 문자 단위로 훈련함
# - 하나의 example만 훈련에 사용함
# - Cell의 종류는 BasicRNNCell을 사용함 (첫번째 Reference 참조)
# - dynamic_rnn방식 사용 (기존 tf.nn.rnn보다 더 시간-계산 효율적이라고 함)
# - AdamOptimizer를 사용

import tensorflow as tf
import numpy as np

# Input/Ouput data
char_raw = 'hello_world_good_morning_see_you_hello_great'
char_list = list(set(char_raw))
char_to_idx = {c: i for i, c in enumerate(char_list)}
idx_to_char = {i: c for i, c in enumerate(char_list)}
char_data = [char_to_idx[c] for c in char_raw]
char_data_one_hot = tf.one_hot(char_data, depth=len(
    char_list), on_value=1., off_value=0., axis=1, dtype=tf.float32)
char_input = char_data_one_hot[:-1, :]  # 'hello_world_good_morning_see_you_hello_grea'
char_output = char_data_one_hot[1:, :]  # 'ello_world_good_morning_see_you_hello_great'
with tf.Session() as sess:
    char_input = char_input.eval()
    char_output = char_output.eval()

# Learning parameters
learning_rate = 0.001
max_iter = 200

# Network Parameters
n_input_dim = char_input.shape[1]
n_input_len = char_input.shape[0]
n_output_dim = char_output.shape[1]
n_output_len = char_output.shape[0]
n_hidden = 100

# TensorFlow graph
# (batch_size) x (time_step) x (input_dimension)
x = tf.placeholder(tf.float32, [1, None, n_input_dim])
# (batch_size) x (time_step) x (output_dimension)
y = tf.placeholder(tf.float32, [1, None, n_output_dim])

# Parameters
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_output_dim], seed=1))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_output_dim], seed=1))
}

# RNN-cell
def RNN(x, weights, biases):
    rnn = tf.contrib.rnn.BasicRNNCell(n_hidden)
    outputs, states = tf.nn.dynamic_rnn(rnn, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Learning
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_x = char_input.reshape((1, char_input.shape[0], n_input_dim))
    train_y = char_output.reshape((1, char_output.shape[0], n_output_dim))
    for i in range(max_iter):
        _, loss, p = sess.run([optimizer, cost, pred],
                              feed_dict={x: train_x, y: train_y})
        pred_out = np.argmax(p, axis=1)
        print('Epoch: {:>4}'.format(i + 1), '/', str(max_iter),
              'Cost: {:4f}'.format(loss), 'Predict:', ''.join([idx_to_char[i] for i in pred_out]))


# ### Result:
# - 대략 23번째 Epoch에서부터 이미 첫번째 단어 (hello)를 제외하고 모두 제대로 예측함
# - 67번째 Epoch에 이르러서 모든 연쇄를 정확히 예측함
# - LSTM의 경우보다 예측을 더 빨리 (Epoch 비교시) 잘 해냄 (Cost 비교시)
