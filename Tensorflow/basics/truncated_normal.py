# 2017-03-11 jkang

import tensorflow as tf
W = tf.Variable(tf.truncated_normal([20, 30]), name='var_weights')
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W)
