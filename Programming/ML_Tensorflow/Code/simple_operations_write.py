# 2017-03-11 jkang

import tensorflow as tf

a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
x = tf.add(a, b, name='add')
y = tf.mul(a, b, name='mul')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph', sess.graph)
    x, y = sess.run([x, y])
    print('\nx:%s\ny:%s' % (x, y))

writer.close()

# Run tensorboard
# >> tensorboard --logdir='./graph' --port 6006
