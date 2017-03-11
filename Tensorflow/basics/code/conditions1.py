# 2017-03-11 jkang
# practice tf.cond
# ref: http://web.stanford.edu/class/cs20si

import tensorflow as tf

x = tf.random_uniform([], -1, 1)  # random value from -1 ~ 1
y = tf.random_uniform([], -1, 1)  # random value from -1 ~ 1
out = tf.cond(tf.less(x, y), lambda: tf.add(x, y), lambda: tf.sub(x, y))
'''
if 1st arg of tf.cond is TRUE:
   tf.add(x,y) is run
if 1st arg of tf.cond is FALSE:
   tf.sub(x,y) is run
'''

sess = tf.InteractiveSession()
print(sess.run(tf.less(x, y)))
print(sess.run(out))
sess.close()
