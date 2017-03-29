# 2017-03-11 jkang
# practice tf.case
# ref: http://web.stanford.edu/class/cs20si

import tensorflow as tf

x = tf.random_uniform([], -2, 2)
y = tf.random_uniform([], -2, 2)


def f1():
    return tf.add(x, y)


def f2():
    return tf.sub(x, y)


def f3():
    return tf.constant(0, dtype=tf.float32)

val = tf.case({tf.less(x, y): f2, tf.greater(x, y): f1},
              default=f3, exclusive=True)

sess = tf.InteractiveSession()
print(sess.run(tf.less(x, y)))
print(sess.run(tf.greater(x, y)))
print(sess.run(val))
sess.close()
