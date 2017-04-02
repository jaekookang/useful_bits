# helper functions for tensorflow
# 2017-03-09 jkang
# ref: http://pythonkim.tistory.com/62

import tensorflow as tf
import numpy as np


def show_constant(val):
    sess = tf.InteractiveSession()
    print(val.eval())
    sess.close()


def show_variable(val):
    sess = tf.InteractiveSession()
    val.initializer.run()
    print(val.eval())
    sess.close()


def show_detail(val):
    sess = tf.InteractiveSession()
    name = val.name
    shape = tuple(dim.value for dim in val.get_shape())
    rank = len(shape)
    print('name:', name)
    print('shape:', shape)
    print('rank:', rank)
    sess.close()


def show_operation(ops):
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    out = sess.run(ops)
    sess.close()
    return out
