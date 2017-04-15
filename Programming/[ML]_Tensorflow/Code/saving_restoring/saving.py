# # Example of saving TensorFlow variables
# 2017-04-11 jkang  
# Python3.5  
# TensorFlow1.0.1  
# 
# Reference:  
# - https://www.tensorflow.org/programmers_guide/variables
# - https://github.com/maestrojeong/tensorflow_basic

import tensorflow as tf
import os

# Make save directory
if not os.path.exists('save'):
    os.mkdir('save')

# Define variables
var1 = tf.Variable(2., name='var1'); print('var1:', var1)
var2 = tf.Variable(5., name='var2'); print('var2:', var2)
var3 = tf.add(var1, var2, name='add'); print('var3:', var3)

save_dir = 'save/all_var.ckpt' # Make sure to include 
                               #'directory_name/saving_file_name.ckpt'
saver = tf.train.Saver() # Initiate Saver object

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, save_dir) # Save
    
    print('var1 = {}'.format(sess.run(var1)))
    print('var2 = {}'.format(sess.run(var2)))
    print('add = {}'.format(sess.run(var3)))

