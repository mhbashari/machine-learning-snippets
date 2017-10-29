import numpy as np
import tensorflow as tf


def py_input_fn(size=100):
    actual_data = np.random.normal(size=[size])
    return actual_data


plh = tf.placeholder(tf.int8)
data = tf.py_func(py_input_fn, [plh], (tf.double))
with tf.Session() as sess:
    print(sess.run(data,feed_dict={plh:5}))
