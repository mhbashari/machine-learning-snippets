import numpy as np
import tensorflow as tf

matrix = np.random.random([1024, 64])
ids = np.array([0, 5, 17, 33])
print(matrix[ids].shape)
embeddings = tf.Variable(
        tf.random_uniform([1024, 64], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, [0, 5, 17, 33])
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(embed).shape)
