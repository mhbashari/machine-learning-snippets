import tensorflow as tf
x = tf.Variable(3.0)
g_x = x * x
fog_x = g_x * g_x
opt = tf.train.GradientDescentOptimizer(0.1)
grads = opt.compute_gradients(fog_x)
grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in grads]
apply_placeholder_op = opt.apply_gradients(grad_placeholder)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
grad_vals = sess.run([grad[0] for grad in grads])
print(grad_vals)