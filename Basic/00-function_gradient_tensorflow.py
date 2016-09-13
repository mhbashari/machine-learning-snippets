import tensorflow as tf

def get_gradient(fx):
    opt = tf.train.GradientDescentOptimizer(0.1)
    grads = opt.compute_gradients(fx)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    grad_vals = sess.run([grad[0] for grad in grads])
    return grad_vals

point = 3.0
x = tf.Variable(float(point))
g_x = x * x
fog_x = g_x * g_x
print(get_gradient(fog_x))
# [108.0]