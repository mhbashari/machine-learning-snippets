import tensorflow as tf

data = tf.constant([1, 2, 3, 4])


def fn(previous_output, current_input):
    return current_input ** 2


def fn2(previous_output, current_input):
    return current_input + previous_output


sess = tf.Session()
run = sess.run(tf.scan(fn=fn, elems=data))
print(run)
run = sess.run(tf.scan(fn=fn2, elems=data))
print(run)
# [ 1  4  9 16]
# [ 1  3  6 10]
