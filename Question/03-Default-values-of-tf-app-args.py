import tensorflow as tf
flags = tf.app.flags
flags.FLAGS._parse_flags()
for key, value in flags.FLAGS.__dict__['__flags'].items():
    print(key, value)
