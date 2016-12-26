import tensorflow as tf
flags = tf.app.flags
flags.FLAGS._parse_flags()
print(("\n".join(flags.FLAGS.__dict__['__flags'].keys())))
