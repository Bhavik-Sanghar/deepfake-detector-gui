import tensorflow as tf

def gram_matrix(x):
    x = tf.transpose(x, (0, 3, 1, 2))
    features = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], -1])
    gram = tf.matmul(features, features, transpose_b=True)
    return gram / tf.cast(tf.shape(features)[-1], tf.float32)
