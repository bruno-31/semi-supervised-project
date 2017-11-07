import tensorflow as tf

init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)


def leakyReLu(x, alpha=0.1, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)


def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def discriminator(inp, is_training, init=False):
    x = tf.reshape(inp, [-1, 32, 32, 3])

    x = tf.layers.conv2d(x, 96, [3, 3], padding='SAME')
    x = tf.layers.batch_normalization(x, training=is_training)
    x = leakyReLu(x)
    x = tf.layers.conv2d(x, 96, [3, 3], padding='SAME')
    x = tf.layers.batch_normalization(x, training=is_training)
    x = leakyReLu(x)
    x = tf.layers.conv2d(x, 96, [3, 3], padding='SAME')
    x = tf.layers.batch_normalization(x, training=is_training)
    x = leakyReLu(x)

    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
    x = tf.layers.dropout(x, rate=0.5, training=is_training)

    x = tf.layers.conv2d(x, 192, [3, 3], padding='SAME')
    x = tf.layers.batch_normalization(x, training=is_training)
    x = leakyReLu(x)
    x = tf.layers.conv2d(x, 192, [3, 3], padding='SAME')
    x = tf.layers.batch_normalization(x, training=is_training)
    x = leakyReLu(x)
    x = tf.layers.conv2d(x, 192, [3, 3], padding='SAME')
    x = tf.layers.batch_normalization(x, training=is_training)
    x = leakyReLu(x)

    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
    x = tf.layers.dropout(x, rate=0.5, training=is_training)

    x = tf.layers.conv2d(x, 192, [3, 3])
    x = tf.layers.batch_normalization(x, training=is_training)
    x = leakyReLu(x)
    x = tf.layers.conv2d(x, 192, [1, 1])
    x = tf.layers.batch_normalization(x, training=is_training)
    x = leakyReLu(x)
    x = tf.layers.conv2d(x, 192, [1, 1])
    x = tf.layers.batch_normalization(x, training=is_training)
    x = leakyReLu(x)

    x = tf.layers.max_pooling2d(x, pool_size=6, strides=1, name='avg_pool_0')
    x = tf.squeeze(x, [1, 2])

    intermediate_layer = x

    logits = tf.layers.dense(x, 10)

    return logits, intermediate_layer


def generator(z_seed, is_training, init):
    return
