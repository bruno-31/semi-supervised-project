import tensorflow as tf

init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)


def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def discriminator(inp, is_training):
    activation = leakyReLu
    x = inp
    x = gaussian_noise_layer(x, std=0.15)

    with tf.variable_scope('conv_1'):
        x = tf.layers.conv2d(x, 96, 3, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = activation(x)
    with tf.variable_scope('conv_2'):
        x = tf.layers.conv2d(x, 96, 3, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = activation(x)
    with tf.variable_scope('conv_3'):
        x = tf.layers.conv2d(x, 96, 3, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = activation(x)

    x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
    x = tf.layers.dropout(x, 0.5, training=is_training)

    with tf.variable_scope('conv_4'):
        x = tf.layers.conv2d(x, 192, 3, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = activation(x)
    with tf.variable_scope('conv_5'):
        x = tf.layers.conv2d(x, 192, 3, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = activation(x)
    with tf.variable_scope('conv_6'):
        x = tf.layers.conv2d(x, 192, 3, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = activation(x)

    x = tf.layers.max_pooling2d(x, 2, 2, padding='SAME')
    x = tf.layers.dropout(x, 0.5, training=is_training)

    with tf.variable_scope('conv_7'):
        x = tf.layers.conv2d(x, 192, 3, padding='VALID')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = activation(x)
    with tf.variable_scope('conv_8'):
        x = tf.layers.conv2d(x, 192, 1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = activation(x)

        intermediate_layer = x

    with tf.variable_scope('conv_9'):
        x = tf.layers.conv2d(x, 192, 1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = activation(x)

    x = tf.layers.average_pooling2d(x, pool_size=6, strides=1)
    x = tf.squeeze(x)

    with tf.variable_scope('fc'):
        logits = tf.layers.dense(x, units=10, activation=None)



    return logits, intermediate_layer


def generator(z_seed, is_training):
    x = z_seed
    with tf.variable_scope('dense'):
        x = tf.layers.dense(x, 4 * 4 * 512, name='fc1', activation=tf.nn.relu, kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x,training=is_training)

    x = tf.reshape(x, [-1, 4, 4, 512])

    with tf.variable_scope('deconv_1'):
        x = tf.layers.conv2d_transpose(x, 256, 5, 2, padding='SAME', name='deconv1', activation=tf.nn.relu,
                                       kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training)

    with tf.variable_scope('deconv_2'):
        x = tf.layers.conv2d_transpose(x, 128, 5, 2, padding='SAME', name='deconv2', activation=tf.nn.relu,
                                       kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training)

    with tf.variable_scope('deconv_3'):
        x = tf.layers.conv2d_transpose(x, 3, 5, 2, padding='SAME', name='output_generator', activation=tf.nn.tanh,
                                       kernel_initializer=init_kernel)

    return x
