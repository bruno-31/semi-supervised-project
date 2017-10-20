import tensorflow as tf
import nn  # OpenAI implemetation of weightnormalization (Salimans & Kingma)

init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)


def leakyReLu(x, alpha=0.2, name=None):
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
    counter = {}
    x = tf.reshape(inp, [-1, 32, 32, 3])

    x = tf.layers.dropout(x, rate=0.2, training=is_training, name='dropout_0')

    x = nn.conv2d(x, 32, nonlinearity=leakyReLu, init=init, counters=counter)
    x = nn.conv2d(x, 32, nonlinearity=leakyReLu, init=init, counters=counter)
    x = nn.conv2d(x, 32, stride=[2, 2], nonlinearity=leakyReLu, init=init, counters=counter)  # => 16*16
    x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_1')

    x = nn.conv2d(x, 64, nonlinearity=leakyReLu, init=init, counters=counter)
    x = nn.conv2d(x, 32, nonlinearity=leakyReLu, init=init, counters=counter)
    x = nn.conv2d(x, 64, stride=[2, 2], nonlinearity=leakyReLu, init=init, counters=counter)  # => 8*8
    x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_2')

    x = nn.conv2d(x, 128, padding='VALID', nonlinearity=leakyReLu, init=init, counters=counter)  # 8*8
    # x = nn.conv2d(x, 128, padding='SAME', nonlinearity=leakyReLu, init=init, counters=counter)  # 8*8
    x = nn.nin(x, 192, counters=counter, nonlinearity=leakyReLu, init=init)
    x = nn.nin(x, 192, counters=counter, nonlinearity=leakyReLu, init = init)

    x = tf.layers.max_pooling2d(x, pool_size=8, strides=1, name='avg_pool_0')  # batch *1*1* 192
    x = tf.squeeze(x, [1, 2])

    intermediate_layer = x

    logits = nn.dense(x, 10, nonlinearity=None, init=init, counters=counter)

    return logits, intermediate_layer


def generator(z_seed, is_training, init):
    counter = {}
    x = z_seed
    with tf.variable_scope('dense_1'):
        x = tf.layers.dense(x, units=4 * 4 * 512, kernel_initializer=tf.random_normal_initializer(stddev=0.05))
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_1')

    x = tf.reshape(x, [-1, 4, 4, 512])

    with tf.variable_scope('deconv_1'):
        x = tf.layers.conv2d_transpose(x, 256, [5, 5], strides=[2, 2], padding='SAME', kernel_initializer=init_kernel)
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_2')

    with tf.variable_scope('deconv_2'):
        x = tf.layers.conv2d_transpose(x, 128, [5, 5], strides=[2, 2], padding='SAME', kernel_initializer=init_kernel)
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_3')
    # including weightnorm     # [batch,32,32,3]
    with tf.variable_scope('deconv_3'):
        output = nn.deconv2d(x, num_filters=3, filter_size=[5, 5], stride=[2, 2], nonlinearity=tf.tanh, init=init,
                             counters=counter)
    # with tf.variable_scope('deconv_3'):
    #     output = tf.layers.conv2d_transpose(x, 3, [5, 5], strides=[2, 2], padding='SAME', activation=tf.tanh, kernel_initializer=init_kernel)
    return output


def deconv_weight_norm(x, num_filters, output_shape, filter_size=[5, 5], strides=[2, 2], activation=None):

    V = tf.get_variable('V', shape=filter_size + [num_filters, int(x.get_shape()[-1])], dtype=tf.float32,
                        initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
    g = tf.get_variable('g', [], tf.float32, trainable=True)
    b = tf.get_variable('b', [num_filters], tf.float32)

    W = g * tf.nn.l2_normalize(V,[0,1,3])
    x = tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=strides)
    if activation is not None:
        x = activation(x)
    return x