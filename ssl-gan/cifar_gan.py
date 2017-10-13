import tensorflow as tf
import nn

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


def discriminator(inp, is_training, init=False):
    counter = {}
    x = tf.reshape(inp, [-1,32,32,3])
    x = tf.layers.dropout(x,rate=0.2, training=is_training, name='dropout_0')

    x = nn.conv2d(x, 96, nonlinearity=leakyReLu, init=init, counters=counter)
    x = nn.conv2d(x, 96, nonlinearity=leakyReLu, init=init, counters=counter)
    x = nn.conv2d(x, 96, stride=[2, 2], nonlinearity=leakyReLu, init=init, counters=counter)
    # 16*16
    x = tf.layers.dropout(x,rate=0.5, training=is_training, name='dropout_1')

    x = nn.conv2d(x, 192, nonlinearity=leakyReLu, init=init, counters=counter)
    x = nn.conv2d(x, 192, nonlinearity=leakyReLu, init=init, counters=counter)
    x = nn.conv2d(x, 192, stride=[2, 2], nonlinearity=leakyReLu, init=init, counters=counter)
    # 8*8
    x = tf.layers.dropout(x,rate=0.5, training=is_training, name='dropout_2')

    x = nn.conv2d(x, 192, padding='VALID', nonlinearity=leakyReLu, init=init, counters=counter)  # 5*5
    x = nn.nin(x, 192, nonlinearity=leakyReLu,init=init,counters=counter)
    x = nn.nin(x, 192, nonlinearity=leakyReLu,init=init,counters=counter)

    print(x)
    intermediate_layer = x

    x = tf.layers.average_pooling2d(x, pool_size=8, strides=1, name='avg_pool_0')
    x = tf.squeeze(x, [1, 2])

    logits = nn.dense(x, 10, nonlinearity=None, init=init, counters=counter)

    return logits, intermediate_layer


def generator(batch_size, is_training, init):
    counter = {}
    z_seed = tf.random_uniform([batch_size, 100])

    x = nn.dense(z_seed, 4 * 4 * 512, init=init, counters=counter)
    x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_1')
    x = tf.nn.relu(x)

    x = tf.reshape(x, [-1, 4, 4, 512])

    x = nn.deconv2d(x, 256,filter_size=[5,5], stride=[2,2],init=init, counters=counter)
    x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_2')
    x = tf.nn.relu(x)

    x = nn.deconv2d(x, 128,filter_size=[5,5], stride=[2,2], init=init, counters=counter)
    x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_3')
    x = tf.nn.relu(x)

    output = nn.deconv2d(x, 3,filter_size=[5,5], stride=[2,2], nonlinearity=tf.nn.tanh, init=init, counters=counter)
    return output
