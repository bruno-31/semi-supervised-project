import tensorflow as tf
import nn # OpenAI implemetation of weightnormalization (Salimans & Kingma)

init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)


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
    x = nn.conv2d(x, 96, stride=[2, 2], nonlinearity=leakyReLu, init=init, counters=counter) #=> 16*16
    x = tf.layers.dropout(x,rate=0.5, training=is_training, name='dropout_1')

    x = nn.conv2d(x, 192, nonlinearity=leakyReLu, init=init, counters=counter)
    x = nn.conv2d(x, 192, stride=[2, 2], nonlinearity=leakyReLu, init=init, counters=counter) # => 8*8
    # 8*8
    x = tf.layers.dropout(x,rate=0.5, training=is_training, name='dropout_2')

    x = nn.conv2d(x, 192, padding='VALID', nonlinearity=leakyReLu, init=init, counters=counter)  # 8*8


    x = tf.layers.average_pooling2d(x, pool_size=8, strides=1, name='avg_pool_0') # batch *1*1* 192
    x = tf.squeeze(x, [1, 2])

    intermediate_layer = x

    logits = nn.dense(x, 1, nonlinearity=None, init=init, counters=counter)

    return logits, intermediate_layer


def generator(batch_size, is_training, init):

    counter = {}
    z_seed = tf.random_uniform([batch_size, 100])
    # z_seed = tf.random_normal([batch_size, 100])

    x = z_seed

    with tf.variable_scope('dense_1'):
        x = tf.layers.dense(x, units=4*4*512, kernel_initializer=tf.random_normal_initializer(stddev=0.05))
    # x = nn.dense(z_seed, 4 * 4 * 512, init=init, counters=counter)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_1')

    x = tf.reshape(x, [-1, 4, 4, 512])

    # x = nn.deconv2d(x, 256,filter_size=[5,5], stride=[2,2],init=init, counters=counter) # [batch,256,8,8]
    # x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_2')
    # x = tf.nn.relu(x)
    #
    # x = nn.deconv2d(x, 128,filter_size=[5,5], stride=[2,2], init=init, counters=counter) # [batch,256,16,16]
    # x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_3')
    # x = tf.nn.relu(x)
    with tf.variable_scope('deconv_1'):
        x = tf.layers.conv2d_transpose(x, 256, [5, 5], strides=[2,2], padding='SAME', kernel_initializer=init_kernel)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_2')

    with tf.variable_scope('deconv_2'):
        x = tf.layers.conv2d_transpose(x, 128, [5,5], strides=[2,2], padding='SAME', kernel_initializer=init_kernel)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_3')

    # including weightnorm
    with tf.variable_scope('deconv_3'):
        output = nn.deconv2d(x, 3,filter_size=[5,5], stride=[2,2], nonlinearity=tf.nn.tanh, init=init, counters=counter)
    # with tf.variable_scope('deconv3'):
    #     output = tf.layers.conv2d_transpose(x, 3, [5, 5], strides=[2, 2], padding='SAME', activation=tf.tanh, kernel_initializer=init_kernel)
    # [batch,32,32,3]
    return output
