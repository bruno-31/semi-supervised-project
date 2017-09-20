import tensorflow as tf
import openai_tf_weightnorm as otw


def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)


def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def inference(inp, is_training, num_classes, init):
    x = inp
    # adding noise for regularization
    with tf.variable_scope('gaussian_noise_layer'):
        x = otw.gaussian_noise_layer(x, 0.15)

    with tf.variable_scope('conv_1'):
        x = otw.conv2d(x, 96, nonlinearity=leakyReLu, init=init)
    with tf.variable_scope('conv_2'):
        x = otw.conv2d(x, 96, nonlinearity=leakyReLu, init=init)
    with tf.variable_scope('conv_3'):
        x = otw.conv2d(x, 96, nonlinearity=leakyReLu, init=init)

    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')
    # x = tf.layers.dropout(x, 0.5, training=is_training)

    with tf.variable_scope('conv_4'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, init=init)
    with tf.variable_scope('conv_5'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, init=init)
    with tf.variable_scope('conv_6'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, init=init)

    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')
    x = tf.layers.dropout(x, 0.5, training=is_training)

    with tf.variable_scope('conv_7'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, init=init, pad='VALID')
    with tf.variable_scope('conv_8'):
        x = otw.conv2d(x, 192, filter_size=[1,1], nonlinearity=leakyReLu, init=init)
    with tf.variable_scope('conv_9'):
        x = otw.conv2d(x, 192, filter_size=[1,1], nonlinearity=leakyReLu, init=init)

    x = tf.layers.average_pooling2d(x, pool_size=6, strides=1)

    x = tf.squeeze(x)

    y = otw.dense(x, num_classes, name='fc1', nonlinearity=None, init=init)
    return y
