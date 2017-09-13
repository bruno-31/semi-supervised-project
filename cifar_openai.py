"""
The implementation in Tensorflow of OpenAI's GAN defined in the Improved techniques for training GANs paper (2016).
"""
import logging
import tensorflow as tf
import openai_tf_weightnorm as otw
from hyperparam import *

bs = BATCH_SIZE

logger = logging.getLogger(__name__)

def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)
init_bias = None


#
# DISCRIMINATOR
#
def discriminator(inp, is_training, num_classes, init, **kwargs):
    
    x = inp

    #adding noise for regularization
    with tf.variable_scope('gaussian_noise_layer'):
        x = otw.gaussian_noise_layer(x, 0.2)

    with tf.variable_scope('conv1'):
        x = otw.conv2d(x, 96, nonlinearity=leakyReLu, name='conv1', init=init)
    with tf.variable_scope('conv2'):
        x = otw.conv2d(x, 96, nonlinearity=leakyReLu, name = "conv2", init=init)
    with tf.variable_scope('conv3'):
        x = otw.conv2d(x, 96, nonlinearity=leakyReLu, stride=[2,2], name = "conv3", init=init)

    x = tf.layers.dropout(x, 0.5, training=is_training)

    with tf.variable_scope('conv4'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, name = "conv4", init=init)
    with tf.variable_scope('conv5'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, name = "conv5", init=init)
    with tf.variable_scope('conv6'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, stride=[2,2], name = "conv6", init=init)

    x = tf.layers.dropout(x, 0.5, training=is_training)

    with tf.variable_scope('conv7'):
        x = otw.conv2d(x, 192, nonlinearity=leakyReLu, name = "conv7", init=init)

    x = tf.reshape(x, [bs,8,8,192])

    #network-in-network layers
    with tf.variable_scope('nin1'):
        x = otw.nin(x, 192, name="nin1", nonlinearity=leakyReLu, init=init)
    with tf.variable_scope('nin2'):
        x = otw.nin(x, 192, name="nin2", nonlinearity=leakyReLu, init=init)

    #global pooling
    x = tf.layers.average_pooling2d(x,8,4)

    #minibatch-discrimination, based on the Improved techniques for GANs paper
    x = tf.squeeze(x, [1,2])
    # with tf.variable_scope('minibatch-discrimination'):
    #     x = otw.minibatch_disrimination(x, 192, 100, 100, name='minibatch-discrimination')

    # The name parameters here are crucial!
    # The order of definition and inclusion in output is crucial as well! You must define y1 before y2, and also include
    # them in output in the order.
    with tf.variable_scope('discriminator'):
        with tf.variable_scope('fc1'):
            y1 = otw.dense(x, 1000, name='fc1', nonlinearity=leakyReLu, init=init)
        with tf.variable_scope('fc2'):
            y1 = otw.dense(y1, num_classes, name='fc2', nonlinearity=None, init=init)

    return y1


#
# GENERATOR
#
def generator(inp, is_training, output_size, init, **kwargs):
    # We only allow the discriminator model to work on CIFAR-sized data.
    assert output_size == IMAGE_DIM
    
    # [batch_size, init_z + init_label]
    x = tf.contrib.layers.flatten(inp)

    with tf.variable_scope('dense4'):
        x = otw.dense(x, 4*4*512, name='fc4', kernel_initializer=init_kernel, wn=False, init=init)
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training, name='fc4/batch_norm')

    x = tf.reshape(x, [bs,4,4,512])

    # Transposed convolution outputs [batch, 8, 8, 256]
    with tf.variable_scope('deconv1'):
        x = otw.deconv2d(x, 256, filter_size=[5,5], stride=[2,2], padding='SAME', name="deconv1", wn=False, init=init)
        x = tf.nn.relu(x)  
        x = tf.layers.batch_normalization(x, training=is_training, name='deconv1/batch_norm')

    # Transposed convolution outputs [batch, 16, 16, 128]
    with tf.variable_scope('deconv2'):
        x = otw.deconv2d(x, 128, filter_sieze=[5,5], stride=[2,2], padding='SAME', name="deconv2", wn=False, init=init)
        x = tf.nn.relu(x)    
        x = tf.layers.batch_normalization(x, training=is_training, name='deconv2/batch_norm')

    # Transposed convolution outputs [batch, 32, 32, 3]
    with tf.variable_scope('output_generator'):
        x = otw.deconv2d(x, 3, filter_size=[5,5], stride=[2,2], nonlinearity=tf.tanh, name = "output_generator", init=init)
        return x