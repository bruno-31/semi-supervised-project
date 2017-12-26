import tensorflow as tf
import nn

init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)
init_w = tf.random_normal_initializer(mean=0, stddev=0.1)


def gaussian_noise_layer(input_layer, std, deterministic):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    y= tf.cond(deterministic, lambda :input_layer, lambda :input_layer+noise)
    return y


def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)


def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def l2normalize(x):
    epsilon = 1e-6
    w = tf.get_variable('w', shape=[500, 28*28],initializer=init_kernel)
    bias = tf.get_variable('b',shape=[28*28], initializer=tf.zeros_initializer())
    s = tf.get_variable('s', shape=[],dtype=tf.float32,trainable=True, initializer=tf.ones_initializer)
    w = w * s / (tf.sqrt(epsilon+tf.reduce_sum(tf.square(w))))
    out = tf.matmul(x,w)+bias
    return tf.nn.sigmoid(out)

def discriminator(inp, is_training, init=False,reuse=False, getter =None):
    with tf.variable_scope('discriminator_model', reuse=reuse,custom_getter=getter):

        counter = {}
        x = inp

        x = gaussian_noise_layer(x, std=0.3,deterministic= ~is_training)
        x = nn.dense(x, 1000, nonlinearity=tf.nn.relu, init=init, counters=counter,train_scale=False, init_w=init_w)

        x = gaussian_noise_layer(x, std=0.5, deterministic=~is_training)
        x = nn.dense(x, 500, nonlinearity=tf.nn.relu, init=init, counters=counter,train_scale=False, init_w=init_w)

        x = gaussian_noise_layer(x, std=0.5, deterministic=~is_training)
        x = nn.dense(x, 250, nonlinearity=tf.nn.relu, init=init, counters=counter,train_scale=False, init_w=init_w)

        x = gaussian_noise_layer(x, std=0.5, deterministic=~is_training)
        x = nn.dense(x, 250, nonlinearity=tf.nn.relu, init=init, counters=counter, train_scale=False, init_w=init_w)

        x = gaussian_noise_layer(x, std=0.5, deterministic=~is_training)
        x = nn.dense(x, 250, nonlinearity=tf.nn.relu, init=init, counters=counter, train_scale=False, init_w=init_w)

        inter_layer = x

        x = gaussian_noise_layer(x, std=0.5, deterministic=~is_training)
        logits = nn.dense(x, 10, nonlinearity=None, init=init, counters=counter, train_scale=True, init_w=init_w)


        return logits, inter_layer

def generator(z_seed,is_training, init=False,reuse=False):
    with tf.variable_scope('generator_model', reuse=reuse):
        counter = {}
        with tf.variable_scope('dense1'):
            x = tf.layers.dense(z_seed, 500, name='fc1', activation=None)
            x = tf.layers.batch_normalization(x,training=is_training)
            x = tf.nn.softplus(x)
            # x = tf.nn.relu(x)
        with tf.variable_scope('dense2'):
            x = tf.layers.dense(x, 500, name='fc1', activation=None)
            x = tf.layers.batch_normalization(x,training=is_training)
            x = tf.nn.softplus(x)
            # x = tf.nn.relu(x)

        with tf.variable_scope('dense3'):
            x = nn.dense(x,28**2, tf.sigmoid,init=init,counters=counter, train_scale=True)

        return x
