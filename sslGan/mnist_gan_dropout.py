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


def discriminator(inp,is_training):
    x = inp
    with tf.variable_scope('dense1'):
        x = gaussian_noise_layer(x, std=0.3)
        x = tf.layers.dense(x, 1000, name='fc1', activation=tf.nn.relu)
    with tf.variable_scope('dense2'):
        x = tf.layers.dense(x, 500, name='fc1', activation=tf.nn.relu)
        x = tf.layers.dropout(x,0.5, is_training)
    with tf.variable_scope('dense3'):
        x = tf.layers.dense(x, 250, name='fc1', activation=tf.nn.relu)
        x = tf.layers.dropout(x,0.5, is_training)

    with tf.variable_scope('dense4'):
        x = tf.layers.dense(x, 250, name='fc1', activation=tf.nn.relu)
        x = tf.layers.dropout(x,0.5, is_training)

    inter_layer = x
    with tf.variable_scope('dense5'):
        x = tf.layers.dense(x, 250, name='fc1', activation=tf.nn.relu)
        x = tf.layers.dropout(x,0.5, is_training)

    with tf.variable_scope('dense6'):
        logits = tf.layers.dense(x, 11, name='fc1', activation=None)

    real_class_logits, fake_class_logits = tf.split(logits, [10, 1], 1)

    mx = tf.reduce_max(real_class_logits, 1, keep_dims=True)
    stable_real_class_logits = real_class_logits - mx

    gan_logits = tf.log(tf.reduce_sum(tf.exp(stable_real_class_logits), 1)) + tf.squeeze(mx) - fake_class_logits

    return logits, gan_logits, inter_layer


def generator(batch_size,is_training):

    # z_seed = tf.truncated_normal([batch_size, 100], mean=0, stddev=1, name='z_seed')
    z_seed = tf.random_uniform([batch_size, 100],name='z_seed')

    with tf.variable_scope('dense1'):
        x = tf.layers.dense(z_seed, 500, name='fc1', activation=tf.nn.softplus)
        x = tf.layers.batch_normalization(x,training=is_training)
    with tf.variable_scope('dense2'):
        x = tf.layers.dense(x, 500, name='fc1', activation=tf.nn.softplus)
        x = tf.layers.batch_normalization(x,training=is_training)
    with tf.variable_scope('dense3'):
        x = tf.layers.dense(x, 28*28, name='fc1', activation=tf.nn.sigmoid)
        # x = tf.nn.l2_normalize(x,dim=[0,1])

    return x
