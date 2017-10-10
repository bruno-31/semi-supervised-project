import tensorflow as tf

h1_size = 56 * 56
h2_size = 350
Z_DIM = 100
IM_HEIGHT = 28
IM_WIDTH = 28
IM_SIZE = IM_HEIGHT * IM_WIDTH


def build_generator(z_seed, is_training):
    # z_seed = tf.truncated_normal([batch_size, Z_DIM], mean=0, stddev=1, name='z_seed')
    batch_size = 100
    h0 = tf.reshape(z_seed, shape=[100, 2, 2, 25])
    h0 = tf.nn.relu(h0)

    with tf.variable_scope('conv_layer_1'):
        output1_shape = [batch_size, 3, 3, 256]
        w1 = tf.get_variable('g_w1', [5, 5, 256, 25], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('g_b1', [256], initializer=tf.constant_initializer(.1))
        h1 = tf.nn.conv2d_transpose(h0, w1, output1_shape, strides=[1, 2, 2, 1], padding='SAME') + b1
        h1 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, scope='bn1', is_training=True)
        h1 = tf.nn.relu(h1)

    with tf.variable_scope('conv_layer_2'):
        output2_shape = [batch_size, 6, 6, 128]
        w2 = tf.get_variable('g_w2', [5, 5, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('g_b2', [128], initializer=tf.constant_initializer(.1))
        h2 = tf.nn.conv2d_transpose(h1, w2, output2_shape, strides=[1, 2, 2, 1], padding='SAME') + b2
        h2 = tf.contrib.layers.batch_norm(h2, center=True, scale=True, scope='bn2', is_training=True)
        h2 = tf.nn.relu(h2)

    with tf.variable_scope('conv_layer_3'):
        output3_shape = [batch_size, 12, 12, 64]
        w3 = tf.get_variable('g_w3', [5, 5, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b3 = tf.get_variable('g_b3', [64], initializer=tf.constant_initializer(.1))
        h3 = tf.nn.conv2d_transpose(h2, w3, output3_shape, strides=[1, 2, 2, 1], padding='SAME') + b3
        h3 = tf.contrib.layers.batch_norm(h3, center=True, scale=True, scope='bn3', is_training=True)
        h3 = tf.nn.relu(h3)

    with tf.variable_scope('conv_layer_4'):
        output4_shape = [batch_size, 28, 28, 1]
        w4 = tf.get_variable('g_w4', [5, 5, 1, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b4 = tf.get_variable('g_b4', [1], initializer=tf.constant_initializer(.1))
        h4 = tf.nn.conv2d_transpose(h3, w4, output4_shape, strides=[1, 2, 2, 1], padding='VALID') + b4
        h4 = tf.contrib.layers.batch_norm(h4, center=True, scale=True, scope='bn4', is_training=True)
        g_output = tf.nn.tanh(h4)
    return g_output


def build_discriminator(x_input):

    with tf.variable_scope('conv_layer_d_1'):
        x_input = tf.reshape(x_input, shape=[-1, 28, 28, 1])

        w1 = tf.get_variable('d_w1', [5, 5, 1, 16], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('d_b1', [16], dtype=tf.float32, initializer=tf.zeros_initializer())
        h1 = tf.nn.relu(tf.nn.conv2d(x_input, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
        h1 = tf.nn.avg_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv_layer_d_2'):
        w2 = tf.get_variable('d_w2', [5, 5, 16, 32], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('d_b2', [32], dtype=tf.float32, initializer=tf.zeros_initializer())
        h2 = tf.nn.relu(tf.nn.conv2d(h1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
        h2 = tf.nn.avg_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('fc_layer_d_3'):
        w3 = tf.get_variable('d_w3', [7 * 7 * 32, 1024], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b3 = tf.get_variable('d_b3', [1024], dtype=tf.float32, initializer=tf.zeros_initializer())
        h2 = tf.reshape(h2, shape=[-1, 7 * 7 * 32])
        h3 = tf.nn.relu(tf.matmul(h2, w3) + b3)

    with tf.variable_scope('fc_layer_d_4'):
        w4 = tf.get_variable('d_w4', [1024, 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b4 = tf.get_variable('d_b4', [1], dtype=tf.float32, initializer=tf.zeros_initializer())
        discriminator_output = tf.matmul(h3, w4) + b4

    return discriminator_output
