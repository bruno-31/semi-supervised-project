import tensorflow as tf

h1_size = 56 * 56
h2_size = 350
Z_DIM = 100
IM_HEIGHT = 28
IM_WIDTH = 28
IM_SIZE = IM_HEIGHT * IM_WIDTH


def build_generator(batch_size):
    z_seed = tf.truncated_normal([batch_size, Z_DIM], mean=0, stddev=1, name='z_seed')
    # 1 feature map 56*56
    with tf.variable_scope('fc_layer_1'):
        w1 = tf.get_variable('g_w1', [Z_DIM, h1_size], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('g_b1', [h1_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        h1 = tf.reshape(tf.matmul(z_seed, w1) + b1, shape=[-1, 56, 56, 1])
        h1 = tf.contrib.layers.batch_norm(h1, epsilon=1e-5, scope='bn1', is_training= True)
        h1 = tf.nn.relu(h1)
    # 50 features map 56*56
    with tf.variable_scope('conv_layer_2'):
        w2 = tf.get_variable('g_w2', [3, 3, 1, Z_DIM // 2], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('g_b2', [Z_DIM // 2], dtype=tf.float32, initializer=tf.zeros_initializer())
        h2 = tf.nn.conv2d(h1, w2, strides=[1, 2, 2, 1], padding='SAME')+b2
        h2 = tf.contrib.layers.batch_norm(h2, epsilon=1e-5, scope='bn2', is_training= True)
        h2 = tf.nn.relu(h2)
        h2 = tf.image.resize_images(h2, size=[56, 56])
    # 25 features map 56*56
    with tf.variable_scope('conv_layer_3'):
        w3 = tf.get_variable('g_w3', [3, 3, Z_DIM // 2, Z_DIM // 4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b3 = tf.get_variable('g_b3', [Z_DIM // 4], dtype=tf.float32, initializer=tf.zeros_initializer())
        h3 = tf.nn.conv2d(h2, w3, strides=[1, 2, 2, 1], padding='SAME')+b3
        h3 = tf.contrib.layers.batch_norm(h3, epsilon=1e-5, scope='bn3', is_training= True)
        h3 = tf.nn.relu(h3)
        h3 = tf.image.resize_images(h3, size=[56, 56])
    # output batch*28 *28*1
    with tf.variable_scope('final_conv_layer_4'):
        w4 = tf.get_variable('g_w4', [1, 1, Z_DIM // 4, 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b4 = tf.get_variable('g_b4', [1], dtype=tf.float32, initializer=tf.zeros_initializer())
        g_output = tf.nn.conv2d(h3, w4, strides=[1, 2, 2, 1], padding='SAME')+b4
        g_output = tf.tanh(g_output)
    return g_output


def build_discriminator(x_input, keep_prob):
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
