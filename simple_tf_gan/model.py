import tensorflow as tf

h1_size = 150
h2_size = 350
Z_DIM = 100
IM_HEIGHT = 28
IM_WIDTH = 28
IM_SIZE = IM_HEIGHT * IM_WIDTH

def build_generator(batch_size):
    z_seed = tf.truncated_normal([batch_size, Z_DIM], mean=0, stddev=1, name='z_seed')
    with tf.name_scope('layer_1'):
        w1 = tf.get_variable('g_w1', [Z_DIM, h1_size], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('g_b1', [h1_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        h1 = tf.nn.relu(tf.matmul(z_seed, w1) + b1)
    with tf.name_scope('layer_2'):
        w2 = tf.get_variable('g_w2', [h1_size, h2_size], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('g_b2', [h2_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    with tf.name_scope('layer_3'):
        w3 = tf.get_variable('g_w3', [h2_size, IM_SIZE], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        b3 = tf.get_variable('g_b3', [IM_SIZE], dtype=tf.float32, initializer=tf.zeros_initializer())
        g_output = tf.nn.tanh(tf.matmul(h2, w3) + b3)
    return g_output


def build_discriminator(x_input, keep_prob):
    with tf.name_scope('layer_d_1'):
        w1 = tf.get_variable('d_w1', [IM_SIZE, h2_size], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('d_b1', [h2_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        h1 = tf.nn.relu(tf.matmul(x_input, w1) + b1)
        h1 = tf.nn.dropout(h1, keep_prob=keep_prob)

    with tf.name_scope('layer_d_2'):
        w2 = tf.get_variable('d_w2', [h2_size, h1_size], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('d_b2', [h1_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        h2 = tf.nn.dropout(h2, keep_prob=keep_prob)

    with tf.name_scope('layer_d_3'):
        w3 = tf.get_variable('d_w3', [h1_size, 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        b3 = tf.get_variable('d_b3', [1], dtype=tf.float32, initializer=tf.zeros_initializer())
        d_output = tf.matmul(h2, w3) + b3
    return d_output