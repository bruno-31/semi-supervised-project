import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.misc
import shutil
import os
import conv_model

max_iter = 10000
batch_size = 256
output_path = 'output'
log_path = 'log'
freq_print = 10
freq_save = 100
KEEP_RATE = 0.7
IM_HEIGHT = 28
IM_WIDTH = 28
IM_SIZE = IM_HEIGHT * IM_WIDTH

flags = tf.app.flags
flags.DEFINE_string("log_dir", 'log', "log directory ['log']")
flags.DEFINE_string('out_dir', 'output', 'output directory')
FLAGS = flags.FLAGS

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def store_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    # display function
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], IM_HEIGHT, IM_WIDTH)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
        scipy.misc.imsave(fname, img_grid)


def main(_):
    print(FLAGS.log_dir)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.mkdir(log_path)
    print("start training")
    # collecting data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # placeholders
    x_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, IM_SIZE], name="x_data")
    keep_pl = tf.placeholder(dtype=tf.float32, name="dropout_keep_rate")

    # build model
    with tf.variable_scope("generator_model"):
        x_generated = conv_model.build_generator(batch_size)

    with tf.variable_scope("discriminator_model") as scope:  # we use only one model for discriminator with 2 inputs
        dis_gen = conv_model.build_discriminator(x_generated, keep_pl)
        scope.reuse_variables()
        dis_data = conv_model.build_discriminator(x_data, keep_pl)


    with tf.name_scope('generator_loss'):
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen, labels=tf.ones_like(dis_gen)))
        dis_logits_on_generated = tf.reduce_mean(tf.sigmoid(dis_gen))
    with tf.name_scope('discriminator_loss'):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_data, labels=tf.fill([batch_size,1],0.9)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_gen, labels=tf.zeros_like(dis_gen)))
        d_loss = d_loss_fake + d_loss_real
        dis_logits_on_real = tf.reduce_mean(tf.sigmoid(dis_data))


    # collecting 2 list of training variables corresponding to discriminator and generator
    tvars = tf.trainable_variables()  # return list trainable variables
    d_vars = [var for var in tvars if 'd_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]

    for var in d_vars:  # display trainable vars for sanity check
        print(var.name)
    for var in g_vars:
        print(var.name)

    optimzer = tf.train.AdamOptimizer(0.0001)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        # g_trainer = optimzer.minimize(g_loss, var_list=g_vars, name='generator_trainer')
        g_grads = optimzer.compute_gradients(g_loss, var_list=g_vars)
        # g_grads = list(zip(g_grads, tf.trainable_variables()))
        g_trainer = optimzer.apply_gradients(grads_and_vars=g_grads)

    d_grads = optimzer.compute_gradients(d_loss, var_list=d_vars)
    # d_grads = list(zip(d_grads, tf.trainable_variables()))
    d_trainer = optimzer.apply_gradients(grads_and_vars=d_grads)
    # d_trainer = optimzer.minimize(d_loss, var_list=d_vars, name='discriminator_trainer')

    # summary
    with tf.name_scope('generator'):
        tf.summary.scalar('Generator_loss', g_loss)
    with tf.name_scope('discriminator'):
        tf.summary.scalar('Discriminator_real_loss', d_loss_real)
        tf.summary.scalar('Discriminator_fake_loss', d_loss_fake)
        tf.summary.scalar('Discriminator_total_loss', d_loss)
        tf.summary.scalar('logits_discriminator_on_generated', dis_logits_on_generated)
        tf.summary.scalar('logits_discriminator_on_real', dis_logits_on_real)


    with tf.name_scope('gradient'):
        for grad, var in d_grads:
            tf.summary.histogram(var.name, grad)

        for grad, var in g_grads:
            tf.summary.histogram(var.name, grad)

    # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    # summary_gen = tf.summary.merge([v for v in summaries if "discriminator/" in v.name])
    # summary_cls = tf.summary.merge([v for v in summaries if "generator/" in v.name])

    tf.summary.image('Generated_images', x_generated, 10)  # add 10 generated images to summary
    x_data_reshaped = tf.reshape(x_data, shape=[-1, 28, 28, 1])
    tf.summary.image('data_images', x_data_reshaped, 10)
    merged = tf.summary.merge_all()

    # train
    with tf.Session() as sess:
        # write tensorflow summary for monitoring on tensorboard
        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        for i in range(max_iter):
            x_batch, _ = mnist.train.next_batch(batch_size)
            x_batch = 2 * x_batch.astype(np.float32) - 1  # set image dynamic to [-1 1]

            sess.run(d_trainer, feed_dict={x_data: x_batch, keep_pl: KEEP_RATE})
            sess.run(g_trainer, feed_dict={x_data: x_batch, keep_pl: KEEP_RATE})

            if i % 100 == 0:
                print("step %d" % (i))

            if i % freq_print == 0:
                summary = sess.run(merged, feed_dict={x_data: x_batch, keep_pl: KEEP_RATE})
                writer.add_summary(summary, i)

            if i % freq_save == 0:
                sample_images = sess.run(x_generated, feed_dict={x_data: x_batch, keep_pl: KEEP_RATE})
                # store_result(sample_images, os.path.join(FLAGS.out_dir, "sample%s.jpg" % i))
                #   saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)

        # im_g_sample = sess.run(x_generated)
        # store_result(im_g_sample, os.path.join(output_path, "final_samples.jpg"))


if __name__ == '__main__':
    tf.app.run()
