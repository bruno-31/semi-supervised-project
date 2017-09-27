import os
import time
import numpy as np
import tensorflow as tf
from data import cifar10_input
import cifar_gan

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size [128]")
flags.DEFINE_integer("moving_average", 100, "moving average [100]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('log_dir', './log', 'log directory')
flags.DEFINE_integer('seed', 1546, 'seed ')
flags.DEFINE_integer('seed_data', 64, 'seed data')
flags.DEFINE_integer('labeled', 400, 'labeled data per class')
flags.DEFINE_float('learning_rate', 0.003, 'learning_rate[0.003]')
FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)

    # Random seed
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    rng_data = np.random.RandomState(FLAGS.seed_data)  # seed shuffling

    # load CIFAR-10
    trainx, trainy = cifar10_input._get_dataset(FLAGS.data_dir, 'train')  # float [-1 1] images
    testx, testy = cifar10_input._get_dataset(FLAGS.data_dir, 'test')
    trainx_unl = trainx.copy()
    trainx_unl2 = trainx.copy()
    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)
    nr_batches_test = int(testx.shape[0] / FLAGS.batch_size)

    '''construct graph'''
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='data_input')
    lbl = tf.placeholder(tf.float32, [FLAGS.batch_size, 10], name='lbl_input')
    z_seed = tf.placeholder(tf.float32, [100], name='z_seed')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    learning_rate_pl = tf.placeholder(tf.float32, [], name='learning_rate_pl')

    with tf.variable_scope('generator_model'):
        gen_inp = cifar_gan.generator(z_seed, is_training_pl)

    with tf.variable_scope('discriminator_model') as dis_scope:
        logits_lab, _ = cifar_gan.discriminator(inp, is_training_pl)
        dis_scope.reuse_variables()
        logits_unl, layer_real = cifar_gan.discriminator(inp, is_training_pl)
        logits_fake, layer_fake = cifar_gan.discriminator(inp, is_training_pl)

    with tf.name_scope('loss_functions'):
        # Taken from improved gan, T. Salimans
        with tf.name_scope('discriminator'):
            l_lab = logits_lab[:, tf.argmax(lbl, axis=1)]
            loss_lab = - tf.reduce_mean(l_lab) + tf.reduce_mean(tf.reduce_logsumexp(logits_lab, axis=1))
            loss_unl = - 0.5 * tf.reduce_mean(tf.reduce_logsumexp(logits_unl, axis=1)) \
                       + 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(logits_unl, axis=1))) \
                       + 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(logits_fake, axis=1)))
            correct_pred = tf.equal(tf.argmax(logits_lab,1), tf.argmax(lbl,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        with tf.name_scope('generator'):
            loss_gen = tf.reduce_mean(tf.square(layer_real-layer_fake))

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl, beta1=0.9)

        with tf.control_dependencies(update_ops_dis):
            train_dis_op = optimizer.minimize(loss_lab+loss_unl, var_list=dvars, name='dis_optimizer')
        with tf.control_dependencies(update_ops_gen):
            train_gen_op = optimizer.minimize(loss_gen, var_list=gvars, name='gen_optimizer')

    # select labeled data
    inds = rng_data.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy = trainy[inds]
    txs = []
    tys = []
    for j in range(10):
        txs.append(trainx[trainy == j][:FLAGS.labeled])
        tys.append(trainy[trainy == j][:FLAGS.labeled])
    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)

    '''//////perform training //////'''
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        train_batch = 0

        for epoch in range(200):
            begin = time.time()

            # construct randomly permuted minibatches
            trainx = []
            trainy = []
            for t in range(int(np.ceil(trainx_unl.shape[0] / float(txs.shape[0])))):
                inds = rng.permutation(txs.shape[0])
                trainx.append(txs[inds])
                trainy.append(tys[inds])
            trainx = np.concatenate(trainx, axis=0)
            trainy = np.concatenate(trainy, axis=0)
            trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]
            trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

            loss_lab, loss_unl, train_err = [0, 0, 0]

            for t in range(nr_batches_train):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size

                feed_dict = {inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             unl: trainx_unl[ran_from:ran_to]}

                sess.run(train_discriminator, feed_dict=feed_dict)

                loss_lab += ll
                loss_unl += lu
                train_err += te

                feed_dict = {unl: trainx_unl[ran_from:ran_to]}

                sess.run(train_generator, feed_dict=feed_dict)

                loss_lab /= nr_batches_train
                loss_unl /= nr_batches_train
                train_err /= nr_batches_train

            for t in range(nr_batches_test):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: testx[ran_from:ran_to],
                             lbl: testy[ran_from:ran_to],
                             is_training_pl: False}

                test_tp += sess.run(eval_correct, feed_dict=feed_dict)

            test_tp /= testx.shape[0]

            print("Epoch %d--Batch %d--Time = %ds | loss train = %.4f | train acc = %.4f | test acc = %.4f" %
                  (epoch, train_batch, time.time() - begin, train_loss, train_tp, test_tp))


if __name__ == '__main__':
    tf.app.run()
