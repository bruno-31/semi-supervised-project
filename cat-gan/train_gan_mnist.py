import os
import time
import numpy as np
import tensorflow as tf
import mnist_gan


flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size [128]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('log_dir', './log', 'log directory')
flags.DEFINE_integer('seed', 146, 'seed')
flags.DEFINE_integer('seed_data', 646, 'seed data')
flags.DEFINE_integer('labeled', 100, 'labeled image per class[100]')
flags.DEFINE_float('learning_rate', 0.003, 'learning_rate[0.003]')
flags.DEFINE_float('unl_weight', 0., 'unlabeled weight [1.]')
FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)

    # Random seed
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    rng_data = np.random.RandomState(FLAGS.seed_data)  # seed shuffling
    print('loading data')
    # load MNIST data
    data = np.load('mnist.npz')
    trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(np.float32)
    trainx_unl = trainx.copy()
    trainx_unl2 = trainx.copy()
    trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)
    testx = data['x_test'].astype(np.float32)
    testy = data['y_test'].astype(np.int32)
    nr_batches_test = int(testx.shape[0] / FLAGS.batch_size)

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

    print('labeled digits : ', len(tys))
    print(tys)

    '''construct graph'''
    print('constructing graph')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 28*28], name='labeled_data_input')
    unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 28*28], name='unlabeled_data_input')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')

    acc_train_pl = tf.placeholder(tf.float32,[],'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32,[],'acc_test_pl')

    with tf.variable_scope('generator_model'):
        gen_inp = mnist_gan.generator(FLAGS.batch_size, is_training_pl)

    with tf.variable_scope('discriminator_model') as dis_scope:
        logits_lab, _ = mnist_gan.discriminator(inp, is_training_pl)
        dis_scope.reuse_variables()
        logits_unl, layer_real = mnist_gan.discriminator(unl, is_training_pl)
        logits_fake, layer_fake = mnist_gan.discriminator(gen_inp, is_training_pl)

    with tf.name_scope('loss_functions'):
        # Taken from improved gan, T. Salimans
        l_unl = tf.reduce_logsumexp(logits_fake, axis=1)
        l_gen = tf.reduce_logsumexp(logits_unl, axis=1)

        with tf.name_scope('discriminator'):
            loss_lab = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbl, logits=logits_lab))
            loss_unl = - 0.5 * tf.reduce_mean(l_unl) \
                       + 0.5 * tf.reduce_mean(tf.nn.softplus(l_unl)) \
                       + 0.5 * tf.reduce_mean(tf.nn.softplus(l_gen))
            loss_dis = loss_unl+ FLAGS.unl_weight * loss_lab

            accuracy_dis = tf.reduce_mean(tf.cast(tf.less(l_unl,0),tf.float32))
            correct_pred = tf.equal(tf.cast(tf.argmax(logits_lab,1),tf.int32), tf.cast(lbl,tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope('generator'):
            m1 = tf.reduce_mean(layer_real, axis=0)
            m2 = tf.reduce_mean(layer_fake, axis=0)
            loss_gen = tf.reduce_mean(tf.square(m1-m2))
            # loss_gen = - 0.5 * tf.reduce_mean(l_unl) \
            #            + 0.5 * tf.reduce_mean(tf.nn.softplus(l_unl))
            fool_rate = tf.reduce_mean(tf.cast(tf.less(l_gen,0),tf.float32))

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()

        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=0.003, beta1=0.5)
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.003, beta1=0.5)

        train_dis_op = optimizer_dis.minimize(loss_dis, var_list=dvars, name='dis_optimizer')
        with tf.control_dependencies(update_ops_gen):
            train_gen_op = optimizer_gen.minimize(loss_gen, var_list=gvars, name='gen_optimizer')

    with tf.name_scope('dis_summary'):
        tf.summary.scalar('loss_labeled', loss_lab,['dis'])
        tf.summary.scalar('loss_unlabeled', loss_unl,['dis'])
        tf.summary.scalar('cls_accuracy', accuracy,['dis'])
        tf.summary.scalar('dis_accuracy', accuracy_dis,['dis'])

    with tf.name_scope('gen_summary'):
        tf.summary.scalar('loss_generator', loss_gen,['gen'])
        tf.summary.scalar('fool_rate', fool_rate, ['gen'])
        tf.summary.histogram('logits_generated',l_gen,['gen'])
        tf.summary.histogram('logits_unlabeled_data', l_unl, ['gen'])

    with tf.name_scope('epoch_summary'):
        tf.summary.image('input_digits', tf.reshape(inp, [-1,28,28,1]),10,['epoch'])
        tf.summary.image('gen_digits', tf.reshape(gen_inp, [-1, 28,28,1]), 10,['epoch'])
        tf.summary.scalar('accuracy_train', acc_train_pl,['epoch'])
        tf.summary.scalar('accuracy_test', acc_test_pl,['epoch'])

    sum_op_dis = tf.summary.merge_all('dis')
    sum_op_gen = tf.summary.merge_all('gen')
    sum_op_epoch = tf.summary.merge_all('epoch')

    '''//////perform training //////'''
    print('start training')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        for epoch in range(200):
            begin = time.time()

            # construct randomly permuted minibatches
            trainx = []
            trainy = []
            for t in range(int(np.ceil(trainx_unl.shape[0] / float(txs.shape[0])))): #same size lbl and unlb
                inds = rng.permutation(txs.shape[0])
                trainx.append(txs[inds])
                trainy.append(tys[inds])
            trainx = np.concatenate(trainx, axis=0)
            trainy = np.concatenate(trainy, axis=0)
            trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])] # shuffling unl dataset
            trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

            train_loss_lab, train_loss_unl, train_loss_gen, train_acc, test_acc, train_batch = [0, 0, 0, 0, 0, 0]
            # training
            for t in range(nr_batches_train):

                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size

                # train discriminator
                feed_dict = {inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             unl: trainx_unl[ran_from:ran_to],
                             is_training_pl: True}

                _, ll, lu, acc, sm = sess.run([train_dis_op,loss_lab,loss_unl,accuracy, sum_op_dis], feed_dict=feed_dict)

                train_loss_lab += ll
                train_loss_unl += lu
                train_acc += acc

                writer.add_summary(sm, train_batch)

                # train generator
                _, lg, sm = sess.run([train_gen_op, loss_gen, sum_op_gen], feed_dict={unl: trainx_unl[ran_from:ran_to],
                                                                                    is_training_pl: True})
                train_loss_gen += lg
                writer.add_summary(sm, train_batch)
                train_batch += 1

            train_loss_lab /= nr_batches_train
            train_loss_unl /= nr_batches_train
            train_acc /= nr_batches_train

            # Testing
            for t in range(nr_batches_test):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: testx[ran_from:ran_to],
                             lbl: testy[ran_from:ran_to],
                             is_training_pl: False}
                test_acc += sess.run(accuracy, feed_dict=feed_dict)

            test_acc /= nr_batches_test

            sum = sess.run(sum_op_epoch,feed_dict={inp: trainx[ran_from:ran_to],
                                                   is_training_pl:False,
                                                   acc_train_pl:train_acc,
                                                   acc_test_pl:test_acc})
            writer.add_summary(sum,epoch)

            print("Epoch %d--Time = %ds | loss gen = %.4f | loss lab = %.4f | loss unl = %.4f "
                  "| train acc = %.4f| test acc = %.4f"
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_lab, train_loss_unl, train_acc, test_acc))


if __name__ == '__main__':
    tf.app.run()
