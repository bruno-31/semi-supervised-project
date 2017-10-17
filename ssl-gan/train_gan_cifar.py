import os
import time
import sys
import numpy as np
import tensorflow as tf
import cifar10_input
import cifar_gan


flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size [128]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log', 'log directory')
flags.DEFINE_integer('seed', 0, 'seed ')
flags.DEFINE_integer('seed_data', 0, 'seed data')
flags.DEFINE_integer('labeled', 400, 'labeled data per class')
flags.DEFINE_float('learning_rate', 0.0003, 'learning_rate[0.003]')
flags.DEFINE_integer('freq_print', 50, 'frequency image print tensorboard [20]')
flags.DEFINE_float('unl_weight', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('lbl_weight', 0.0, 'unlabeled weight [1.]')
FLAGS = flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.lower(), value))
print("")


def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def main(_):
    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)

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

    print('labeled images : ', len(tys))
    print(tys)

    '''construct graph'''
    print('constructing graph')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='labeled_data_input_pl')
    unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='unlabeled_data_input_pl')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input_pl')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    lr_pl = tf.placeholder(tf.float32, [], name='learning_rate_pl')

    gen = cifar_gan.generator
    dis = cifar_gan.discriminator

    with tf.variable_scope('generator_model') as scope:
        gen_init = gen(FLAGS.batch_size, is_training=is_training_pl, init=True)
        [print(v.name) for v in tf.trainable_variables()]
        print("")
        scope.reuse_variables()
        gen_inp = gen(FLAGS.batch_size, is_training_pl, init=False)
        [print(v.name) for v in tf.trainable_variables()]
        print("")
    with tf.variable_scope('discriminator_model') as scope:
        dis(inp, is_training_pl, init=True)
        scope.reuse_variables()
        logits_lab, _ = dis(inp, is_training_pl)
        logits_unl, layer_real = dis(unl, is_training_pl)
        logits_gen, layer_fake = dis(gen_inp, is_training_pl)

    with tf.variable_scope("model_test") as test_scope:
        _, _ = dis(inp, is_training_pl, True)
        test_scope.reuse_variables()
        logits_test, _ = dis(inp, is_training_pl, False)

    with tf.name_scope('loss_functions'):
        with tf.name_scope('generator_loss'):
            loss_gen = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_gen, labels=tf.ones_like(logits_gen)))
        with tf.name_scope('discriminator_loss'):
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_unl, labels=tf.ones_like(logits_unl)))
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_gen, labels=tf.zeros_like(logits_gen)))
            loss_dis = d_loss_fake + d_loss_real
        fool_rate = tf.reduce_mean(tf.cast(tf.greater(logits_gen,0),tf.float32))
        accuracy_dis_unl = tf.reduce_mean(tf.cast(tf.greater(logits_unl,0),tf.float32))
        accuracy_dis_gen = tf.reduce_mean(tf.cast(tf.less(logits_gen,0),tf.float32))
        accuracy_dis= 0.5 * accuracy_dis_gen + 0.5 *accuracy_dis_unl


    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()

        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        testvars = [var for var in tvars if 'model_test' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=lr_pl, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_pl, beta1=0.5, name='gen_optimizer')

        with tf.control_dependencies(update_ops_gen):
            train_gen_op = optimizer_gen.minimize(loss_gen, var_list=gvars)

        training_op = optimizer_dis.minimize(loss_dis, var_list=dvars)
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        maintain_averages_op = ema.apply(dvars)

        with tf.control_dependencies(update_ops_dis):
            with tf.control_dependencies([training_op]):
                train_dis_op = tf.group(maintain_averages_op)

        # copy_graph = [tf.assign(x,ema.average(y)) for x,y in zip(testvars,dvars)]
        copy_graph = [tf.assign(x, y) for x, y in zip(testvars, dvars)]

    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_unlabeled', loss_dis, ['dis'])
            tf.summary.scalar('discriminator_accuracy', accuracy_dis, ['dis'])
            tf.summary.scalar('discriminator_accuracy_fake_samples', accuracy_dis_gen, ['dis'])
            tf.summary.scalar('discriminator_accuracy_unl_samples', accuracy_dis_unl, ['dis'])
            tf.summary.scalar('loss_discriminator', loss_dis, ['dis'])
            tf.summary.scalar('learning_rate', lr_pl, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_gen, ['gen'])
            tf.summary.scalar('fool_rate', fool_rate, ['gen'])

        with tf.name_scope('image_summary'):
            tf.summary.image('gen_digits', gen_inp, 20, ['image'])
            tf.summary.image('input_images', inp, 10, ['image'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_im = tf.summary.merge_all('image')

    init_gen= [var.initializer for var in gvars][:-3]
    [print(x.name) for x in init_gen]
    '''//////perform training //////'''
    print('start training')
    with tf.Session() as sess:
        sess.run(init_gen)
        init = tf.global_variables_initializer()
        # Data-Dependent Initialization of Parameters as discussed in DP Kingma and Salimans Paper
        sess.run(init, feed_dict={inp: trainx_unl[0:FLAGS.batch_size], is_training_pl: True})
        print('initialization done')
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        train_batch = 0

        for epoch in range(200):
            begin = time.time()
            train_loss_dis, train_loss_unl, train_loss_gen, train_acc, test_acc = [0, 0, 0, 0, 0]

            # construct randomly permuted minibatches
            trainx = []
            trainy = []
            for t in range(int(np.ceil(trainx_unl.shape[0] / float(txs.shape[0])))):  # same size lbl and unlb
                inds = rng.permutation(txs.shape[0])
                trainx.append(txs[inds])
                trainy.append(tys[inds])
            trainx = np.concatenate(trainx, axis=0)
            trainy = np.concatenate(trainy, axis=0)
            trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]  # shuffling unl dataset
            trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

            lr = min(1., 3 - epoch / 400) * FLAGS.learning_rate # decayed learning rate feed to lr_pl
            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size

                # train discriminator
                feed_dict = {inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             unl: trainx_unl[ran_from:ran_to],
                             is_training_pl: False,
                             lr_pl: lr}
                _, ld, sm = sess.run([train_dis_op, loss_dis, sum_op_dis],
                                              feed_dict=feed_dict)
                train_loss_dis += ld
                writer.add_summary(sm, train_batch)

                # train generator
                _, lg, sm = sess.run([train_gen_op, loss_gen, sum_op_gen], feed_dict={unl: trainx_unl2[ran_from:ran_to],
                                                                                      is_training_pl: True, lr_pl: lr})
                train_loss_gen += lg
                train_batch += 1
                writer.add_summary(sm, train_batch)

                if t % FLAGS.freq_print == 0:
                    x = np.random.randint(0, 4000)
                    sm = sess.run(sum_op_im,
                                  feed_dict={is_training_pl: True, inp: trainx_unl[x:x + FLAGS.batch_size], lr_pl: lr})
                    writer.add_summary(sm, train_batch)

            train_loss_dis /= nr_batches_train
            train_loss_gen /= nr_batches_train


            print("Epoch %d--time = %ds--lr = %.1f | loss gen = %.4f | loss dis = %.4f |"
                  % (epoch, time.time() - begin,lr, train_loss_gen, train_loss_dis))


if __name__ == '__main__':
    tf.app.run()
