import os
import sys
import time

import numpy as np
import tensorflow as tf

from cond_gan_mnist import generator, discriminator

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 100, "batch size [100]")
flags.DEFINE_string('logdir', './log_bruno/000', 'log directory')
flags.DEFINE_integer('seed', 146, 'seed')
flags.DEFINE_integer('seed_data', 646, 'seed data')
flags.DEFINE_integer('seed_tf', 646, 'tf random seed')
flags.DEFINE_integer('freq_print', 100, 'image summary frequency [100]')
flags.DEFINE_boolean('enable_print', False, 'enable generated digits printing [F]')
flags.DEFINE_integer('labeled', 10, 'labeled image per class[10]')
flags.DEFINE_float('learning_rate_d', 0.003, 'learning_rate dis[0.003]')
flags.DEFINE_float('learning_rate_g', 0.003, 'learning_rate gen[0.003]')
flags.DEFINE_float('learning_rate_c', 0.003, 'learning_rate gen[0.003]')
# weights loss
flags.DEFINE_float('gen_cat_weight', 0.0, 'categorical generator weight [1.]')
flags.DEFINE_float('gen_bin_weight', 1.0, 'categorical generator weight [1.]')
flags.DEFINE_float('f_match_weight', 0.0, 'categorical generator weight [0.]')
flags.DEFINE_float('cat_weight_dis', 0.0, 'categorical generator weight [1.]')
flags.DEFINE_float('lambda_cls', 0.1, 'categorical generator weight [1.]')

FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.lower(), value))
print("")


def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def cat_z():
    x = np.random.randint(0, 10, 100)
    array = np.zeros([100, 10])
    array[range(100), x] = 1
    return array


def balance_z():
    codes = []
    for i in range(10):
        codes.append(np.ones(10) * i)
    codes = np.concatenate(codes)

    def to_one_hot(arr,depth=10):
        array = np.zeros([arr.shape[0],depth])
        array[range(arr.shape[0]),arr] = 1
        return array
    return to_one_hot(codes.astype(int),depth=10)

def std_loss(layers):
    layer_split = tf.split(layers, 10, axis=0)
    def std_tensor(x):
        _, std = tf.nn.moments(x, axes=[0])
        return std

    y = list(map(std_tensor, layer_split))
    return tf.nn.l2_loss(y)



def main(_):
    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)
    # Random seed
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    rng_data = np.random.RandomState(FLAGS.seed_data)  # seed shuffling
    tf.set_random_seed(FLAGS.seed_tf)
    print('loading data  ... ')
    # load MNIST data
    data = np.load('./data/mnist.npz')
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

    '''construct graph'''
    print('construct graph')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28], name='labeled_data_input_pl')
    unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28], name='unlabeled_data_input_pl')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input_pl')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    lbl_fake = tf.placeholder(tf.float32, [FLAGS.batch_size, 10], name='label_fake_pl')

    gen = generator
    dis = discriminator

    with tf.variable_scope('generator_model'):
        gen_inp = gen(FLAGS.batch_size, lbl_fake, is_training=is_training_pl)

    with tf.variable_scope('discriminator_model') as scope:
        dis(inp, is_training_pl, init=True)  # Data driven initialization
        scope.reuse_variables()
        logits_dis_lab, logits_cls_lab, layer_lbl = dis(inp, is_training_pl)
        logits_dis_unl, _, layer_real = dis(unl, is_training_pl)
        logits_dis_gen, logits_cls_gen, layer_fake = dis(gen_inp, is_training_pl)

    with tf.variable_scope("model_test") as test_scope:
        dis(inp, is_training_pl, True)
        test_scope.reuse_variables()
        _, logits_test, _ = dis(inp, is_training_pl, False)

    with tf.name_scope('loss_functions'):
        sigmoid = tf.nn.sigmoid_cross_entropy_with_logits

        loss_dis_unl = tf.reduce_mean(sigmoid(logits=logits_dis_unl, labels=tf.ones([FLAGS.batch_size, 1])))
        loss_dis_gen = tf.reduce_mean(sigmoid(logits=logits_dis_gen, labels=tf.zeros([FLAGS.batch_size, 1])))
        loss_dis = loss_dis_unl + loss_dis_gen

        m1 = tf.reduce_mean(layer_real, axis=0)
        m2 = tf.reduce_mean(layer_fake, axis=0)
        loss_features_matching = tf.reduce_mean(tf.square(m1 - m2))
        loss_gen_bin = tf.reduce_mean(sigmoid(logits=logits_dis_gen, labels=tf.ones([FLAGS.batch_size, 1])))
        loss_gen = 0 * loss_features_matching + 1* loss_gen_bin

        loss_q = 0.1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_cls_gen, labels=lbl_fake))
        loss_c = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits_cls_lab, labels=tf.one_hot(lbl, 10)))

        # variance leayer conditioned to each label
        loss_std = .001 * std_loss(layer_fake)

        # cond_fm
        l_m1  = tf.reduce_mean(layer_lbl)
        l_m2 = tf.reduce_mean(la)
        loss_gen += loss_q
        loss_dis += loss_q
        loss_c += loss_q

        # loss_gen += loss_std
        # loss_dis += loss_std
        # loss_c += loss_std

        accuracy_dis_unl = tf.reduce_mean(tf.cast(tf.greater(logits_dis_unl, 0), tf.float32))
        accuracy_dis_gen = tf.reduce_mean(tf.cast(tf.less(logits_dis_gen, 0), tf.float32))
        accuracy_dis = 0.5 * accuracy_dis_unl + 0.5 * accuracy_dis_gen
        fool_rate = tf.reduce_mean(tf.cast(tf.greater(logits_dis_gen, 0), tf.float32))

        correct_pred = tf.equal(tf.cast(tf.argmax(logits_cls_lab, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy_cls = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        correct_pred_test = tf.equal(tf.cast(tf.argmax(logits_test, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy_cls_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model/D_weights' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        qvars = [var for var in tvars if 'discriminator_model/Q_weights' in var.name]
        disvars = [var for var in tvars if 'discriminator_model/shared_weights' in var.name]
        copyvars = [var for var in tvars if 'discriminator_model' in var.name]

        [print(var.name) for var in dvars + disvars + qvars]
        print('')

        testvars = [var for var in tvars if 'model_test' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]

        optimizer_cls = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_c, beta1=0.5, name='cls_optimizer')
        optimizer_dis = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_d, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_g, beta1=0.5, name='gen_optimizer')

        train_dis_op = optimizer_dis.minimize(loss_dis, var_list=dvars + disvars)
        train_cls_op = optimizer_cls.minimize(loss_c, var_list=disvars + qvars + gvars)
        with tf.control_dependencies(update_ops_gen):
            train_gen_op = optimizer_gen.minimize(loss_gen, var_list=gvars)

        ema = tf.train.ExponentialMovingAverage(decay=0.9999)  # moving average for testing
        maintain_averages_op = ema.apply(copyvars)
        with tf.control_dependencies([train_cls_op]):
            train_cls_op_ema = tf.group(maintain_averages_op)

        copy_graph = [tf.assign(x, ema.average(y)) for x, y in zip(testvars, copyvars)]

    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_dis', loss_dis, ['dis'])
            tf.summary.scalar('loss_dis_unl', loss_dis_unl, ['dis'])
            tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])
            tf.summary.scalar('discriminator_accuracy', accuracy_dis, ['dis'])
            tf.summary.scalar('discriminator_accuracy_gen_samples', accuracy_dis_gen, ['dis'])
            tf.summary.scalar('discriminator_accuracy_unl_samples', accuracy_dis_unl, ['dis'])

        with tf.name_scope('cls_summary'):
            tf.summary.scalar('loss_cls_fake', loss_q, ['cls'])
            tf.summary.scalar('classifier_accuracy', accuracy_cls, ['cls'])
            tf.summary.scalar('clustering_std', std_loss(layer_fake), ['cls'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_gen, ['gen'])
            tf.summary.scalar('loss_gen_bin', loss_gen_bin, ['gen'])
            tf.summary.scalar('loss_feature_matching', loss_features_matching, ['gen'])
            tf.summary.scalar('fool_rate', fool_rate, ['gen'])

        with tf.name_scope('image_summary'):
            tf.summary.image('gen_digits_rnd_class', tf.reshape(gen_inp, [-1, 28, 28, 1]), 8, ['rnd_image'])
            tf.summary.image('gen_digits_deter_class', tf.reshape(gen_inp, [-1, 28, 28, 1]), 30, ['deter_image'])

        sum_op_cls = tf.summary.merge_all('cls')
        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_im_rnd = tf.summary.merge_all('rnd_image')
        sum_op_im_deter = tf.summary.merge_all('deter_image')

    '''//////perform training //////'''
    print('start training')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        # Data-Dependent Initialization of Parameters as discussed in DP Kingma and Salimans Paper
        sess.run(init, feed_dict={inp: trainx_unl[0:FLAGS.batch_size], is_training_pl: True})
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        batch = 0
        print('data driven initialization done\n')

        for epoch in range(200):
            begin = time.time()

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

            train_loss_cls, train_loss_dis, train_loss_gen, train_acc, test_acc = [0, 0, 0, 0, 0]
            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                # train discriminator
                feed_dict = {unl: trainx_unl[ran_from:ran_to],
                             is_training_pl: True,
                             lbl_fake: cat_z()}
                _, ld, sm = sess.run([train_dis_op, loss_dis, sum_op_dis], feed_dict=feed_dict)
                train_loss_dis += ld
                writer.add_summary(sm, batch)

                # train generator
                feed_dict = {unl: trainx_unl2[ran_from:ran_to],
                             is_training_pl: True,
                             lbl_fake: cat_z()}
                _, lg, sm = sess.run([train_gen_op, loss_gen, sum_op_gen], feed_dict=feed_dict)
                train_loss_gen += lg
                writer.add_summary(sm, batch)

                # train classifier
                feed_dict = {inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             is_training_pl: True,
                             lbl_fake: cat_z()}
                _, lc, acc, sm = sess.run([train_cls_op_ema, loss_c, accuracy_cls, sum_op_cls], feed_dict=feed_dict)
                train_acc += acc
                train_loss_cls += lc
                writer.add_summary(sm, batch)

                # print
                if batch % FLAGS.freq_print == 0 & FLAGS.enable_print:
                    ran_from = np.random.randint(0, trainx_unl.shape[0] - FLAGS.batch_size)
                    ran_to = ran_from + FLAGS.batch_size
                    code = np.zeros((100, 10))
                    code[:, (np.random.randint(0, 10))] = np.ones(100)
                    # print('code',code[0])
                    sm = sess.run(sum_op_im_deter, feed_dict={unl: trainx_unl[ran_from:ran_to],
                                                              is_training_pl: False,
                                                              lbl_fake: code})
                    writer.add_summary(sm, batch)

                    sm = sess.run(sum_op_im_rnd, feed_dict={unl: trainx_unl[ran_from:ran_to],
                                                            is_training_pl: False,
                                                            lbl_fake: cat_z()})
                    writer.add_summary(sm, batch)
                batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_cls /= nr_batches_train
            train_loss_dis /= nr_batches_train
            train_acc /= nr_batches_train

            # Testing classifier
            sess.run(copy_graph)
            for t in range(nr_batches_test):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: testx[ran_from:ran_to],
                             lbl: testy[ran_from:ran_to],
                             is_training_pl: False}
                test_acc += sess.run(accuracy_cls_test, feed_dict=feed_dict)
            test_acc /= nr_batches_test

            print("| Epoch %d | time = %ds | loss gen = %.4f | loss cls = %.4f | loss dis = %.4f "
                  "| train acc = %.4f| test acc = %.4f |"
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_cls, train_loss_dis, train_acc, test_acc))


if __name__ == '__main__':
    tf.app.run()
