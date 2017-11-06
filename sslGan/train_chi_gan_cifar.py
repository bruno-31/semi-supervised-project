import os
import time
import numpy as np
import tensorflow as tf
import cifar10_input
import cifar_chi_gan
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size [128]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log/x000', 'log directory')
flags.DEFINE_integer('seed', 4, 'seed ')
flags.DEFINE_integer('seed_data', 4, 'seed data')
flags.DEFINE_integer('labeled', 400, 'labeled data per class')
flags.DEFINE_float('learning_rate', 0.0003, 'learning_rate[0.003]')
flags.DEFINE_integer('freq_print', 500, 'frequency image print tensorboard [20]')
flags.DEFINE_float('unl_weight', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('lbl_weight', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('gen_weight_fmatch', 1.0, 'unlabeled weight [1.]')
flags.DEFINE_float('gen_weight_bin', 0.0 , 'unlabeled weight [1.]')
flags.DEFINE_float('ma_decay', 0.9999 , 'moving average testing, 0 to disable  [0.9999]')
flags.DEFINE_integer('freq_save', 50, 'frequency saver epoch')


FLAGS = flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.lower(), value))
print("")

def main(_):
    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)

    filename = FLAGS.logdir + '/param.txt'
    if not os.path.isfile(filename):
        os.mknod(filename)
    file = open(filename, "a")
    file.write("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        file.write("{}={}".format(attr.lower(), value))
        file.write("\n")
    file.close()

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

    # print('labeled images : ', len(tys))

    '''construct graph'''
    print('constructing graph')
    unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='unlabeled_data_input_pl')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='labeled_data_input_pl')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input_pl')
    lr_pl = tf.placeholder(tf.float32,[],name='learning_rate_pl')
    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')

    gen = cifar_chi_gan.generator
    dis = cifar_chi_gan.discriminator

    random_z = tf.random_uniform([FLAGS.batch_size, 100], name='random_z')
    with tf.variable_scope('generator_model') as scope:
        gen(random_z, is_training_pl, init=True)
        scope.reuse_variables()
        gen_inp = gen(random_z, is_training_pl, init=False)

    with tf.variable_scope('discriminator_model') as scope:
        dis(unl, is_training_pl, init=True)
        scope.reuse_variables()
        l_lab_cls, l_lab_dis, _ = dis(inp, is_training_pl, init=False)
        l_gen_cls, l_gen_dis, layer_fake = dis(gen_inp, is_training_pl, init=False)
        l_unl_cls, l_unl_dis, layer_real = dis(unl, is_training_pl, init=False)

    with tf.variable_scope("model_test") as test_scope:
        dis(inp, is_training_pl, True)
        test_scope.reuse_variables()
        l_test, _,_ = dis(inp, is_training_pl, False)

    with tf.name_scope('loss_functions'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits
        sigmoid = tf.nn.sigmoid_cross_entropy_with_logits

        loss_cls = tf.reduce_mean(xentropy(logits=l_lab_cls, labels=lbl))
        loss_dis_unl = tf.reduce_mean(sigmoid(logits=l_unl_dis, labels=tf.ones_like(l_unl_dis)))
        loss_dis_gen = tf.reduce_mean(sigmoid(logits=l_gen_dis, labels=tf.zeros_like(l_gen_dis)))
        loss_dis = loss_dis_unl + loss_dis_gen

        accuracy_dis_gen = tf.reduce_mean(tf.cast(tf.less(l_gen_dis, 0), tf.float32))
        accuracy_dis_unl = tf.reduce_mean(tf.cast(tf.greater(l_unl_dis, 0), tf.float32))

        correct_pred = tf.equal(tf.cast(tf.argmax(l_lab_cls, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        correct_pred_test = tf.equal(tf.cast(tf.argmax(l_test, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))

        # GENERATOR
        m1 = tf.reduce_mean(layer_real, axis=0)
        m2 = tf.reduce_mean(layer_fake, axis=0)
        loss_gen_fmatch = tf.reduce_mean(tf.square(m1 - m2))
        loss_gen_bin = tf.reduce_mean(sigmoid(logits=l_gen_dis, labels=tf.ones_like(l_gen_dis)))
        fool_rate = tf.reduce_mean(tf.cast(tf.greater(l_gen_dis, 0), tf.float32))
        loss_gen = FLAGS.gen_weight_bin * loss_gen_bin + FLAGS.gen_weight_fmatch * loss_gen_fmatch

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()

        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        cvars = dvars[9:]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        testvars = [var for var in tvars if 'model_test' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=lr_pl, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_pl, beta1=0.5, name='gen_optimizer')
        optimizer_cls = tf.train.AdamOptimizer(learning_rate=lr_pl/3, beta1=0.5, name='cls_optimizer')

        with tf.control_dependencies(update_ops_gen):
            train_gen_op = optimizer_gen.minimize(loss_gen, var_list=gvars)

        train_dis_op = optimizer_dis.minimize(loss_dis, var_list=dvars)

        cls_op = optimizer_cls.minimize(loss_cls, var_list=cvars)

        ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ma_decay)
        maintain_averages_op = ema.apply(dvars)

        with tf.control_dependencies(update_ops_dis):
            with tf.control_dependencies([cls_op]):
                train_cls_op = tf.group(maintain_averages_op)

        copy_graph = [tf.assign(x, ema.average(y)) for x, y in zip(testvars, dvars)]

    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('discriminator_accuracy_unl', accuracy_dis_unl, ['dis'])
            tf.summary.scalar('discriminator_accuracy_gen', accuracy_dis_gen, ['dis'])
            tf.summary.scalar('loss_discriminator', loss_dis, ['dis'])

        with tf.name_scope('cls_summary'):
            tf.summary.scalar('loss_cls', loss_cls, ['cls'])
            tf.summary.scalar('classifier_accuracy', accuracy, ['cls'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_gen, ['gen'])
            tf.summary.scalar('fool_rate', fool_rate, ['gen'])

        with tf.name_scope('image_summary'):
            tf.summary.image('gen_digits', gen_inp, 20, ['image'])
            tf.summary.image('input_images', unl, 1, ['image'])

        with tf.name_scope('epoch_summary'):
            tf.summary.scalar('accuracy_train', acc_train_pl, ['epoch'])
            tf.summary.scalar('accuracy_test', acc_test_pl, ['epoch'])
            tf.summary.scalar('learning_rate', lr_pl,['epoch'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_cls = tf.summary.merge_all('cls')
        sum_op_im = tf.summary.merge_all('image')
        sum_op_epoch = tf.summary.merge_all('epoch')

    # [print(var.name) for var in gvars]
    init_gen = [var.initializer for var in gvars][:-3]

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
    # [print(op.name) for op in update_ops_gen]

    saver = tf.train.Saver()

    '''//////perform training //////'''
    print('start training')
    with tf.Session() as sess:
        feed_dict_init = {inp: trainx_unl[:FLAGS.batch_size], unl: trainx_unl[:FLAGS.batch_size], is_training_pl: True}
        sess.run(init_gen, feed_dict_init) # pre init of non data dependent layers
        init = tf.global_variables_initializer()
        sess.run(init, feed_dict_init) # Data-Dependent Initialization of Parameters as discussed in DP Kingma and Salimans Paper
        print('initialization done')
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        train_batch = 0

        # tvars = tf.trainable_variables()
        # print(" ... sanity check trainable variables ... ")
        # [print(v.name) for v in tvars]

        glob_begin = time.time()
        max_test_acc = 0

        for epoch in range(1200):
            begin = time.time()
            train_loss_lab, train_loss_unl, train_loss_gen, train_acc, test_acc = [0, 0, 0, 0, 0]

            lr = FLAGS.learning_rate * min(3-epoch/400,1)

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

            # training
            for t in range(nr_batches_train):

                display_progression_epoch(t, nr_batches_train)
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size

                # train discriminator
                feed_dict = {unl: trainx_unl[ran_from:ran_to], is_training_pl: True, lr_pl: lr}
                _, lu, sm = sess.run([train_dis_op, loss_dis, sum_op_dis],
                                     feed_dict=feed_dict)
                train_loss_unl += lu
                writer.add_summary(sm, train_batch)

                # train classifier
                feed_dict = {is_training_pl: True,
                             inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             lr_pl: lr}
                _, acc, lb, sm = sess.run([train_cls_op, accuracy, loss_cls, sum_op_cls],
                                              feed_dict=feed_dict)
                train_loss_lab += lb
                train_acc += acc
                writer.add_summary(sm, train_batch)

                # train generator
                _, lg, sm = sess.run([train_gen_op, loss_gen, sum_op_gen], feed_dict={unl: trainx_unl2[ran_from:ran_to],
                                                                                      is_training_pl: True,
                                                                                      lr_pl:lr})
                train_loss_gen += lg
                train_batch += 1
                writer.add_summary(sm, train_batch)

                if t % FLAGS.freq_print == 0:
                    x = np.random.randint(0, 4000)
                    sm = sess.run(sum_op_im,
                                  feed_dict={is_training_pl: True, unl: trainx_unl[x:x + FLAGS.batch_size]})
                    writer.add_summary(sm, train_batch)

            train_loss_lab /= nr_batches_train
            train_loss_unl /= nr_batches_train
            train_loss_gen /= nr_batches_train
            train_acc /= nr_batches_train

            # Testing
            sess.run(copy_graph)
            for t in range(nr_batches_test):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: testx[ran_from:ran_to],
                             lbl: testy[ran_from:ran_to],
                             is_training_pl: False}
                test_acc += sess.run(accuracy_test, feed_dict=feed_dict)

            test_acc /= nr_batches_test
            max_test_acc = max(test_acc, max_test_acc)

            sum = sess.run(sum_op_epoch, feed_dict={acc_train_pl: train_acc,
                                                    acc_test_pl: test_acc,
                                                    lr_pl:lr})
            writer.add_summary(sum, epoch)

            if epoch % FLAGS.freq_save == 0:
                save_path = saver.save(sess, os.path.join(FLAGS.logdir, 'model.ckpt'))
                print("Model saved in file: %s" % (save_path))

            print("Epoch %d--Time = %ds Lr = %0.2e | loss gen = %.4f | loss lab = %.4f | loss unl = %.4f "
                  "| train acc = %.4f| test acc = %.4f"
                  % (epoch, time.time() -begin,lr, train_loss_gen, train_loss_lab, train_loss_unl, train_acc, test_acc))

        print("Training Done in %ds, max test acc = %0.4f"%(time.time()-glob_begin, max_test_acc))

        file = open(filename, "a")
        file.write("\nmax accuray test_set : %0.4f\n" % (max_test_acc))
        file.close()

if __name__ == '__main__':
    tf.app.run()
