import os
import time

import numpy as np
import tensorflow as tf

from data import cifar10_input
from cifar_gan_code import generator, discriminator
import sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size [100]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log/000', 'log directory')
flags.DEFINE_integer('seed', 1, 'seed ')
flags.DEFINE_integer('seed_data', 1, 'seed data')
flags.DEFINE_integer('labeled', 400, 'labeled data per class')
flags.DEFINE_float('learning_rate', 0.0003, 'learning_rate[0.003]')
flags.DEFINE_float('ma_decay', 0.9999 , 'moving average [0.9999]')

flags.DEFINE_boolean('nabla',False,'enable gradient regularization')
flags.DEFINE_float('nabla_weight', .001, 'gradient regularization weight [.01]')
flags.DEFINE_float('condfm', .0, 'cond feature matching weight [.01]')
flags.DEFINE_float('var', .05, 'cond feature matching weight [.01]')


flags.DEFINE_boolean('print_deter',True,'enable deterministic code printing [True')
flags.DEFINE_integer('freq_summary', 10, 'frequency scalar print tensorboard [500]')
flags.DEFINE_integer('freq_test', 1, 'frequency scalar print tensorboard [500]')
flags.DEFINE_integer('freq_save', 100, 'frequency saver epoch[100]')
flags.DEFINE_integer('freq_img', 5000, 'frequency image print tensorboard [500]')


FLAGS = flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.lower(), value))
print("")


def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss-learning cf Saliman et Al. 2016
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter

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

    print("Data:") # sanity check input data
    print('train shape %d | batch training %d \ntest shape %d  |  batch  testing %d' \
          % (trainx.shape[0], nr_batches_train, testx.shape[0], nr_batches_test))
    print('histogram train', np.histogram(trainy, bins=10)[0])
    print('histogram test ', np.histogram(testy, bins=10)[0])
    print("histogram labeled", np.histogram(tys, bins=10)[0])
    print("")

    '''construct graph'''
    print('constructing graph')
    unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='unlabeled_data_input_pl')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='labeled_data_input_pl')
    lbl_pl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input_pl')
    code_pl = tf.placeholder(tf.int32, [FLAGS.batch_size,], name='code_gen_img_pl')
    lr_pl = tf.placeholder(tf.float32,[],name='learning_rate_pl')
    #tensorboard pl
    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')
    acc_test_pl_ema = tf.placeholder(tf.float32, [], 'acc_test_pl')

    random_z = tf.random_uniform([FLAGS.batch_size, 100], name='random_z')
    code = tf.random_uniform([FLAGS.batch_size],minval=0,maxval=10,dtype=tf.int32)
    generator(random_z, code,is_training_pl, init=True)
    gen_inp = generator(random_z, code,is_training_pl, init=False,reuse=True)
    gen_inp_deterministic = generator(random_z, code_pl,is_training_pl, init=False,reuse=True)

    discriminator(unl, is_training_pl, init=True)
    logits_lab, layer_lbl = discriminator(inp, is_training_pl, init=False,reuse=True)
    logits_gen, layer_fake = discriminator(gen_inp, is_training_pl, init=False,reuse=True)
    logits_unl, layer_real = discriminator(unl, is_training_pl, init=False,reuse=True)

    with tf.name_scope('loss_functions'):
        #discriminator loss
        l_unl = tf.reduce_logsumexp(logits_unl, axis=1)
        l_gen = tf.reduce_logsumexp(logits_gen, axis=1)
        loss_lab = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbl_pl, logits=logits_lab))
        loss_unl = - 0.5 * tf.reduce_mean(l_unl) \
                   + 0.5 * tf.reduce_mean(tf.nn.softplus(l_unl)) \
                   + 0.5 * tf.reduce_mean(tf.nn.softplus(l_gen))
        loss_dis = loss_unl + loss_lab

        accuracy_dis = tf.reduce_mean(tf.cast(tf.less(l_unl, 0), tf.float32))
        correct_pred = tf.equal(tf.cast(tf.argmax(logits_lab, 1), tf.int32), tf.cast(lbl_pl, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # generator loss
        m1 = tf.reduce_mean(layer_real, axis=0)
        m2 = tf.reduce_mean(layer_fake, axis=0)
        loss_fm = tf.reduce_mean(tf.abs(m1 - m2))

        dpart1 = tf.dynamic_partition(layer_fake, partitions=code, num_partitions=10)
        dpart2 = tf.dynamic_partition(layer_lbl, partitions=lbl_pl, num_partitions=10)
        l=[]
        for i in range(10):
            m1 = tf.reduce_mean(dpart1[i], axis=0)
            m2 = tf.reduce_mean(dpart2[i], axis=0)
            val_nan = tf.zeros(dtype=tf.float32, shape=[])
            val_mean = tf.reduce_mean(tf.abs(m1 - m2))
            m22 = tf.where(tf.is_nan(tf.reduce_mean(m1)), val_nan, val_mean) # to avoid nan in sum
            l.append(m22)

        loss_condfm = tf.reduce_mean(l)

        h = []
        for i in range(10):
            _, sigma = tf.nn.moments(dpart1[i], axes=[0])
            val_nan = tf.zeros(dtype=tf.float32, shape=[])
            val_mean = tf.reduce_mean(m1)
            m1 = tf.where(tf.is_nan(val_mean), val_nan, val_mean)  # to avoid nan in sum
            h.append(m1)
        loss_var = tf.reduce_mean(h)

        loss_gen = loss_fm + FLAGS.condfm * loss_condfm + FLAGS.var * loss_var
        loss_dis += FLAGS.var * loss_var
        fool_rate = tf.reduce_mean(tf.cast(tf.less(l_gen, 0), tf.float32))

        # grad = tf.gradients(logits_gen, random_z)
        # dd = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))
        # ddx =tf.reduce_mean(tf.square(dd - 0))
        # if FLAGS.nabla:
        #     loss_dis+= FLAGS.nabla_weight * ddx
        #     loss_gen+= FLAGS.nabla_weight * ddx
        #     print('gradient reg enabled ...')

        k = []
        for j in range(10):
            grad = tf.gradients(logits_gen[j], random_z)
            k.append(grad)
        J = tf.stack(k)
        J = tf.squeeze(J)
        J = tf.transpose(J, perm=[1, 0, 2])  # jacobian
        j_n = tf.square(tf.norm(J, axis=[1, 2]))
        j_loss_gen = tf.reduce_mean(j_n)
        if FLAGS.nabla:
            loss_dis+= FLAGS.nabla_weight * j_loss_gen
            loss_gen+= FLAGS.nabla_weight * j_loss_gen
            print('gradient reg enabled ...')

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()

        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=lr_pl, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_pl, beta1=0.5, name='gen_optimizer')

        with tf.control_dependencies(update_ops_gen):
            train_gen_op = optimizer_gen.minimize(loss_gen, var_list=gvars)

        dis_op = optimizer_dis.minimize(loss_dis, var_list=dvars)

        ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ma_decay)
        maintain_averages_op = ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op)

        # averaged parameters for testing
        logits_ema, _ = discriminator(inp, is_training_pl, getter=get_getter(ema), reuse=True)
        correct_pred_ema = tf.equal(tf.cast(tf.argmax(logits_ema, 1), tf.int32), tf.cast(lbl_pl, tf.int32))
        accuracy_ema = tf.reduce_mean(tf.cast(correct_pred_ema, tf.float32))

    with tf.name_scope('summary'):
        with tf.name_scope('discriminator'):
            tf.summary.scalar('discriminator_accuracy', accuracy_dis, ['dis'])
            tf.summary.scalar('loss_discriminator', loss_dis, ['dis'])
            tf.summary.scalar('gradient_loss', ddx, ['dis'])

        with tf.name_scope('generator'):
            tf.summary.scalar('loss_generator', loss_gen, ['gen'])
            tf.summary.scalar('fool_rate', fool_rate, ['gen'])
            tf.summary.scalar('loss_cond_fm', loss_condfm,['gen'])
            tf.summary.scalar('loss_fm',loss_fm,['gen'])


        with tf.name_scope('images'):
            if not FLAGS.print_deter:
                tf.summary.image('gen_img', gen_inp, 20, ['image'])
            else:
                tf.summary.image('gen_img', gen_inp_deterministic, 20, ['image'])

        with tf.name_scope('epoch'):
            tf.summary.scalar('accuracy_train', acc_train_pl, ['epoch'])
            tf.summary.scalar('accuracy_test_moving_average', acc_test_pl_ema, ['epoch'])
            tf.summary.scalar('accuracy_test_raw', acc_test_pl, ['epoch'])
            tf.summary.scalar('learning_rate', lr_pl,['epoch'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_im = tf.summary.merge_all('image')
        sum_op_epoch = tf.summary.merge_all('epoch')


    init_gen = [var.initializer for var in gvars][:-3]

    saver = tf.train.Saver()

    '''//////perform training //////'''
    print('start training')
    with tf.Session() as sess:
        sess.run(init_gen)
        init = tf.global_variables_initializer()
        # Data-Dependent Initialization of Parameters as discussed in DP Kingma and Salimans Paper
        sess.run(init, feed_dict={inp:trainx_unl[:FLAGS.batch_size],unl: trainx_unl[:FLAGS.batch_size],
                                  is_training_pl: True})
        print('initialization done\n')
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        train_batch = 0

        for epoch in range(1200):
            begin = time.time()
            train_loss_lab, train_loss_unl, train_loss_gen, train_acc, test_acc, test_acc_ma = [0,0, 0, 0, 0, 0]

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
                feed_dict = {unl: trainx_unl[ran_from:ran_to],
                             is_training_pl: True,
                             inp: trainx[ran_from:ran_to],
                             lbl_pl: trainy[ran_from:ran_to],
                             lr_pl: lr}
                _, acc, lu, lb, sm = sess.run([train_dis_op, accuracy, loss_lab, loss_unl, sum_op_dis],
                                     feed_dict=feed_dict)
                train_loss_unl += lu
                train_loss_lab += lb
                train_acc += acc
                if train_batch % FLAGS.freq_summary == 0:
                    writer.add_summary(sm, train_batch)

                # train generator
                _, lg, sm = sess.run([train_gen_op, loss_gen, sum_op_gen], feed_dict={unl: trainx_unl2[ran_from:ran_to],
                                                                                      is_training_pl: True,
                                                                                      inp: trainx[ran_from:ran_to],
                                                                                      lbl_pl: trainy[ran_from:ran_to],
                                                                                      lr_pl:lr,
                                                                                      code_pl: np.zeros(FLAGS.batch_size)})
                train_loss_gen += lg
                if train_batch % FLAGS.freq_summary == 0:
                    writer.add_summary(sm, train_batch)

                 # image summaries
                if ((train_batch % FLAGS.freq_img == 0) & (train_batch != 0)) | (epoch == 1199):
                    sm = sess.run(sum_op_im,{is_training_pl: False,code_pl:np.random.randint(0,10,FLAGS.batch_size)})
                    writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_lab /= nr_batches_train
            train_loss_unl /= nr_batches_train
            train_loss_gen /= nr_batches_train
            train_acc /= nr_batches_train

            # Testing
            if (epoch % FLAGS.freq_test == 0) | (epoch == 1199):
                for t in range(nr_batches_test):
                    ran_from = t * FLAGS.batch_size
                    ran_to = (t + 1) * FLAGS.batch_size
                    feed_dict = {inp: testx[ran_from:ran_to],
                                 lbl_pl: testy[ran_from:ran_to],
                                 is_training_pl: False}
                    acc,acc_ema = sess.run([accuracy, accuracy_ema], feed_dict=feed_dict)
                    test_acc+=acc
                    test_acc_ma += acc_ema
                test_acc /= nr_batches_test
                test_acc_ma /= nr_batches_test
                # epoch summary
                sum = sess.run(sum_op_epoch, feed_dict={acc_train_pl: train_acc,
                                                        acc_test_pl: test_acc,
                                                        acc_test_pl_ema:test_acc_ma,
                                                        lr_pl:lr})
                writer.add_summary(sum, epoch)

            if epoch % FLAGS.freq_save == 0 & epoch != 0:
                save_path = saver.save(sess, os.path.join(FLAGS.logdir, 'model.ckpt'))
                print("Model saved in file: %s" % (save_path))

            print("Epoch %d--Time = %ds Lr = %0.2e | loss gen = %.4f | loss lab = %.4f | loss unl = %.4f "
                  "| train acc = %.4f| test acc = %.4f | test acc ema = %0.4f"
                  % (epoch, time.time() -begin,lr, train_loss_gen, train_loss_lab, train_loss_unl, train_acc, test_acc,test_acc_ma))


if __name__ == '__main__':
    tf.app.run()
