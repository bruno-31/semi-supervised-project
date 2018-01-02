import os
import time
import numpy as np
import tensorflow as tf
from mnist_gan_new import generator, discriminator
import sys
from tensorflow.python.client import timeline


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 100, "batch size [100]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log_mnist/000', 'log directory')
flags.DEFINE_integer('labeled', 10, 'labeled image per class[100]')
flags.DEFINE_float('learning_rate_d', 0.003, 'learning_rate dis[0.003]')
flags.DEFINE_float('learning_rate_g', 0.003, 'learning_rate gen[0.003]')
flags.DEFINE_float('ma_decay', 0.9999 , 'moving average [0.9999]')

flags.DEFINE_float('step_print', 1200 , 'scale perturbation')
flags.DEFINE_float('freq_print', 1200, 'scale perturbation')

flags.DEFINE_integer('seed', 111, 'seed')
flags.DEFINE_integer('seed_data', 111, 'seed data')
flags.DEFINE_integer('seed_tf', 111, 'tf random seed')

flags.DEFINE_float('scale', 0.15 , 'scale perturbation')
flags.DEFINE_boolean('nabla', False , 'enable nabla reg')
flags.DEFINE_float('nabla_w', 0.1 , 'weight nabla reg')
flags.DEFINE_boolean('soft', True , 'enable nabla reg softmaxed')



FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.lower(), value))
print("")

def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush

def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm


def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp


def entropy_y_x(logit):
    p = tf.nn.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))

def get_normalized_vector(d):
    d /= (1e-12 + tf.reduce_max(tf.abs(d), range(1, len(d.get_shape())), keep_dims=True))
    d /= tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), range(1, len(d.get_shape())), keep_dims=True))
    return d

def main(_):
    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)

    # Random seed
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    rng_data = np.random.RandomState(FLAGS.seed_data)  # seed shuffling
    print('loading data')
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

    print("Data:") # sanity check input data
    print('train shape %d | batch training %d \ntest shape %d  |  batch  testing %d' \
          % (trainx.shape[0], nr_batches_train, testx.shape[0], nr_batches_test))
    print('histogram train', np.histogram(trainy, bins=10)[0])
    print('histogram test ', np.histogram(testy, bins=10)[0])
    print("histogram labeled", np.histogram(tys, bins=10)[0])
    print("")

    '''construct graph'''
    print('constructing graph')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28], name='labeled_data_input_pl')
    unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28], name='unlabeled_data_input_pl')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input_pl')
    noise_pl = tf.placeholder(tf.float32,[5,100],name='noise_pl')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')
    acc_test_pl_ema = tf.placeholder(tf.float32, [], 'acc_test_pl')
    kl_weight = tf.placeholder(tf.float32, [], 'kl_weight')

    random_z = tf.random_uniform([FLAGS.batch_size, 100], name='random_z')
    d = tf.random_normal([FLAGS.batch_size, 100], mean=0, stddev=1)
    print(d)
    perturb =  FLAGS.scale * d / (tf.expand_dims(tf.norm(d, axis=1), axis=1) * tf.ones([1, 100])+1e-12)
    # random_z_pert = random_z + FLAGS.scale * perturb / (tf.expand_dims(tf.norm(perturb, axis=1), axis=1) * tf.ones([1, 100]))

    generator(random_z,is_training_pl,init=True)
    gen_inp = generator(random_z, is_training=is_training_pl,reuse=True)
    gen_inp_perturb = generator(random_z+perturb, is_training=is_training_pl,reuse=True)
    # gen_inp_pl = generator(noise_pl, is_training=is_training_pl,reuse=True)

    discriminator(inp, is_training_pl, init=True)
    logits_lab, layer_lab = discriminator(inp, is_training_pl,reuse=True)
    logits_unl, layer_real = discriminator(unl, is_training_pl,reuse=True)
    logits_gen, layer_fake = discriminator(gen_inp, is_training_pl,reuse=True)
    logits_gen_perturb, layer_fake_perturb = discriminator(gen_inp_perturb, is_training_pl,reuse=True)

    with tf.name_scope('loss_functions'):
        l_unl = tf.reduce_logsumexp(logits_unl, axis=1)
        l_gen = tf.reduce_logsumexp(logits_gen, axis=1)
        # DISCRIMINATOR
        loss_lab = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lbl, logits=logits_lab))
        loss_unl = - 0.5 * tf.reduce_mean(l_unl) \
                   + 0.5 * tf.reduce_mean(tf.nn.softplus(l_unl)) \
                   + 0.5 * tf.reduce_mean(tf.nn.softplus(l_gen))
        loss_dis = loss_unl + loss_lab

        accuracy_dis = tf.reduce_mean(tf.cast(tf.less(l_unl, 0), tf.float32))
        correct_pred = tf.equal(tf.cast(tf.argmax(logits_lab, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # GENERATOR
        m1 = tf.reduce_mean(layer_real, axis=0)
        m2 = tf.reduce_mean(layer_fake, axis=0)
        loss_gen = tf.reduce_mean(tf.square(m1 - m2))
        fool_rate = tf.reduce_mean(tf.cast(tf.less(l_gen, 0), tf.float32))

        k=[]
        for j in range(10):
            grad = tf.gradients(logits_gen[j], random_z)
            k.append(grad)
        J=tf.stack(k)
        J = tf.squeeze(J)
        J = tf.transpose(J,perm=[1,0,2]) # jacobian
        j_n = tf.square(tf.norm(J,axis=[1,2]))
        j_loss = tf.reduce_mean(j_n)

        # grad = tf.square(tf.norm(logits_gen-logits_gen_perturb,axis=1))
        # j_loss = tf.reduce_mean(grad, axis=0)

        # j_loss = kl_divergence_with_logit(logits_gen_perturb, logits_gen)

        # kl = kl_divergence_with_logit(logits_gen_perturb,logits_gen)
        # print(kl)
        # grad = tf.gradients(kl,perturb)[0]
        # # grad = tf.gradients(kl, [perturb], aggregation_method=2)[0]
        # print(grad)
        # j_loss = tf.square(tf.norm(grad))

        # kl = kl_divergence_with_logit(logits_gen, logits_gen)
        # print(kl)
        # grad = tf.gradients(kl, random_z)[0]
        # # grad = tf.gradients(kl, [perturb], aggregation_method=2)[0]
        # print(grad)
        # j_loss = tf.square(tf.norm(grad))

        # j_loss = entropy_y_x(logits_gen)

        if FLAGS.nabla:
            loss_dis += 0.01 * j_loss
            # loss_dis += 0.01 * j_loss + 0.01 * entropy_y_x(logits_gen) + 0.01 * entropy_y_x(logits_unl)
            # loss_gen += 0.01 * j_loss
            print('grad reg enabled')

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_d, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_g, beta1=0.5, name='gen_optimizer')

        with tf.control_dependencies(update_ops_gen):
            train_gen_op = optimizer_gen.minimize(loss_gen, var_list=gvars)

        dis_op = optimizer_dis.minimize(loss_dis, var_list=dvars)

        ema = tf.train.ExponentialMovingAverage(decay=FLAGS.ma_decay)
        maintain_averages_op = ema.apply(dvars)

        with tf.control_dependencies([dis_op]):
            train_dis_op = tf.group(maintain_averages_op)

        logits_ema, _ = discriminator(inp, is_training_pl, getter=get_getter(ema), reuse=True)
        correct_pred_ema = tf.equal(tf.cast(tf.argmax(logits_ema, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy_ema = tf.reduce_mean(tf.cast(correct_pred_ema, tf.float32))


    with tf.name_scope('summary'):
        with tf.name_scope('discriminator'):
            tf.summary.scalar('discriminator_accuracy', accuracy_dis, ['dis'])
            tf.summary.scalar('loss_discriminator', loss_dis, ['dis'])
            tf.summary.scalar('j_loss', j_loss,['gen'])
            # tf.summary.scalar('cross_entrop', tf.reduce_mean(d_xentropy),['gen'])


        with tf.name_scope('generator'):
            tf.summary.scalar('loss_generator', loss_gen, ['gen'])
            tf.summary.scalar('fool_rate', fool_rate, ['gen'])

        with tf.name_scope('images'):
            tf.summary.image('gen_images', tf.reshape(gen_inp,[-1,28,28,1]),5, ['image'])

        with tf.name_scope('epoch'):
            tf.summary.scalar('accuracy_train', acc_train_pl, ['epoch'])
            tf.summary.scalar('accuracy_test_moving_average', acc_test_pl_ema, ['epoch'])
            tf.summary.scalar('accuracy_test_raw', acc_test_pl, ['epoch'])


        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_im = tf.summary.merge_all('image')
        sum_op_epoch = tf.summary.merge_all('epoch')


    init_gen = [var.initializer for var in gvars][:-3]
    # [print(var.name) for var in gvars]
    saver = tf.train.Saver()
    '''//////perform training //////'''
    print('start training')
    with tf.Session() as sess:
        tf.set_random_seed(rng.randint(2**10))
        sess.run(init_gen)
        init = tf.global_variables_initializer()
        #Data-Dependent Initialization of Parameters as discussed in DP Kingma and Salimans Paper
        sess.run(init, feed_dict={inp: trainx_unl[0:FLAGS.batch_size], is_training_pl: True, kl_weight:0})
        print('initialization done')

        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        train_batch = 0
        for epoch in range(200):
            begin = time.time()
            w_kl = FLAGS.nabla_w * max(1/200 * epoch-1/2,0)
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

            train_loss_lab, train_loss_unl, train_loss_gen, train_acc, test_acc, test_acc_ma, train_j_loss = [0, 0, 0, 0, 0, 0,0]
            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)

                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size

                # train discriminator
                feed_dict = {inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             unl: trainx_unl[ran_from:ran_to],
                             is_training_pl: True,
                             kl_weight:w_kl}
                _, ll, lu, acc, jl,sm = sess.run([train_dis_op, loss_lab, loss_unl, accuracy, j_loss,sum_op_dis],
                                              feed_dict=feed_dict)
                train_loss_lab += ll
                train_loss_unl += lu
                train_j_loss += jl
                train_acc += acc
                if (train_batch % FLAGS.step_print) == 0:
                    writer.add_summary(sm, train_batch)

                # train generator
                _, lg, sm = sess.run([train_gen_op, loss_gen, sum_op_gen], feed_dict={unl: trainx_unl2[ran_from:ran_to],
                                                                                      is_training_pl: True,
                                                                                      noise_pl:np.random.rand(5,100),
                                                                                      kl_weight:w_kl})
                train_loss_gen += lg
                if ((train_batch % FLAGS.step_print) == 0):
                    writer.add_summary(sm, train_batch)

                if ((train_batch % FLAGS.freq_print) == 0):
                    sm = sess.run(sum_op_im, feed_dict={is_training_pl: False})
                    writer.add_summary(sm, train_batch)

                train_batch += 1
            train_loss_lab /= nr_batches_train
            train_loss_unl /= nr_batches_train
            train_acc /= nr_batches_train
            train_j_loss /= nr_batches_train

            # Testing
            for t in range(nr_batches_test):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: testx[ran_from:ran_to],
                             lbl: testy[ran_from:ran_to],
                             is_training_pl: False}
                acc, acc_ema = sess.run([accuracy,accuracy_ema], feed_dict=feed_dict)
                test_acc += acc
                test_acc_ma += acc_ema
            test_acc /= nr_batches_test
            test_acc_ma /= nr_batches_test

            # Plotting
            sum = sess.run(sum_op_epoch, feed_dict={acc_train_pl: train_acc,
                                                    acc_test_pl: test_acc,
                                                    acc_test_pl_ema: test_acc_ma})
            writer.add_summary(sum, epoch)

            # if ((epoch % 10) == 0):
            #     epsilon = 0.15
            #     rand_noise = np.random.rand(5,100)
            #     rand_direction = np.random.randn(5,100)
            #     rand_direction /= (np.expand_dims(np.linalg.norm(rand_direction, axis=1), axis=0).T * np.ones([1, 100]))
            #     print(rand_direction.shape)
            #     sum_inter = tf.summary.image('interpol'+str(var), tf.reshape(gen_inp_pl, [-1, 28, 28, 1]), 30, ['interpol'])
            #     var += 1
            #     for iter in range(10):
            #         noise = rand_noise + (epsilon * iter) *rand_direction
            #         sum = sess.run(sum_inter, {noise_pl:noise,is_training_pl:False})
            #         writer.add_summary(sum, iter )

            print("epoch %d | time = %ds | jloss = %0.4f | loss gen = %.4f | loss lab = %.4f | loss unl = %.4f "
                  "| train acc = %.4f| test acc = %.4f | test acc ma = %.4f"
                  % (epoch, time.time() - begin, train_j_loss, train_loss_gen, train_loss_lab, train_loss_unl, train_acc, test_acc, test_acc_ma))

        save_path = saver.save(sess, os.path.join(FLAGS.logdir, 'model.ckpt'))
        print('model saved in %s'%(save_path))

if __name__ == '__main__':
    tf.app.run()

