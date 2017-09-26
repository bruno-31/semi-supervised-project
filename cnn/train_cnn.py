import os
import time
import numpy as np
import tensorflow as tf
import cifar_model, cifar10_input

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size [128]")
flags.DEFINE_integer("moving_average", 100, "moving average [100]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('log_dir', './log', 'log directory')
flags.DEFINE_integer('seed', 1546, 'seed[1]')
flags.DEFINE_float('learning_rate', 0.003, 'learning_rate[0.003]')
FLAGS = flags.FLAGS

ENABLE_MA = False

def zca_whiten(X, Y, epsilon=1e-5):
    X = X.reshape([-1, 32 * 32 * 3])
    Y = Y.reshape([-1, 32 * 32 * 3])
    # compute the covariance of the image data
    # cov = np.cov(X, rowvar=True)  # cov is (N, N)
    cov = np.dot(X.T, X) / X.shape[0]
    # singular value decomposition
    U, S, V = np.linalg.svd(cov)  # U is (N, N), S is (N,)
    # build the ZCA matrix
    zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))

    # transform the image data       zca_matrix is (3072,3072)
    X_white = np.dot(X, zca_matrix)  # zca is (3072, 3072)
    Y_white = np.dot(Y, zca_matrix)

    X_white = X_white.reshape(-1, 32, 32, 3)
    Y_white = Y_white.reshape([-1, 32, 32, 3])
    return X_white, Y_white


def decayed_lr(batch):
    '''
    compute decayed learning rate : for the last 100 epochs the learning rate
    is linearly decayed to zero.
    '''
    if batch >= 100:
        return FLAGS.learning_rate * (2 - batch / 100)

    return FLAGS.learning_rate


def main(_):
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)

    # Random seed
    rng = np.random.RandomState(FLAGS.seed)

    # load CIFAR-10
    trainx, trainy = cifar10_input._get_dataset(FLAGS.data_dir, 'train')  # float [0 1] images
    testx, testy = cifar10_input._get_dataset(FLAGS.data_dir, 'test')
    # overfitting test
    trainx = trainx[:10000]
    trainy = trainy[:10000]

    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)
    nr_batches_test = int(testx.shape[0] / FLAGS.batch_size)

    # whitten data
    print('Starting zca preprocessing')
    begin = time.time()
    trainx -= np.mean(trainx, axis=0)
    trainx /= np.std(trainx, axis=0)
    testx -= np.mean(trainx, axis=0)
    testx /= np.std(trainx, axis=0)
    trainx, testx = zca_whiten(trainx, testx, epsilon=0.1)
    print('Preprocessing done in : %ds' % (time.time() - begin))

    '''construct graph'''
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='data_input')
    lbl = tf.placeholder(tf.float32, [FLAGS.batch_size, 10], name='lbl_input')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    accuracy_epoch = tf.placeholder(tf.float32, [], name='epoch_pl')
    learning_rate_pl = tf.placeholder(tf.float32, [], name='learning_rate_pl')

    with tf.variable_scope('cnn_model'):
        logits = cifar_model.inference(inp, is_training_pl)

    with tf.name_scope('loss_function'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(lbl, tf.int64)))
        correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(lbl, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        eval_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl, beta1=0.9)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    # vars = tf.trainable_variables() # sanity check trainable vars
    # for var in vars:
    #     print(var.name)

    # Summaries
    with tf.name_scope('per_batch_summary'):
        tf.summary.scalar('loss', loss, ['batch'])
        tf.summary.scalar('accuracy', accuracy, ['batch'])
        tf.summary.scalar('learning rate', learning_rate_pl, ['batch'])

    with tf.name_scope('per_epoch_summary'):
        tf.summary.scalar('accuracy epoch', accuracy_epoch, ['per_epoch'])
        tf.summary.merge(tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                         ['per_epoch'])
        with tf.name_scope('input_data'):
            tf.summary.image('input image', inp, 10, ['per_epoch'])
            tf.summary.histogram('first input image', tf.reshape(inp[0],[-1]), ['per_epoch'])
            tf.summary.histogram('input labels', tf.argmax(lbl, axis=1),['per_epoch'])
            tf.summary.histogram('output logits', tf.argmax(logits, axis=0),['per_epoch'])

    sum_op = tf.summary.merge_all('batch')
    sum_epoch_op = tf.summary.merge_all('per_epoch')

    '''//////perform training //////'''
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        train_batch = 0
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'test'), sess.graph)

        for epoch in range(200):
            begin = time.time()

            # randomly permuted minibatches
            inds = rng.permutation(trainx.shape[0])
            trainx = trainx[inds]
            trainy = trainy[inds]

            train_loss, train_tp, test_tp = [0, 0, 0]

            for t in range(nr_batches_train):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             is_training_pl: True,
                             learning_rate_pl: decayed_lr(epoch)}
                _, ls, tp, sm = sess.run([train_op, loss, eval_correct, sum_op], feed_dict=feed_dict)

                train_loss += ls
                train_tp += tp
                train_batch += 1
                train_writer.add_summary(sm, train_batch)

            train_loss /= nr_batches_train
            train_tp /= trainx.shape[0]

            for t in range(nr_batches_test):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: testx[ran_from:ran_to],
                             lbl: testy[ran_from:ran_to],
                             is_training_pl: False}

                test_tp += sess.run(eval_correct, feed_dict=feed_dict)

            test_tp /= testx.shape[0]

            '''/////epoch summary/////'''
            sm = sess.run(sum_epoch_op, {accuracy_epoch: train_tp,
                                             inp: trainx[:FLAGS.batch_size],
                                             lbl:trainy[:FLAGS.batch_size],
                                             is_training_pl: False})
            train_writer.add_summary(sm, epoch)
            x = np.random.randint(0,testx.shape[0]-FLAGS.batch_size) # random batch extracted in testx
            sm = sess.run(sum_epoch_op, {accuracy_epoch: test_tp,
                                         inp: testx[x:x+FLAGS.batch_size],
                                         lbl: testy[x:x+FLAGS.batch_size],
                                         is_training_pl:False})
            test_writer.add_summary(sm, epoch)
            print("Epoch %d, Train batch %d, time = %ds : loss train = %.4f, train acc = %.4f, ""test acc = %.4f" %
                  (epoch, train_batch, time.time() - begin, train_loss, train_tp, test_tp))

if __name__ == '__main__':
    tf.app.run()
