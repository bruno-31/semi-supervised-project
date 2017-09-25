import os
import time
import numpy as np
import tensorflow as tf
import cifar_model, cifar10_input

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size [128]")
flags.DEFINE_integer("moving_average", 10, "moving average [100]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('log_dir', './log', 'log directory')
flags.DEFINE_float('seed', 1546, 'seed[1]')
FLAGS = flags.FLAGS


def zca_whiten(X, Y):
    X = X.reshape([-1, 32 * 32 * 3])
    Y = Y.reshape([-1, 32 * 32 * 3])
    # compute the covariance of the image data
    cov = np.cov(X, rowvar=True)  # cov is (N, N)
    # singular value decomposition
    U, S, V = np.linalg.svd(cov)  # U is (N, N), S is (N,)
    # build the ZCA matrix
    epsilon = 1e-5
    zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))

    # transform the image data       zca_matrix is (N,N)
    X_white = np.dot(zca_matrix, X)  # zca is (N, 3072)
    Y_white = np.dot(zca_matrix, Y)  # zca is (N, 3072)

    X_white = X_white.reshape(-1, 32, 32, 3)
    Y_white = Y_white.reshape([-1, 32, 32, 3])
    return X_white, Y_white


def main(_):
    print("logdir :=  " + FLAGS.log_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)

    # Random seed
    rng = np.random.RandomState(FLAGS.seed)

    # load CIFAR-10
    trainx, trainy = cifar10_input._get_dataset(FLAGS.data_dir, 'train')  # float [0 1] images
    testx, testy = cifar10_input._get_dataset(FLAGS.data_dir, 'test')

    # # overfitting test
    # trainx = trainx[:1000]
    # trainy = trainy[:1000]

    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)
    nr_batches_test = int(testx.shape[0] / FLAGS.batch_size)

    # whitten data
    print('starting preprocessing')
    trainx -= np.mean(trainx, axis=0)
    # trainx /= np.std(trainx, axis=0)
    testx -= np.mean(trainx, axis=0)
    # testx /= np.std(trainx, axis=0)
    # trainx, testx = zca_whiten(trainx, testx)
    print('preprocessing done')

    '''construct graph'''
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 32, 32, 3], name='data_input')
    lbl = tf.placeholder(tf.float32, [FLAGS.batch_size, 10], name='lbl_input')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    accuracy_epoch = tf.placeholder(tf.float32,[])

    with tf.variable_scope('cnn_model') as cnn_scope:
        logits = cifar_model.inference(inp, is_training_pl)

    with tf.name_scope('loss_function'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(lbl,tf.int64)))
        correct_prediction = tf.equal(tf.arg_max(logits, 1), tf.arg_max(lbl, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        eval_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.003, beta1=0.9)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    # TODO add decayed loss

    vars = tf.trainable_variables()
    for var in vars:
        print(var.name)

    # Summaries
    tf.summary.scalar('loss', loss,['train'])
    tf.summary.scalar('accuracy', accuracy, ['train'])
    tf.summary.scalar('accuracy epoch', accuracy_epoch,['epoch'])

    sum_op = tf.summary.merge_all('train')
    sum_epoch_op = tf.summary.merge_all('epoch')

    '''//////perform training //////'''
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        train_batch = 0
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir,'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir,'test'), sess.graph)

        print(nr_batches_train)

        for epoch in range(1000):
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
                             is_training_pl: True}

                _, ls, tp, sm = sess.run([train_op, loss, eval_correct, sum_op], feed_dict=feed_dict)
                if t >= (nr_batches_train-FLAGS.moving_average): # moving average training
                    train_loss += ls
                    train_tp += tp
                train_batch += 1
                train_writer.add_summary(sm, train_batch)
            train_loss /= FLAGS.moving_average
            train_tp /= (FLAGS.batch_size * FLAGS.moving_average)

            for t in range(nr_batches_test):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: testx[ran_from:ran_to],
                             lbl: testy[ran_from:ran_to],
                             is_training_pl: False}

                test_tp += sess.run(eval_correct, feed_dict=feed_dict)

            test_tp /= testx.shape[0]

            sm = sess.run(sum_epoch_op, {accuracy_epoch:test_tp})
            test_writer.add_summary(sm, epoch)
            sm = sess.run(sum_epoch_op, {accuracy_epoch: train_tp})
            train_writer.add_summary(sm, epoch)

            print("Epoch %d, Train batch %d, time = %ds : loss train (last 100) = %.4f, train acc (last 100) = %.4f, test acc = %.4f" % (
                epoch, train_batch,time.time() - begin, train_loss, train_tp, test_tp))


if __name__ == '__main__':
    tf.app.run()
