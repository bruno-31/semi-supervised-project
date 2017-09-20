import os
import numpy as np
import tensorflow as tf
import cifar10_input, cifar_openai, simple_model
from hyperparam import *
import logging
from preprocessing import unaply

global _wrapped_ops
_wrapped_ops = set()

# Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[console])
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logger = logging.getLogger("gan.test")



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


def _wrap_update_ops(op, *args, **kwargs):
    # this function is to update weights of batch normalization effectively
    global _wrapped_ops  # variable can be modified inside function
    print("build")
    # This function attaches all UPDATE_OPS generated by the operation op to an identity() operation call that happens
    # after op.
    all_ops_pre = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    rv = op(*args, **kwargs)  # op = model
    new_ops = [k for k in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if k not in all_ops_pre]

    print("Update ops: {} has {} update ops.".format(_deep_name(rv), len(new_ops)))
    print("Adding dependency: {} is followed by {}".format(rv.__repr__(), new_ops.__repr__()))

    _wrapped_ops = _wrapped_ops.union(new_ops)
    # We force the new dependencies:
    with tf.control_dependencies(new_ops):
        return _deep_identity(rv)


def _deep_identity(rv):
    if type(rv) is list or type(rv) is tuple:
        return tuple([_deep_identity(x) for x in rv])
    else:
        return tf.identity(rv)


def _deep_name(rv):
    if type(rv) is list or type(rv) is tuple:
        return "({})".format(", ".join(_deep_name(x) for x in rv))
    else:
        return rv.name


def _weight_decay_regularizer(*patterns):
    """Return a weighted L2-norm by pattern matching variable names against weights."""
    _weight_list = []
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        for pattern, weight in patterns:
            if pattern in v.name:
                _weight_list.append(v * weight)
                break
        else:
            _weight_list.append(v)

    for element in _weight_list:
        print(element.name)

    return tf.reduce_sum(tf.square(
        tf.concat([tf.reshape(v, [-1]) for v in _weight_list], axis=0))) / 2 * WEIGHT_DECAY


def main():
    print("logdir :=  " + LOG_DIR)
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    print("Collecting data")
    data = cifar10_input.TrainData()

    logger.info("log directory {} is  ...".format(LOG_DIR))

    # placeholders
    inp_lbl_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE]+list(IMAGE_DIM), name="data_labeled")
    lbl_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NUM_CLASS], name="labels_data")
    inp_unl_pl = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE]+list(IMAGE_DIM), name="data_unlabeled")
    z_seed = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE]+list(Z_DIM), name="z_seed")
    train_pl = tf.placeholder(dtype=tf.bool, shape=[], name="is_training")

    global_step = tf.Variable(initial_value=0, dtype=tf.int32, name='global_step', trainable=False)

    # init_models = True
    # build model
    with tf.variable_scope('generator_model') as gen_scope:
        inp_gen_init = _wrap_update_ops(cifar_openai.generator, z_seed, train_pl, IMAGE_DIM,init=True )
        gen_scope.reuse_variables()
        inp_gen = _wrap_update_ops(cifar_openai.generator, z_seed, train_pl, IMAGE_DIM, init=False)

    with tf.variable_scope('discriminator_model') as dis_scope:
        dis_lbl_init = _wrap_update_ops(cifar_openai.discriminator, inp_lbl_pl, is_training=train_pl, num_classes=11, init=True)
        dis_scope.reuse_variables()

        dis_lbl = _wrap_update_ops(cifar_openai.discriminator, inp_lbl_pl, is_training=train_pl, num_classes=11, init=False)
        dis_unl = _wrap_update_ops(cifar_openai.discriminator, inp_unl_pl, is_training=train_pl, num_classes=11, init=False)
        dis_gen = _wrap_update_ops(cifar_openai.discriminator, inp_gen, is_training=train_pl, num_classes=11, init=False)

    '''loss '''
    label_dis_fake = np.zeros((BATCH_SIZE, 11))
    label_dis_fake[:, -1] = np.ones((BATCH_SIZE))
    label_dis_fake = tf.constant(label_dis_fake)

    with tf.name_scope('generator_loss'):        #     loss have been checked
        # <N+1
        loss_gen = tf.clip_by_value(- tf.reduce_mean(
            tf.log(tf.ones([BATCH_SIZE]) - tf.nn.softmax(dis_gen)[:, -1])),1e-10, 1e6)
        pred_gen_fake = tf.equal(tf.arg_max(dis_gen, 1), tf.arg_max(label_dis_fake, 1))
        fool_rate = 1 - tf.reduce_mean(tf.cast(pred_gen_fake, tf.float32))
        # TODO features matching

    with tf.name_scope('discriminator_loss'):     #     loss have been checked
        # = N+1
        loss_dis_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dis_gen, labels=label_dis_fake))
        correct_pred_dis_fake = tf.equal(tf.arg_max(dis_gen, 1), tf.arg_max(label_dis_fake, 1))
        acc_dis_gen = tf.reduce_mean(tf.cast(correct_pred_dis_fake, tf.float32))
        # < N+1 clip avoid NAN log
        loss_dis_unl = tf.clip_by_value(- tf.reduce_mean(
            tf.log(tf.ones([BATCH_SIZE]) - tf.nn.softmax(dis_unl)[:, -1])),1e-10, 1e6)
        incorrect_pred_dis_real = tf.equal(tf.arg_max(dis_unl, 1), tf.arg_max(label_dis_fake, 1))
        acc_dis_unl = 1 - tf.reduce_mean(tf.cast(incorrect_pred_dis_real, tf.float32))
        # = lbl
        loss_dis_lbl = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dis_lbl, labels=lbl_pl))
        correct_pred_dis_lbl = tf.equal(tf.arg_max(dis_lbl, 1), tf.arg_max(lbl_pl, 1))
        acc_dis_lbl = tf.reduce_mean(tf.cast(correct_pred_dis_lbl, tf.float32))

        loss_dis = loss_dis_lbl + loss_dis_gen + loss_dis_unl
        # loss_dis = loss_dis_gen + loss_dis_unl

    # '''weight reg'''
    # print('adding weight decay discriminator')
    # loss_dis += _weight_decay_regularizer(['/discriminator_model/',WEIGHT_DECAY])
    # print('adding weight decay generator')
    # loss_gen += _weight_decay_regularizer(['/generator_model/', WEIGHT_DECAY])

    # collecting 2 list of training variables corresponding to discriminator and generator
    tvars = tf.trainable_variables()  # return list trainable variables
    d_vars = [var for var in tvars if 'discriminator_model' in var.name]
    g_vars = [var for var in tvars if 'generator_model' in var.name]

    # print("trainable var for discriminator")
    # for var in d_vars:  # display trainable vars for sanity check
    #     print(var.name)
    # print("trainable var for generator")
    # for var in g_vars:
    #     print(var.name)

    # optimizer and dependencies
    optimzer = tf.train.AdamOptimizer(0.0003)
    g_trainer = optimzer.minimize(loss_gen, var_list=g_vars, name='generator_trainer')
    d_trainer = optimzer.minimize(loss_dis, var_list=d_vars, name='discriminator_trainer')

    # build summaries
    with tf.name_scope('Generator'):
        tf.summary.scalar('cross_entropy_loss', loss_gen)
        tf.summary.scalar('fool_rate', fool_rate)

    with tf.name_scope('Discriminator'):
        tf.summary.scalar('Gen_loss', loss_dis_unl)
        tf.summary.scalar('Unlabeled_loss', loss_dis_gen)
        tf.summary.scalar('Labeled_loss', loss_dis_lbl)
        tf.summary.scalar('Total_loss', loss_dis)
        tf.summary.scalar('Accuracy_unlabeled', acc_dis_unl)
        tf.summary.scalar('Accuracy_fake_gen', acc_dis_gen)
        tf.summary.scalar('Accuracy_labeled', acc_dis_lbl)


    with tf.name_scope('Image'):
        tf.summary.image('Generated_images',tf.map_fn(unaply, inp_gen), 10)  # add 10 generated images to summary
        tf.summary.image('Data_images',tf.map_fn(unaply, inp_lbl_pl), 10)

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summary_gen = tf.summary.merge([v for v in summaries if "Discriminator/" in v.name])
    summary_cls = tf.summary.merge([v for v in summaries if "Generator/" in v.name])
    summary_im = tf.summary.merge([v for v in summaries if "Image/" in v.name])

    merged = tf.summary.merge_all()

    inc_global_step = tf.assign(global_step,global_step+1)



    sv = tf.train.Supervisor(logdir=LOG_DIR,
                             global_step=global_step,
                             summary_op=None,
                             save_model_secs=60,
                             init_feed_dict={inp_lbl_pl:next(data.labelled)[0],
                                             lbl_pl:np.append(next(data.labelled)[1], np.zeros((BATCH_SIZE,1)),axis=1), #add 11th lbl
                                             inp_unl_pl:next(data.unlabelled)[0],
                                             z_seed:next(data.rand_vec),
                                             train_pl:True})

    with sv.managed_session() as sess:
        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        batch = sess.run(global_step)
        print("start training from batch "+str(batch))
        while not sv.should_stop():

            batch = sess.run(global_step)

            if batch >= NUM_BATCH:
                sv.saver.save(sess, os.path.join(LOG_DIR+'model.ckpt'), global_step=global_step)
                sv.stop()
                break

            inp, lbl = next(data.labelled)
            lbl = np.append(lbl, np.zeros((BATCH_SIZE, 1)), axis=1)
            inp_unl, _ = next(data.unlabelled)

            _, sum_dis = sess.run([d_trainer, summary_cls], feed_dict = {inp_lbl_pl:inp,
                                                                         lbl_pl:lbl,
                                                                         inp_unl_pl:inp_unl,
                                                                         z_seed:next(data.rand_vec),
                                                                         train_pl:True})
            writer.add_summary(sum_dis, batch)

            inp, lbl = next(data.labelled)
            lbl = np.append(lbl, np.zeros((BATCH_SIZE, 1)), axis=1)
            inp_unl, _ = next(data.unlabelled)

            _, sum_gen =sess.run([g_trainer, summary_gen], feed_dict={inp_lbl_pl:inp,
                                                                     lbl_pl:lbl,
                                                                     inp_unl_pl:inp_unl,
                                                                     z_seed:next(data.rand_vec),
                                                                     train_pl:True})
            writer.add_summary(sum_gen, batch)

            if batch % 50 == 0:
                # print("batch : "+str(batch))
                logger.info('Step {}: Minimizing the error...'.format(batch))

                sum_im = sess.run(summary_im, feed_dict={inp_lbl_pl:inp,
                                                         lbl_pl:lbl,
                                                         inp_unl_pl:inp_unl,
                                                         z_seed:next(data.rand_vec),
                                                         train_pl:True})

                writer.add_summary(sum_im, batch)

            sess.run(inc_global_step)


if __name__ == '__main__':
    main()