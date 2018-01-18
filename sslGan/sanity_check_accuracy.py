import os
import time

import numpy as np
import tensorflow as tf

from data import cifar10_input
from data import svhn_data
from svhn_large import discriminator, generator
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


dir = './logsvhn/train_only_batch100/model-400'
data_dir =  './data/svhn'
batch = 100

testx, testy = svhn_data.load(data_dir, 'test')
def rescale(mat):
    return np.transpose(((-127.5 + mat) / 127.5), (3, 0, 1, 2))
testx = rescale(testx)
nr_batches_test = int(testx.shape[0] / 100)
print('num batch: ',nr_batches_test)
print('num examples: ',testy.shape[0])

idx = np.random.permutation(testx.shape[0])
testx=testx[idx]
testy=testy[idx]

print('batch size ',batch)
print('data test loaded')
print('dir: ', dir)

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(dir+'.meta')
    new_saver.restore(sess, dir)
    print('model restored')
    correct_op = tf.get_collection("correct_op")[0]

    correct = 0

    for t in range(nr_batches_test):
        ran_from = t * 100
        ran_to = (t + 1) * 100
        # train discriminator
        feed_dict = {'labeled_data_input_pl:0': testx[ran_from:ran_to],
                     'lbl_input_pl:0': testy[ran_from:ran_to],
                     'is_training_pl:0': False}

        corr = sess.run(correct_op, feed_dict=feed_dict)
        corr = np.sum(corr)
        correct += corr
    print(t)
    print('correct: ', correct)
    print('accuracy: ', correct / (nr_batches_test*100))