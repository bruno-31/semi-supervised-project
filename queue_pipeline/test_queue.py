import tensorflow as tf
import numpy as np
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator, img_to_array
import time

inp = tf.placeholder(tf.float32, [10, 28, 28, 1], name='inp_img')
data = np.load('../data/mnist.npz')
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(np.float32)
trainx = np.reshape(trainx, [-1, 28, 28, 1])

def distorted_image(inp):
    x = inp
    x = tf.random_crop(x,[20,20,1])
    x = tf.image.random_flip_left_right(x)
    return x

def preprocess_batch(batch):
    x = tf.map_fn(distorted_image, batch, dtype=None, parallel_iterations=10)
    x = tf.image.resize_nearest_neighbor(x,[28,28])
    return x

# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)
# with tf.device('/cpu:0'):
#     inp_flow = train_datagen.flow(trainx, batch_size=10)
#
# # with tf.device('/cpu:0'):
# #     for batch in train_datagen.flow(x, batch_size=10):
# #         i+=1
# #         print(batch.shape)
# #         if i>10:
# #             break
# #
#
# y = tf.identity(inp)

with tf.Session() as sess:
    begin = time.time()
    print("tf")
    x=sess.run(preprocess_batch(trainx[0:20]))
    print('total time %0.2f'%(time.time()-begin))



