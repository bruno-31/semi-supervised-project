import tensorflow as tf
import numpy as np
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator, img_to_array

inp = tf.placeholder(tf.float32, [10,28,28,1], name='inp_img')
data = np.load('../data/mnist.npz')
trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(np.float32)
trainx = np.reshape(trainx,[-1,28,28,1])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
with tf.device('/cpu:0'):
    inp_flow = train_datagen.flow(trainx, batch_size=10)

# with tf.device('/cpu:0'):
#     for batch in train_datagen.flow(x, batch_size=10):
#         i+=1
#         print(batch.shape)
#         if i>10:
#             break
#

y = tf.identity(inp)

with tf.Session() as sess:
    for t in range(10):
        feed_dict={inp:inp_flow.next()}
        sess.run(y,feed_dict)

