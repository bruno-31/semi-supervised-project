# Preprocessor that normalizes the input.

import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        # std = np.max([np.std(image_np[i, ...]), 1.0 / np.sqrt(32 * 32 * 3)])
        std = 1
        image_np[i, ...] = (image_np[i, ...] - mean) / std
    return image_np


def normalize_image(image_np):
    img = image_np.astype(np.float32) / 127.5 - 1.0
    return img


def normalize_range(data):
    for i in range(len(data)):
        IM = data[i]
        IM = IM - np.min(IM)
        IM = IM / np.max(IM)
        data[i, ...] = IM
    return data


def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros((len(batch_data), 32, 32, 3))

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset + 32,
                                y_offset:y_offset + 32, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch


def horizontal_flip(image, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = np.fliplr(image)

    return image


def pad_data(data, padding_size):
    '''
    Read all the train data into numpy array and add padding_size of 0 paddings on each side of the
    image
    :param padding_size: int. how many layers of zero pads to add on each side?
    :return: all the train data and corresponding labels
    '''

    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)

    return data


def random_color(data, std_color_var):
    for i in range(len(data)):
        im = data[i]
        x = im.reshape((-1, 3))
        pca = PCA(n_components=3)
        pca.fit(x)
        alpha = np.random.rand(1) * std_color_var
        x = x + alpha * pca.components_[:, 0]
        data[i, ...] = x.reshape((32, 32, 3))
    return data


def apply_train(inp):
    """
    Applies transformations to input and labels, returning a tuple of (preprocessed_input, preprocessed_labels).
    input is an ndarray of shape (batch_size, height, width, channels)
    label is an ndarray of shape (batch_size, num_classes)
    You may create or remove elements from the batch as necessary.
    """
    img, label = inp
    img = img.astype(np.float32)
    img = pad_data(img,2)
    img = random_crop_and_flip(img,2)
    img = random_color(img, 0.1)
    img = whitening_image(img)
    return (img, label)


def apply_test(inp):
    img, label = inp
    img = img.astype(np.float32)
    img = whitening_image(img)
    return (img, label)


def unaply(img):
    image = img-tf.reduce_min(img)
    image = image / tf.reduce_max(image)
    return image
    # return tf.cast((image + 1) * 127.5, tf.uint8)