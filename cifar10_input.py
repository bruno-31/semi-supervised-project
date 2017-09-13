# CIFAR10 Downloader

import logging
import pickle
import math
import os
import errno
import tarfile
import shutil
import preprocessing
import numpy as np
from hyperparam import *
import urllib3

logger = logging.getLogger(__name__)

_shuffle = True

def get_train():
    return _get_dataset("train")


def get_test():
    return _get_dataset("test")


def get_shape_input():
    return (None, 32, 32, 3)


def get_shape_label():
    return (None,)


def num_classes():
    return 10


def _unpickle_file(filename):
    logger.debug("Loading pickle file: {}".format(filename))

    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Reorder the data
    img = data[b'data']
    img = img.reshape([-1, 3, 32, 32])
    img = img.transpose([0, 2, 3, 1])
    # Load labels
    lbl = np.array(data[b'labels'])

    return img, lbl


def _get_dataset(split):
    assert split == "test" or split == "train"
    path = "data"
    dirname = "cifar-10-batches-py"
    data_url = "http://10.217.128.198/datasets/cifar-10-python.tar.gz"

    if not os.path.exists(os.path.join(path, dirname)):
        # Extract or download data
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        file_path = os.path.join(path, data_url.split('/')[-1])
        if not os.path.exists(file_path):
            # Download
            logger.warn("Downloading {}".format(data_url))
            with urllib3.PoolManager().request('GET', data_url, preload_content=False) as r, \
                    open(file_path, 'wb') as w:
                shutil.copyfileobj(r, w)

        logger.warn("Unpacking {}".format(file_path))
        # Unpack data
        tarfile.open(name=file_path, mode="r:gz").extractall(path)

    # Import the data
    filenames = ["test_batch"] if split == "test" else \
        ["data_batch_{}".format(i) for i in range(1, 6)]

    imgs = []
    lbls = []
    for f in filenames:
        img, lbl = _unpickle_file(os.path.join(path, dirname, f))
        imgs.append(img)
        lbls.append(lbl)

    # Now we flatten the arrays
    imgs = np.concatenate(imgs)
    lbls = np.concatenate(lbls)

    # Convert images to [0..1] range
    imgs = imgs.astype(np.float32) / 255.0
    return imgs, lbls


def get_data(split, batch_size, labelled_fraction=1.0):
    """The provider function in a dataset.

    This function provides the training, development, and test sets as necessary.

    Args:
        split      (str): "train", "develop", or "test"; to indicate the appropriate split. "train" provides an infinite
                          generator that constantly loops over the input data; the other two end after all data is
                          consumed. Note: "develop" is not implemented.
        batch_size (int): The number of images to provide in each batch. Finite generators will discard partial batches.
        labelled_fraction (float): The fraction of "train" data to provide with labels.

    Returns:
        tuple: (if "train")
            generator: (infinite)
                tuple:
                    (batch_size, 32, 32, 3) Unlabelled image
                    (batch_size, 10, 1)     Zeros
            generator: (infinite)
                tuple:
                    ndarray: (batch_size, 32, 32, 3) Labelled image
                    ndarray: (batch_size, 10, 1)     1-hot label encoding
        tuple: (otherwise)
            int: number of examples in the generator
            generator: (finite)
                tuple:
                    ndarray: (batch_size, 32, 32, 3) Labelled image
                    ndarray: (batch_size, 10, 1)     1-hot label encoding

    """

    if split == "train":
        train = _get_dataset("train")
        # Shuffle the training data
        if _shuffle:
            x, y = train
            p = np.random.permutation(x.shape[0])
            x = x[p, ...]
            y = y[p, ...]
            train = (x, y)

        num_lbl = math.floor(train[0].shape[0] * labelled_fraction)
        # Return (unlabelled, labelled).
        # We strip the labels out and replace them with None, to better data hygene
        return (map(lambda x: (x[0], None), gen(train, batch_size)),
                gen((train[0][:num_lbl, ...], train[1][:num_lbl, ...]), batch_size))

    elif split == "test":
        test = _get_dataset("test")
        nt = (test[0].shape[0] // batch_size) * batch_size
        return (nt, gen(test, batch_size, False))

    assert not "get_data must be called with \"train\" or \"test\"."


def gen(d, batch_size, wrap=True):
    x, y = d
    NUM_CLASS = 10

    def _to_1hot(lbl):
        z = np.zeros(shape=(batch_size, NUM_CLASS))
        z[np.arange(batch_size), lbl.flatten()] = 1
        return z

    assert x.shape[0] == y.shape[0]

    if wrap:
        # The first index of the next batch:
        i = 0  # Type: int
        while True:
            j = i + batch_size
            # If we wrap around the back of the dataset:
            if j >= x.shape[0]:
                rv = list(range(i, x.shape[0])) + list(range(0, j - x.shape[0]))
                yield (x[rv, ...], _to_1hot(y[rv, ...]))
                i = j - x.shape[0]
            else:
                yield (x[i:j, ...], _to_1hot(y[i:j, ...]))
                i = j
    else:
        i = 0
        j = 0
        while j < (x.shape[0] // batch_size) * batch_size:
            j = i + batch_size
            yield (x[i:j, ...], _to_1hot(y[i:j, ...]))
            i = j


def _image_stream_batch(itr, batch_size):
    rx, ry = next(itr)
    if ry is None:
        while True:
            while rx.shape[0] < batch_size:
                ax, ay = next(itr)
                rx = np.concatenate((rx, ax))
            yield (rx[:batch_size, ...], None)
            rx = rx[batch_size:, ...]
    else:
        while True:
            assert rx.shape[0] == ry.shape[0]
            while rx.shape[0] < batch_size:
                ax, ay = next(itr)
                rx = np.concatenate((rx, ax))
                ry = np.concatenate((ry, ay))
            yield (rx[:batch_size, ...], ry[:batch_size, ...])
            rx = rx[batch_size:, ...]
            ry = ry[batch_size:, ...]


# Produces a stream of random data
def _random_stream(batch_size : int, img_size):
    sz = [batch_size] + list(img_size)
    while True:
        yield np.random.normal(size=sz)


class TrainData(object):
    def __init__(self):
        unlabelled, labelled = get_data("train", BATCH_SIZE, LABELED_DATA)
        print("Training data loaded from disk.")

        unlabelled = map(preprocessing.apply_train, unlabelled)
        labelled = map(preprocessing.apply_train, labelled)
        print("Applied training preprocessor.")

        self.unlabelled = _image_stream_batch(unlabelled, BATCH_SIZE)
        self.labelled = _image_stream_batch(labelled, BATCH_SIZE)
        self.rand_vec = _random_stream(BATCH_SIZE, Z_DIM)
        '''
        self.rand_vec        = _random_stream(args.hyperparam.BATCH_SIZE, args.hyperparam.SEED_DIM)
        self.rand_label_vec  = _random_1hot_stream(args.hyperparam.BATCH_SIZE, args.hyperparam.NUM_CLASSES)
        # Present images them in chunks of exactly batch-size:
        self.unlabelled      = _image_stream_batch(unlabelled, args.hyperparam.BATCH_SIZE)
        self.labelled        = _image_stream_batch(labelled, args.hyperparam.BATCH_SIZE)

        # Use to label a discriminator batch as real
        self._label_dis_real = map(lambda a, b: a + b,
                            _value_stream(args.hyperparam.BATCH_SIZE, Y_REAL),
                            _function_stream(lambda: args.hyperparam.label_smoothing(True, args.hyperparam.BATCH_SIZE)))
        # Use to label a discriminator batch as fake
        self._label_dis_fake = map(lambda a, b: a + b,
                            _value_stream(args.hyperparam.BATCH_SIZE, Y_FAKE),
                            _function_stream(lambda: args.hyperparam.label_smoothing(False, args.hyperparam.BATCH_SIZE)))
        # Random flipping support
        self.label_dis_real = _selection_stream([args.hyperparam.label_flipping_prob], self._label_dis_fake, self._label_dis_real)
        self.label_dis_fake = _selection_stream([args.hyperparam.label_flipping_prob], self._label_dis_real, self._label_dis_fake)
        # Use to label a generator batch as real
        self.label_gen_real = _value_stream(args.hyperparam.BATCH_SIZE, Y_REAL)
        '''

