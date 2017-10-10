import time

import numpy as np

import cifar10_input

data_dir = './data/cifar-10-python'


def zca_whiten(X, Y, epsilon=1e-5):
    X = X.reshape([-1, 32 * 32 * 3])
    Y = Y.reshape([-1, 32 * 32 * 3])
    # compute the covariance of the image data
    # cov = np.cov(X, rowvar=True)  # cov is (N, N)
    # print('cov computation..')
    beg_cov = time.time()
    cov = np.dot(X.T, X) / X.shape[0]
    # singular value decomposition
    # print('done, time : %ds' % (time.time() - beg_cov))
    # print('svd computation..')
    beg_svd = time.time()
    U, S, V = np.linalg.svd(cov)  # U is (N, N), S is (N,)
    # print('done, time : %ds' % (time.time() - beg_svd))
    # build the ZCA matrix
    # print('zca computation..')
    beg_zca = time.time()
    zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))
    # print('zca done, time : %ds' % (time.time() - beg_zca))

    # transform the image data       zca_matrix is (N,N)
    X_white = np.dot(X, zca_matrix)  # zca is (N, 3072)
    Y_white = np.dot(Y, zca_matrix, Y)  # zca is (N, 3072)

    X_white = X_white.reshape(-1, 32, 32, 3)
    Y_white = Y_white.reshape([-1, 32, 32, 3])
    return X_white, Y_white


def test(X):
    X = X.reshape([-1, 32 * 32 * 3])
    cov = np.dot(X.T, X) / X.shape[0]
    epsilon = np.linalg.norm(cov-np.eye(X.shape[-1]))
    return epsilon

 # load CIFAR-10
trainx, trainy = cifar10_input._get_dataset(data_dir, 'train')  # float [0 1] images
testx, testy = cifar10_input._get_dataset(data_dir, 'test')

print('mean and std computation ..')
begin = time.time()
m = np.mean(trainx, axis=0)
std = np.mean(trainx, axis=0)
trainx -= m
# trainx /= std
testx -= m
# testx /= std
print('done, time : %ds'%(time.time()-begin))

print('residu cov train :  %.4f'%(test(trainx)))
print('residu cov test :  %.4f'%(test(testx)))

begin = time.time()
trx, ttx = zca_whiten(trainx, testx, epsilon=1e-5)
print('computation over, time : %ds'%(time.time()-begin))
print('residu cov train :  %.4f'%(test(trx)))
print('residu cov test :  %.4f'%(test(ttx)))

begin = time.time()
trx, ttx = zca_whiten(trainx, testx, epsilon=0.1)
print('computation over, time : %ds'%(time.time()-begin))
print('residu cov train :  %.4f'%(test(trx)))
print('residu cov test :  %.4f'%(test(ttx)))

begin = time.time()
trx, ttx = zca_whiten(trainx, testx, epsilon=0.01)
print('computation over, time : %ds'%(time.time()-begin))
print('residu cov train :  %.4f'%(test(trx)))
print('residu cov test :  %.4f'%(test(ttx)))