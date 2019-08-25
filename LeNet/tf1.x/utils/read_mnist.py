import gzip
import struct
import numpy as np


def make_onehot(y, depth):
    eyes = np.eye(depth)
    return eyes[y].reshape([-1, depth])


def read_mnist(one_hot=False):
    with gzip.open(r'MNIST_data/train-images-idx3-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>IIII')
        magic, numimages, numrows, numcolumns = struct.unpack_from('>IIII', data, 0)
        data = struct.unpack_from('>'+str(784*numimages)+'B', data, start)
        train_x = np.array(data).reshape(numimages, numrows, numcolumns, 1)

    with gzip.open(r'MNIST_data/train-labels-idx1-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>II')
        magic, numimages = struct.unpack_from('>II', data, 0)
        data = struct.unpack_from('>'+str(1*numimages)+'B', data, start)
        train_y = np.array(data).reshape(numimages, 1)
        if one_hot:
            train_y = np.array(data)
            train_y = make_onehot(y=train_y, depth=10)

    with gzip.open(r'MNIST_data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>IIII')
        magic, numimages, numrows, numcolumns = struct.unpack_from('>IIII', data, 0)
        data = struct.unpack_from('>'+str(784*numimages)+'B', data, start)
        test_x = np.array(data).reshape(numimages, numrows, numcolumns, 1)

    with gzip.open(r'MNIST_data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>II')
        magic, numimages = struct.unpack_from('>II', data, 0)
        data = struct.unpack_from('>'+str(1*numimages)+'B', data, start)
        test_y = np.array(data).reshape(numimages, 1)
        if one_hot:
            test_y = np.array(data)
            test_y = make_onehot(y=test_y, depth=10)
    return train_x, train_y, test_x, test_y


