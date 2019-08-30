import gzip
import struct
import numpy as np


def _make_onehot(y, depth):
    eyes = np.eye(depth)
    return eyes[y].reshape([-1, depth])


def read_mnist(one_hot=False, standard=False):
    """返回mnist数据，两个参数分别是one_hot和standard来控制是否将数据归一化和变成OneHot
        分别返回训练数据，训练标签，测试数据，测试标签"""
    with gzip.open(r'../MNIST_data/train-images-idx3-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>IIII')
        magic, numimages, numrows, numcolumns = struct.unpack_from('>IIII', data, 0)
        data = struct.unpack_from('>'+str(784*numimages)+'B', data, start)
        train_x = np.array(data).reshape(numimages, numrows, numcolumns, 1)
        if standard:
            train_x = train_x / 255.0

    with gzip.open(r'../MNIST_data/train-labels-idx1-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>II')
        magic, numimages = struct.unpack_from('>II', data, 0)
        data = struct.unpack_from('>'+str(1*numimages)+'B', data, start)
        train_y = np.array(data).reshape(numimages, 1)
        if one_hot:
            train_y = np.array(data)
            train_y = _make_onehot(y=train_y, depth=10)

    with gzip.open(r'../MNIST_data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>IIII')
        magic, numimages, numrows, numcolumns = struct.unpack_from('>IIII', data, 0)
        data = struct.unpack_from('>'+str(784*numimages)+'B', data, start)
        test_x = np.array(data).reshape(numimages, numrows, numcolumns, 1)
        if standard:
            test_x = test_x / 255.0

    with gzip.open(r'../MNIST_data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>II')
        magic, numimages = struct.unpack_from('>II', data, 0)
        data = struct.unpack_from('>'+str(1*numimages)+'B', data, start)
        test_y = np.array(data).reshape(numimages, 1)
        if one_hot:
            test_y = np.array(data)
            test_y = _make_onehot(y=test_y, depth=10)
    return train_x, train_y, test_x, test_y


def read_fashion_mnist(one_hot=False, standard=False):
    """返回fashion_mnist数据，两个参数分别是one_hot和standard来控制是否将数据归一化和变成OneHot
        分别返回训练数据，训练标签，测试数据，测试标签，按照标签获得名字的一个字典"""
    index2name = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal',
                  6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
    with gzip.open(r'../fashion_MNIST/t10k-images-idx3-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>IIII')
        magic, numimages, numrows, numcolumns = struct.unpack_from('>IIII', data, 0)
        data = struct.unpack_from('>'+str(784*numimages)+'B', data, start)
        train_x = np.array(data).reshape(numimages, numrows, numcolumns, 1)
        if standard:
            train_x = train_x / 255.0

    with gzip.open(r'../fashion_MNIST/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>II')
        magic, numimages = struct.unpack_from('>II', data, 0)
        data = struct.unpack_from('>'+str(1*numimages)+'B', data, start)
        train_y = np.array(data).reshape(numimages, 1)
        if one_hot:
            train_y = np.array(data)
            train_y = _make_onehot(y=train_y, depth=10)

    with gzip.open(r'../fashion_MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>IIII')
        magic, numimages, numrows, numcolumns = struct.unpack_from('>IIII', data, 0)
        data = struct.unpack_from('>'+str(784*numimages)+'B', data, start)
        test_x = np.array(data).reshape(numimages, numrows, numcolumns, 1)
        if standard:
            test_x = test_x / 255.0

    with gzip.open(r'../fashion_MNIST/train-labels-idx1-ubyte.gz', 'rb') as f:
        data = f.read()
        start = struct.calcsize('>II')
        magic, numimages = struct.unpack_from('>II', data, 0)
        data = struct.unpack_from('>'+str(1*numimages)+'B', data, start)
        test_y = np.array(data).reshape(numimages, 1)
        if one_hot:
            test_y = np.array(data)
            test_y = _make_onehot(y=test_y, depth=10)
    return train_x, train_y, test_x, test_y, index2name
