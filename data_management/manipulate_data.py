import keras
from keras.datasets import cifar100
from keras.datasets import boston_housing
import numpy as np

directory = "data_management/"
cifar100_dir = directory + "cifar100/"
boston_100_dir = directory + "boston_housing/"


def import_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    np.save(cifar100_dir + 'x_train', x_train)
    np.save(cifar100_dir + 'y_train', y_train)
    np.save(cifar100_dir + 'x_test', x_test)
    np.save(cifar100_dir + 'y_test', y_test)


def load_cifar100():
    x_train = np.load(cifar100 + "x_train.npy")
    y_train = np.load(cifar100 + "y_train.npy")
    x_test = np.load(cifar100 + "x_test.npy")
    y_test = np.load(cifar100 + "x_test.npy")

    return (x_train, y_train), (x_test, y_test)


def import_boston_housing():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    np.save(boston_100_dir + 'x_train', x_train)
    np.save(boston_100_dir + 'y_train', y_train)
    np.save(boston_100_dir + 'x_test', x_test)
    np.save(boston_100_dir + 'y_test', y_test)


def load_boston_housing():
    x_train = np.load(boston_100_dir + "x_train.npy")
    y_train = np.load(boston_100_dir + "y_train.npy")
    x_test = np.load(boston_100_dir + "x_test.npy")
    y_test = np.load(boston_100_dir + "x_test.npy")

    return (x_train, y_train), (x_test, y_test)

