import keras
from keras.datasets import cifar100
import numpy as np

directory = "data_management/datasets/"

def import_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    np.save(directory + 'x_train', x_train)
    np.save(directory + 'y_train', y_train)
    np.save(directory + 'x_test', x_test)
    np.save(directory + 'y_test', y_test)

def load_cifar100():
    x_train = np.load(directory + "x_train.npy")
    y_train = np.load(directory + "y_train.npy")
    x_test = np.load(directory + "x_test.npy")
    y_test = np.load(directory + "x_test.npy")

    return (x_train, y_train), (x_test, y_test)
