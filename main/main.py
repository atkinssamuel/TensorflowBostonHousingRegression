# tf 1.14.0
import tensorflow as tf
import keras
from keras.datasets import cifar100
import numpy as np
from data_management.manipulate_data import import_cifar100, load_cifar100
from main.train import train

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_cifar100()
    train(x_train, y_train)