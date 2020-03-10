from data_management.manipulate_data import import_cifar100, load_cifar100, import_boston_housing, load_boston_housing
from project.train import train
from project.test import test

if __name__ == "__main__":
    _train = 0

    (x_train, y_train), (x_test, y_test) = load_boston_housing()

    # Training Parameters:
    learning_rate = 0.01
    num_epochs = 300
    num_models = 50
    batch_size = 64
    # Testing Parameters:
    checkpoint_file = "epoch_280.ckpt"
    if _train:
        train(x_train, y_train, learning_rate, num_epochs, batch_size, checkpoint_frequency=10, num_models=num_models)
    else:
        test(x_test, y_test, checkpoint_file)