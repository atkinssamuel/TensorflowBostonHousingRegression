import tensorflow as tf
import numpy as np
from project.train import checkpoint_dir, results_folder
import matplotlib.pyplot as plt

def test(x_test, y_test, checkpoint_file):
    y_test = y_test.reshape((np.shape(y_test)[0], 1))

    # Parameters:
    input_nodes = np.shape(x_test)[1]
    hidden_layer_1 = 32
    hidden_layer_2 = 1

    # Defining Layers:
    # Placeholder for batch of inputs:
    x = tf.placeholder(tf.float32, [None, input_nodes])
    # Layer 1 variables:
    W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_layer_1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_layer_1]))
    y1 = tf.math.sigmoid(tf.matmul(x, W1) + b1)
    # layer 2 variables:
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
    b2 = tf.Variable(tf.zeros([hidden_layer_2]))
    y = tf.matmul(y1, W2) + b2
    # Placeholder for batch of targets:
    y_ = tf.placeholder(tf.float32, [None, 1])

    # For weight saving:
    saver = tf.train.Saver()
    checkpoint = checkpoint_dir + checkpoint_file

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)

        test_output = sess.run(y, feed_dict={x: x_test, y_: y_test})
        plt.plot(test_output, label="Approximation")
        plt.plot(y_test, label="Actual")
        plt.title("Actual vs. Approximation")
        plt.ylabel("Estimated Revenue in $")
        plt.legend()
        plt.savefig(results_folder + "testing_results.png")
        plt.show()
        test_loss = np.sum(np.square(test_output - y_test)/np.shape(test_output)[0])
        print("Test Loss =", test_loss)

    return