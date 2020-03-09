import tensorflow as tf
import numpy as np

checkpoint_dir = "weights/"

def train(x_train, y_train):
    y_train = y_train.reshape(np.shape(y_train)[0], np.shape(y_train[1]))

    # Parameters:
    input_nodes = np.shape(x_train)[1]
    hidden_layer_1 = 1
    learning_rate = 0.01
    num_epochs = 3
    batch_size = 64

    # Defining Layers:
    # Placeholder for batch of inputs:
    x = tf.placeholder(tf.float32, [None, input_nodes])
    # Layer 1 variables:
    W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_layer_1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_layer_1]))
    y = tf.math.sigmoid(tf.matmul(x, W1) + b1)
    # Placeholder for batch of targets:
    y_ = tf.placeholder(tf.float32, [None, 1])

    cost = tf.reduce_sum(tf.math.square(y - y_))
    predicted = tf.argmax(y, 1)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Miscellaneous quantities:
    sample_count = np.shape(x_train)[0]

    # For weight saving:
    saver = tf.train.Saver(max_to_keep=1)
    checkpoint = "weights.ckpt"

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_iteration in range(num_epochs):
            for batch in range(int(sample_count / batch_size)):
                batch_x = x_train[batch * batch_size: (1 + batch) * batch_size]
                batch_y = y_train[batch * batch_size: (1 + batch) * batch_size]
                # Instantiating the inputs and targets with the batch values:
                sess.run([optimizer], feed_dict={x: batch_x, y_: batch_y})
        saver.save(sess, checkpoint)
