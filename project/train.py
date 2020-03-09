import tensorflow as tf
import numpy as np

def train(x_train, y_train):
    # Parameters:
    input_nodes = np.shape(x_train)[1]
    hidden_layer_1 = 1
    learning_rate = 0.01
    num_epochs = 3
    batch_size = 64

    x = tf.placeholder(tf.float32, [None, input_nodes])
    W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_layer_1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_layer_1]))
    y = tf.math.sigmoid(tf.matmul(x, W1) + b1)

    targets = np.reshape(y_train, (np.shape(y_train)[0], 1))
    cost = tf.reduce_sum(tf.math.square(y - targets))
    predicted = tf.argmax(y, 1)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Miscellaneous quantities:
    sample_count = np.shape(x_train)[0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_iteration in range(num_epochs):
            for batch in range(int(sample_count)):
                batch_x = x_train[batch * batch_size: (1 + batch) * batch_size]
                print(batch_x.shape)
                batch_y = y_train[batch * batch_size: (1 + batch) * batch_size]
                sess.run([optimizer], feed_dict={x: batch_x, y: batch_y})

