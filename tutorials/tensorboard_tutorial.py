# tf 1.14.0
import tensorflow as tf

a = tf.constant(5.0, tf.float32)
b = tf.constant(3.0, tf.float32)

c = a * b
sess = tf.Session()

# For tensorboard:
# To see generated graph, type the following command in the appropriate directory:
# tensorboard --logdir=="desired_log_directory"
# Then, visit the provided link. Ensure that tensorflow, tensorboard, and other tensorflow related packages have the
# same version of 1.14.0
File = tf.summary.FileWriter("first_logs", sess.graph)

print(sess.run(c))
