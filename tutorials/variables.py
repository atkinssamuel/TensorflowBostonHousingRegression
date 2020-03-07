# tf 1.14.0
import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

# For models with variables, you must initialize the variables with a special operation:
init = tf.global_variables_initializer()
sess = tf.Session()
File = tf.summary.FileWriter("variables_logs", sess.graph)


sess.run(init)
print(sess.run(linear_model, {x:[1, 2, 3, 4]}))



