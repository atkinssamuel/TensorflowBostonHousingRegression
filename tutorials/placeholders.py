# tf 1.14.0
import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

sess = tf.Session()
File = tf.summary.FileWriter("placeholders_logs", sess.graph)

print(sess.run(adder_node, {a: [1, 3], b:[2, 4]}))

