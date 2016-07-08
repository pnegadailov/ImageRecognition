import tensorflow as tf

# Build a graph.
import tensorflow as tf

with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')


# Launch the graph in a session.
sess = tf.Session()
sess.run(tf.initialize_all_variables())

summry_writer = tf.train.SummaryWriter('summary')
summry_writer.add_graph(tf.get_default_graph())

# Evaluate the tensor `c`.


sess.close()