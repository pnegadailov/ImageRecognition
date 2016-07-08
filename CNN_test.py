import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#Init variables

with tf.name_scope('Initialization'):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1,28,28,1])

#Creating network graph

#Layer 1
with tf.name_scope('CONV_Layer_1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

print("Layer 1: " + str(h_pool1.get_shape())) #Debug output to detect the shape

#Layer 2
with tf.name_scope('CONV_Layer_2'):
    W_conv2 = weight_variable([3, 3, 32, 128])
    b_conv2 = bias_variable([128])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

print("Layer 2: " + str(h_pool2.get_shape())) #Debug output to detect the shape

#Final layer
with tf.name_scope('Softmax_final_layer'):
    h_final = tf.reshape(h_pool2, [-1, 7*7*128])

    W_final = weight_variable([7*7*128, 10])
    b_final = bias_variable([10])

    #y_conv = tf.nn.relu(tf.matmul(h_final, W_final) + b_final)
    y_conv=tf.nn.softmax(tf.matmul(h_final, W_final) + b_final)

#Evaluation calculation
with tf.name_scope('Accuracy_calculation'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Training
with tf.name_scope('Training_process'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

tf.scalar_summary("accuracy", accuracy)
tf.merge_all_summaries()

summry_writer = tf.train.SummaryWriter('summary')

for i in range(1000): #Number of iterations. Should be much bigger
    batch = mnist.train.next_batch(50) #Batch size could be bigger (~100-500)

    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
        print("step %d, training accuracy %g"%(i, train_accuracy))

    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


summry_writer.add_summary(tf.merge_all_summaries())
summry_writer.add_graph(tf.get_default_graph())

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()
