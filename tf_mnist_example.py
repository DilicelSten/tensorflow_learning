"""
created on:2018/7/25
author:DilicelSten
target:Learn how to use tensowflow(mnist data)
finished on:2018/7/25
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
# import matplotlib.pyplot as plt

# read dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, shape=[None, 784])
# 0 - 9 digits recognition
Y = tf.placeholder(tf.float32, shape=[None, nb_classes])

# Params
W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis(using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cost function
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(cost)  # learning rate is important

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))

# calculate acccuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# paramaters
training_epochs = 15
batch_size = 100

# build the graph
with tf.Session() as sess:
    # initial tensorflow variables
    sess.run(tf.global_variables_initializer())
    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print("Epoch: ", "%04d" % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost))

    # test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    # get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Labels:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))

    # plot it
    # plt.imshow(mnist.test.images[r:r + 1].
    #            reshape(28, 28), cmap='Greys', interpolation='nearest')
    # plt.show()
