"""
created on:2018/7/25
author:DilicelSten
target:Learn how to use tensowflow(linear regression)
finished on:2018/7/25
"""
import tensorflow as tf


# X and Y data
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

# Now we can use X and Y in place of x_train and y_train
# placeholders for a tensor that will be always fed using feed_dict
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# params
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# our hypothesis y = XW + b
# hypothesis = x_train * W + b
hypothesis = X * W + b

# cost function
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# launch the graph in a session
sess = tf.Session()

# initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# fit the line
for step in range(2001):
    # sess.run(train)
    # if step % 20 == 0:
    #     print(step, sess.run(cost), sess.run(W), sess.run(b))
    cost_val, W_val, b_val, _ = sess.run(
        [cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]}
    )
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)