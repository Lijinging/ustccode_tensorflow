# coding:utf-8

import scipy.io as sio
import tensorflow as tf
import numpy

train = sio.loadmat("data/dataTrain.mat")
test = sio.loadmat("data/dataTest.mat")
trainX = train['data'].astype('float32') / 255.
testX = test['data'].astype('float32') / 255.
trainY = numpy.zeros((trainX.shape[0], 32))
trainY[numpy.array(range(train['label'].shape[0])), train['label'][:, 0]] = 1.
testY = numpy.zeros((testX.shape[0], 32))
testY[numpy.array(range(test['label'].shape[0])), test['label'][:, 0]] = 1.

X = tf.placeholder(tf.float32, [None, 400])
Y_ = tf.placeholder(tf.float32, [None, 32])


def weight_variable(shape, name="weight"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')


trainIndex = 0


def next_batch(batch_size, data, label):
    global trainIndex
    i = trainIndex
    if trainIndex + batch_size > data.shape[0]:
        trainIndex = trainIndex + batch_size - data.shape[0]
        return numpy.row_stack((data[i:], data[:trainIndex])), numpy.row_stack((label[i:], label[:trainIndex]))
    else:
        trainIndex += batch_size
        return data[i:trainIndex], label[i:trainIndex]


x_image = tf.reshape(X, [-1, 20, 20, 1])

W_conv1 = weight_variable([5, 5, 1, 20],"W_conv1")
b_conv1 = bias_variable([20],"b_conv1")
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 20, 50],"W_conv2")
b_conv2 = bias_variable([50],"b_conv2")
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_ip1 = weight_variable([2 * 2 * 50, 500],"W_ip1")
b_ip1 = bias_variable([500],"b_ip1")
h_pool2_flat = tf.reshape(h_pool2, [-1, 2 * 2 * 50])
h_ip1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_ip1) + b_ip1)

keep_prob = tf.placeholder("float")
h_ip1_drop = tf.nn.dropout(h_ip1, keep_prob)

W_ip2 = weight_variable([500, 32],"W_ip2")
b_ip2 = bias_variable([32],"b_ip2")
h_ip2 = tf.matmul(h_ip1_drop, W_ip2) + b_ip2
y = tf.nn.softmax(h_ip2)

cross_entropy = -tf.reduce_sum(Y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


for i in range(3000):
    batch_xs, batch_ys = next_batch(40, trainX, trainY)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={X: testX, Y_: testY, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={X: batch_xs, Y_: batch_ys, keep_prob: 1.0})

saver.save(sess,r"E:\workspace\python\mis.ustc.checkcode\model\model.ckpt")