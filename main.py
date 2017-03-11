# coding:utf-8
import tensorflow as tf
import scipy.io as scio
import numpy as np

'''
设置输入数据
'''
train_data_raw = scio.loadmat("data/dataTrain.mat")
test_data_raw = scio.loadmat("data/dataTest.mat")
# 数据归一化
train_data = train_data_raw['data'].astype('float32') / 255.0
test_data = test_data_raw['data'].astype('float32') / 255.0
# label:
train_label = np.zeros((train_data.shape[0], 32))
test_label = np.zeros((test_data.shape[0], 32))
train_label[np.array(range(train_data_raw['label'].shape[0])), train_data_raw['label'][:, 0]] = 1
test_label[np.array(range(test_data_raw['label'].shape[0])), test_data_raw['label'][:, 0]] = 1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 400])
y_ = tf.placeholder(tf.float32, shape=[None, 32])

x_image = tf.reshape(x, [-1, 20, 20, 1])

W_conv1 = weight_variable([5, 5, 1, 20])  # 第一层卷积层
b_conv1 = bias_variable([20])  # 第一层卷积层的偏置量
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 第一次池化层

W_conv2 = weight_variable([5, 5, 20, 50])  # 第二次卷积层
b_conv2 = bias_variable([50])  # 第二层卷积层的偏置量
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # 第二曾池化层

W_fc1 = weight_variable([2 * 2 * 50, 500])  # 全连接层
b_fc1 = bias_variable([500])  # 偏置量
h_pool2_flat = tf.reshape(h_pool2, [-1, 2 * 2 * 50])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([500, 32])
b_fc2 = bias_variable([32])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())


def next_batch(data, label, begin, length):
    if begin >= data.shape[0]:
        begin = begin % data.shape[0]
    if begin + length > data.shape[0]:
        add = next_batch(data, label, 0, length - (data.shape[0] - begin))
        add[0] = np.row_stack((data[begin:], add[0]))
        add[1] = np.row_stack((label[begin:], add[1]))
    else:
        add = [data[begin:begin + length], label[begin:begin + length]]
    return add


for i in range(3000):
    size = 100
    # batch = mnist.train.next_batch(100)
    batch = next_batch(train_data, train_label, i * size, size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print(i, end=":")
        print("test accuracy %g" % accuracy.eval(feed_dict={
            x: test_data, y_: test_label, keep_prob: 1.0}))
