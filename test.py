# coding:utf-8
import tensorflow as tf
import scipy.io as scio
import numpy as np


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
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
save_path = r"C:\Users\lijin\PycharmProjects\ustccode_tensorflow\model\model.ckpt"
saver.restore(sess, save_path)

'''
设置输入数据
'''
LABELS = np.array(list("23456789ABCDEFGHJKLMNOPQRSTUVWXYZ"))
test_data_raw = scio.loadmat("data/dataTest.mat")
# 数据归一化
test_data = test_data_raw['data'][:10].astype('float32') / 255.0
# label:
y_ = test_data_raw['label'][:10, 0]

'''开始预测'''
pred = sess.run(y, feed_dict={x: test_data, keep_prob:1.0})

print(LABELS[np.argmax(pred, 1)])
print(LABELS[y_])



sess.close()
