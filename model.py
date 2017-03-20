# coding:utf-8

import tensorflow as tf
import numpy as np
import scipy.misc


def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh


def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0, 1, 2])
        std = tf.reduce_mean(tf.square(X - mean), [0, 1, 2])
        X = (X - mean) / tf.sqrt(std + eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, 1, 1, -1])
            b = tf.reshape(b, [1, 1, 1, -1])
            X = X * g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X - mean), 0)
        X = (X - mean) / tf.sqrt(std + eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1, -1])
            b = tf.reshape(b, [1, -1])
            X = X * g + b

    else:
        raise NotImplementedError

    return X


def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

    for n, x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = x

    scipy.misc.imsave(save_path, img)


def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o, labels=t))


class DCGAN():
    def __init__(
            self,
            batch_size=100,
            image_shape=[20, 20, 1],
            dim_z=100,
            dim_y=32,
            dim_w1=1024,
            dim_w2=128,
            dim_w3=64,
            dim_channel=1
    ):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_w1 = dim_w1
        self.dim_w2 = dim_w2
        self.dim_w3 = dim_w3
        self.dim_channel = dim_channel

        self.gen_w1 = tf.Variable(tf.random_normal([dim_z + dim_y, dim_w1], stddev=0.02), name="gen_w1")
        self.gen_w2 = tf.Variable(tf.random_normal([dim_w1 + dim_y, dim_w2 * 5 * 5], stddev=0.02), name="gen_w2")
        self.gen_w3 = tf.Variable(tf.random_normal([5, 5, dim_w3, dim_w2 + dim_y], stddev=0.02), name="gen_w3")
        self.gen_w4 = tf.Variable(tf.random_normal([5, 5, dim_channel, dim_w3 + dim_y], stddev=0.02), name="gen_w4")

        self.discrim_w1 = tf.Variable(tf.random_normal([5, 5, dim_channel + dim_y, dim_w3], stddev=0.02),
                                      name="discrim_w1")
        self.discrim_w2 = tf.Variable(tf.random_normal([5, 5, dim_w3, dim_w2], stddev=0.02), name="discrim_w2")
        self.discrim_w3 = tf.Variable(tf.random_normal([dim_w2 * 5 * 5, dim_w1], stddev=0.02), name='discrim_w3')
        self.discrim_w4 = tf.Variable(tf.random_normal([dim_w1, 1], stddev=0.02), name='discrim_w4')

    def build_model(self):
        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])

        image_real = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        image_gen = tf.nn.sigmoid(self.generate(Z, Y, self.batch_size))
        raw_real = self.discriminate(image_real, Y)
        p_real = tf.nn.sigmoid(raw_real)
        raw_gen = self.discriminate(image_gen, Y)
        p_gen = tf.nn.sigmoid(raw_gen)
        discrim_cost_real = bce(raw_real, tf.ones_like(raw_real))
        discrim_cost_gen = bce(raw_gen, tf.zeros_like(raw_gen))
        discrim_cost = discrim_cost_gen + discrim_cost_real

        gen_cost = bce(raw_gen, tf.ones_like(raw_gen))

        return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen

    def discriminate(self, image, Y):
        yb = tf.reshape(Y, tf.stack([self.batch_size, 1, 1, self.dim_y]))
        x = tf.concat(axis=3, values=[image, yb * tf.ones([self.batch_size, 20, 20, self.dim_y])])
        h1 = lrelu(tf.nn.conv2d(x, self.discrim_w1, strides=[1, 2, 2, 1], padding='SAME'))
        h2 = lrelu(batchnormalize(tf.nn.conv2d(h1, self.discrim_w2, strides=[1, 2, 2, 1], padding='SAME')))
        h3 = tf.reshape(h2, [self.batch_size, -1])
        h4 = lrelu(batchnormalize(tf.matmul(h3, self.discrim_w3)))
        return h4

    def generate(self, Z, Y, batchsize):
        yb = tf.reshape(Y, [batchsize, 1, 1, self.dim_y])
        Z = tf.concat(axis=1, values=[Z, Y])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_w1)))
        h2 = tf.concat(axis=1, values=[h1, Y])
        h3 = tf.nn.relu(batchnormalize(tf.matmul(h2, self.gen_w2)))
        h4 = tf.reshape(h3, [batchsize, 5, 5, self.dim_w2])
        h5 = tf.concat(axis=3, values=[h4, yb * tf.ones([batchsize, 5, 5, self.dim_y])])
        h6 = tf.nn.conv2d_transpose(h5, self.gen_w3, output_shape=[batchsize, 10, 10, self.dim_w3],
                                    strides=[1, 2, 2, 1])
        h7 = tf.nn.relu(batchnormalize(h6))
        h8 = tf.concat(axis=3, values=[h7, yb * tf.ones([batchsize, 10, 10, self.dim_y])])
        h9 = tf.nn.conv2d_transpose(h8, self.gen_w4, output_shape=[batchsize, 20, 20, self.dim_channel],
                                    strides=[1, 2, 2, 1])
        return h9

    def samples_generator(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])

        X = tf.nn.sigmoid(self.generate(Z, Y, batch_size))

        return Z, Y, X
