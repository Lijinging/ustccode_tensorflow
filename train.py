# coding:utf-8

from model import *
import scipy.io
import scipy.misc
import numpy as np

n_epochs = 100
learning_rate = 0.0002
batch_size = 128
image_shape = [20, 20, 1]
dim_z = 100
dim_w1 = 1024
dim_w2 = 128
dim_w3 = 64
dim_channel = 1

visualize_size = 32
visualize_dim = visualize_size * visualize_size

train_raw = scipy.io.loadmat("data/dataTrain.mat")
train_data = train_raw["data"]
train_label = train_raw["label"]

dcgan_model = DCGAN(
    batch_size=batch_size,
    image_shape=image_shape,
    dim_z=dim_z,
    dim_w1=dim_w1,
    dim_w2=dim_w2,
    dim_w3=dim_w3
)

z_tf, y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan_model.build_model()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=10)

t_vars = tf.trainable_variables()
discrim_vars = [var for var in t_vars if 'discrim' in var.name]
gen_vars = [var for var in t_vars if 'gen' in var.name]

train_op_discrim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=discrim_vars)
train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_cost_tf, var_list=gen_vars)

z_tf_sample, y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

tf.global_variables_initializer().run()
Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim, dim_z))
Y_np_sample = OneHot(np.array([i for i in range(32) for j in range(visualize_size)]), visualize_size)

iterations = 0
k = 2
step = 10

for epoch in range(n_epochs):
    index = np.arange(len(train_label))
    np.random.shuffle(index)
    trX = train_data[index]
    trY = train_label[index]

    for start, end in zip(
        range(0, len(trY), batch_size),
        range(batch_size, len(trY), batch_size)
    ):
        Xs = trX[start:end].reshape([-1,20,20,1])/255.
        Ys = OneHot(trY[start:end],32)
        Zs = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)

        if np.mod(iterations, k) != 0:
            _, gen_loss_val = sess.run(
                [train_op_gen, g_cost_tf],
                feed_dict={
                    z_tf:Zs,
                    y_tf:Ys
                }
            )
            print("=========== updating D ==========")
            print("iteration:", iterations)
            print("gen loss:", gen_loss_val)
        else:
            _, discrim_loss_val = sess.run(
                [train_op_discrim, d_cost_tf],
                feed_dict={
                    z_tf: Zs,
                    y_tf: Ys,
                    image_tf: Xs
                }
            )
            print("=========== updating D ==========")
            print("iteration:", iterations)
            print("discrim loss:", discrim_loss_val)

        if iterations < 500:
            step = 10.
        else:
            step = 100.
        if np.mod(iterations, step) == 0:
            generated_samples = sess.run(
                image_tf_sample,
                feed_dict={
                    z_tf_sample: Z_np_sample,
                    y_tf_sample: Y_np_sample
                })
            generated_samples = (generated_samples + 1.) / 2.
            save_visualization(generated_samples, (visualize_size, visualize_size),
                               save_path='./vis/sample_%04d' % int(iterations / step) + '.jpg')

        iterations += 1