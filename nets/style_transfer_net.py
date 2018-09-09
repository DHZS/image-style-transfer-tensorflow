# Author: An Jiaoyang
# 2018/9/7 23:31
# =============================
"""Style Transfer Net
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nets.vgg19_net import VGG19
from tensorflow.contrib.eager.python import tfe


class StyleTransferNet:
    """Style Transfer Net"""
    def __init__(self, vgg19_npy_path, alpha=0.001, beta=1., optimizer='adam', learning_rate=1., momentum=0.9):
        self.vgg19 = VGG19(vgg19_npy_path, use_avg_pool=True)
        self.vgg19.init_all_variables()
        self.content = None
        self.style = None
        self.output = None
        optimizer_list = {'adam': tf.train.AdamOptimizer,
                          'sgd': tf.train.GradientDescentOptimizer,
                          'momentum': lambda lr: tf.train.MomentumOptimizer(lr, momentum)}
        self.optimizer = optimizer_list[optimizer](learning_rate)
        self.alpha = alpha
        self.beta = beta
        self.VGG_MEAN = [103.939, 116.779, 123.68]

    def set_images(self, content_path, style_path):
        self.content = self.read_image(content_path)
        self.style = self.read_image(style_path, output_size=self.content.get_shape().as_list()[1: 3])
        self.output = tfe.Variable(tf.random_normal(self.content.get_shape(), dtype=tf.float32))

    def _loss(self):
        content_layers = self.vgg19(self.content, return_layers=True)
        style_layer = self.vgg19(self.style, return_layers=True)
        output_layer = self.vgg19(self.output, return_layers=True)

        # content loss
        content_layers_name = ['conv4_2']
        losses = []
        for L in content_layers_name:
            losses.append(0.5 * tf.reduce_mean(tf.square(content_layers[L] - output_layer[L])))
        content_loss = tf.add_n(losses)

        # style loss
        style_layer_name = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        layer_weight = [1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]
        losses = []
        for L, w in zip(style_layer_name, layer_weight):
            style = style_layer[L]
            output = output_layer[L]
            shape = style.get_shape().as_list()
            new_shape = [shape[0], shape[1] * shape[2], shape[3]]
            style = tf.reshape(style, new_shape)
            output = tf.reshape(output, new_shape)
            # gram matrix
            g_style = tf.matmul(style, style, transpose_a=True)
            g_output = tf.matmul(output, output, transpose_a=True)
            loss = w * 1 / (4 * (new_shape[1] ** 2) * (new_shape[2] ** 2)) * tf.reduce_mean(tf.square(g_style - g_output))
            losses.append(loss)
        style_loss = tf.add_n(losses)

        # total loss
        total_loss = self.alpha * content_loss + self.beta * style_loss
        return total_loss

    def train(self):
        if self.content is None:
            raise ValueError('Input image is empty.')
        with tf.GradientTape() as g:
            g.watch(self.output)
            loss = self._loss()
        grad = g.gradient(loss, self.output)
        self.optimizer.apply_gradients([(grad, self.output)], global_step=tf.train.get_or_create_global_step())
        return loss

    def read_image(self, path, output_size=None):
        image = tf.image.decode_image(tf.read_file(path))
        if output_size is not None:
            image = tf.image.resize_images(image, output_size)
        image = tf.to_float(image)
        r, g, b = tf.split(image, 3, axis=2)
        image = tf.concat([b - self.VGG_MEAN[0], g - self.VGG_MEAN[1], r - self.VGG_MEAN[2]], axis=2)
        return tf.expand_dims(image, axis=0)

    def get_output_image(self):
        image = self.output[0]
        b, g, r = tf.split(image, 3, axis=2)
        image = tf.concat([r + self.VGG_MEAN[2], g + self.VGG_MEAN[1], b + self.VGG_MEAN[0]], axis=2)
        image = np.clip(image.numpy(), 0, 255).astype(np.uint8)
        return image

    def save_image(self, output_path):
        plt.imsave(output_path, self.get_output_image())




