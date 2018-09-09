# Author: An Jiaoyang
# 2018/9/7 16:00
# =============================
import numpy as np
import tensorflow as tf
from collections import namedtuple
from nets import layers


VGGParams = namedtuple(
    'VGGParams',
    ['input_shape'
     ])


class VGG19(tf.keras.Model):
    """VGG19 net"""
    default_params = VGGParams(
        input_shape=[224, 224, 3]
    )

    def __init__(self, vgg19_npy_path, use_avg_pool=False, params=None, name='vgg19'):
        super(VGG19, self).__init__(name=name)
        self.vgg19_npy_path = vgg19_npy_path
        self.params = VGG19.default_params if params is None else params

        # build layer
        pool = layers.avg_pool if use_avg_pool else layers.max_pool
        self.conv1_1 = layers.conv2d(64, [3, 3], name="conv1_1")
        self.conv1_2 = layers.conv2d(64, [3, 3], name="conv1_2")
        self.pool1 = pool(2, 2, name='pool1')

        self.conv2_1 = layers.conv2d(128, [3, 3], name="conv2_1")
        self.conv2_2 = layers.conv2d(128, [3, 3], name="conv2_2")
        self.pool2 = pool(2, 2, name='pool2')

        self.conv3_1 = layers.conv2d(256, [3, 3], name="conv3_1")
        self.conv3_2 = layers.conv2d(256, [3, 3], name="conv3_2")
        self.conv3_3 = layers.conv2d(256, [3, 3], name="conv3_3")
        self.conv3_4 = layers.conv2d(256, [3, 3], name="conv3_4")
        self.pool3 = pool(2, 2, name='pool3')

        self.conv4_1 = layers.conv2d(512, [3, 3], name="conv4_1")
        self.conv4_2 = layers.conv2d(512, [3, 3], name="conv4_2")
        self.conv4_3 = layers.conv2d(512, [3, 3], name="conv4_3")
        self.conv4_4 = layers.conv2d(512, [3, 3], name="conv4_4")
        self.pool4 = pool(2, 2, name='pool4')

        self.conv5_1 = layers.conv2d(512, [3, 3], name="conv5_1")
        self.conv5_2 = layers.conv2d(512, [3, 3], name="conv5_2")
        self.conv5_3 = layers.conv2d(512, [3, 3], name="conv5_3")
        self.conv5_4 = layers.conv2d(512, [3, 3], name="conv5_4")
        self.pool5 = pool(2, 2, name='pool5')

        self.fc6 = layers.dense(4096, name='fc6')
        self.fc7 = layers.dense(4096, name='fc7')
        self.fc8 = layers.dense(1000, activation=None, name='fc8')

    def call(self, inputs, return_layers=False, **kwargs):
        layer_list = dict()

        net = self.conv1_1(inputs)
        layer_list['conv1_1'] = net
        net = self.conv1_2(net)
        layer_list['conv1_2'] = net
        net = self.pool1(net)

        net = self.conv2_1(net)
        layer_list['conv2_1'] = net
        net = self.conv2_2(net)
        layer_list['conv2_2'] = net
        net = self.pool2(net)

        net = self.conv3_1(net)
        layer_list['conv3_1'] = net
        net = self.conv3_2(net)
        layer_list['conv3_2'] = net
        net = self.conv3_3(net)
        layer_list['conv3_3'] = net
        net = self.conv3_4(net)
        layer_list['conv3_4'] = net
        net = self.pool3(net)

        net = self.conv4_1(net)
        layer_list['conv4_1'] = net
        net = self.conv4_2(net)
        layer_list['conv4_2'] = net
        net = self.conv4_3(net)
        layer_list['conv4_3'] = net
        net = self.conv4_4(net)
        layer_list['conv4_4'] = net
        net = self.pool4(net)

        net = self.conv5_1(net)
        layer_list['conv5_1'] = net
        net = self.conv5_2(net)
        layer_list['conv5_2'] = net
        net = self.conv5_3(net)
        layer_list['conv5_3'] = net
        net = self.conv5_4(net)
        layer_list['conv5_4'] = net
        net = self.pool5(net)

        if return_layers is True:
            return layer_list

        batch_size = net.get_shape().as_list()[0]
        net = tf.reshape(net, [batch_size, -1])  # [-1, 25088]
        net = self.fc6(net)
        net = self.fc7(net)
        logits = self.fc8(net)
        prob = tf.nn.softmax(logits)
        return prob

    def init_all_variables(self):
        # load model weights
        weights = np.load(self.vgg19_npy_path, encoding='latin1').item()
        empty_image = tf.zeros([1] + self.params.input_shape, dtype=tf.float32)
        self.call(empty_image)

        # restore weights
        for layer in self.layers:
            if layer.name in weights:
                layer.set_weights(weights[layer.name])


