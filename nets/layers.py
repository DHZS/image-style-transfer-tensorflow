# Author: An Jiaoyang
# 2018/9/7 23:32
# =============================
import tensorflow as tf
import tensorflow.contrib.slim as slim


weight_decay = 0.0005


def conv2d(filters, kernel_size, strides=(1, 1), padding='SAME',
           dilation_rate=(1, 1), activation=tf.nn.relu,
           kernel_initializer=tf.contrib.layers.xavier_initializer(),
           bias_initializer=tf.zeros_initializer(),
           kernel_regularizer=slim.l2_regularizer(weight_decay),
           bias_regularizer=slim.l2_regularizer(weight_decay),
           name=None, **kwargs):
    """conv2 layer"""
    return tf.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                            dilation_rate=dilation_rate, activation=activation,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer,
                            name=name, **kwargs)


def max_pool(pool_size, strides, padding='SAME', name=None):
    """max pooling layer"""
    return tf.layers.MaxPooling2D(pool_size, strides, padding, name=name)


def avg_pool(pool_size, strides, padding='SAME', name=None):
    """avg pooling layer"""
    return tf.layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)


def dense(units, activation=tf.nn.relu, use_bias=True,
          kernel_initializer=tf.contrib.layers.xavier_initializer(),
          bias_initializer=tf.zeros_initializer(),
          kernel_regularizer=slim.l2_regularizer(weight_decay),
          bias_regularizer=slim.l2_regularizer(weight_decay),
          name=None, **kwargs):
    """dense connected layer"""
    return tf.layers.Dense(units=units, activation=activation, use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           name=name, **kwargs)
