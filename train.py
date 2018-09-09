# Author: An Jiaoyang
# 2018/9/7 22:16
# =============================
import itertools
import tensorflow as tf
from nets.style_transfer_net import StyleTransferNet


conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
tf.enable_eager_execution(conf)


# arguments
tf.flags.DEFINE_string('vgg19_npy_path', './model/vgg19.npy', 'VGG19 model path')
tf.flags.DEFINE_string('content_path', None, 'content image path')
tf.flags.DEFINE_string('style_path', None, 'style image path')
tf.flags.DEFINE_string('output_path', None, 'output image path')
tf.flags.DEFINE_integer('save_interval', 10, 'save image interval')

tf.flags.DEFINE_float('alpha', 0.0005, 'content weight')
tf.flags.DEFINE_float('beta', 1., 'style weight')
tf.flags.DEFINE_float('learning_rate', 5., 'learning rate')
tf.flags.DEFINE_string('optimizer', 'adam', 'optimizer to use, choose in {adam, sgd, momentum}')

FLAGS = tf.flags.FLAGS


def main():
    st = StyleTransferNet(vgg19_npy_path=FLAGS.vgg19_npy_path,
                          alpha=FLAGS.alpha,
                          beta=FLAGS.beta,
                          learning_rate=FLAGS.learning_rate,
                          optimizer=FLAGS.optimizer)
    st.set_images(FLAGS.content_path, FLAGS.style_path)

    for i in itertools.count(start=1):
        loss = st.train()
        print('epochs: {}, loss: {:.4f}\r'.format(i, float(loss)))

        if i % FLAGS.save_interval == 0:
            st.save_image(FLAGS.output_path)
            print('image saved')


if __name__ == '__main__':
    main()

