#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import glob
import io
import os
import tensorflow as tf

from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('images_path', None, 'Path to images (directory).')
flags.DEFINE_string('output_path', None, 'Path to output tfrecord file.')
FLAGS = flags.FLAGS


# 首先要将图像编码为字符或数字特征，这需要调用类 tf.train.Feature。然后，
# 在调用 tf.train.Example 将特征写入协议缓冲区。最后，通过类 tf.python_io.
# TFRecordWriter 将数据写入到 .record 文件中。

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_path):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    label = int(image_path.split('_')[-1].split('.')[0])

    tf_example = tf.train.Example(
        features=tf.train.Feature(feature={
            'image/encoded': bytes_feature(encoded_jpg),
            'image/format': bytes_feature('jpg'.encode()),
            'image/class/label': int64_feature(label),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width)
        })
    )
    return tf_example


def generate_tfrecord(images_path, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    for image_file in glob.glob(images_path):
        tf_example = create_tf_example(image_file)
        writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
    images_path = os.path.join(FLAGS.images_path, '*.jpg')
    images_record_path = FLAGS.output_path
    generate_tfrecord(images_path, images_record_path)


# read tfrecord
slim = tf.contrib.slim


def get_reacord_dataset(record_path, reader=None, image_shape=[28, 28, 3], num_samples=50000, num_classes=10):
    """get a tensorflow record file."""
    if not reader:
        reader = tf.TFRecordReader
        keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64))
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=image_shape,
                                              #image_key='image/encoded',
                                              #format_key='image/format',
                                              channels=3),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    labels_to_names = None
    items_to_descriptions = {
        'image': 'An image with shape image_shape.',
        'label': 'A single integer between 0 and 9.'}
    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)

if __name__ == '__main__':
    tf.app.run()
