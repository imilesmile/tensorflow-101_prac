#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.slim import nets

slim = tf.contrib.slim


def predict(self, preprocessed_inputs):
    """
    Predict prediction tensors from inputs tensor.

        Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
    """
    net, endpoints = nets.resnet_v1.resnet_v1_50(preprocessed_inputs, num_classes=None, is_training=self._is_training)
    net = tf.squeeze(net, axis=[1, 2])
    net = slim.fully_connected(net, num_outputs=self.num_classes, activation_fn=None, scope='Predict')
    prediction_dict = {'logits': net}
    return prediction_dict
