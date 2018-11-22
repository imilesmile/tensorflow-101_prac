#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf

device_type = "/gpu:2"


# define function
def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.transpose(np.array([img, img, img]), (2, 0, 1))
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


def print_np(x, s):
    print ("Type of '%s' is %s" % (s, type(x)))
    print ("Shape of '%s' is %s" % (s, x.shape,))


# load data
cwd = os.getcwd()
path = cwd + "/../../retina_dataset/dataset"
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)
imgs = []
labels = []
# LOAD ALL IMAGES
for i, category in enumerate(categories):
    for f in os.listdir(path + "/" + category):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_exts:
            continue
        fullpath = os.path.join(path + "/" + category, f)
        img = scipy.misc.imresize(imread(fullpath), [224, 224, 3])
        imgs.append(img)  # NORMALIZE IMAGE
        label_curr = np.zeros((ncategories))
        label_curr[i] = 1
        labels.append(label_curr)
print ("Num imgs: %d" % (len(imgs)))
print ("Num labels: %d" % (len(labels)))

# divide data
ndata = len(imgs)
ntrain = int(ndata * 0.7)  # WILL USE 70% FOR TRAINING
ntest = ndata - ntrain
randidx = np.random.permutation(ndata)
trainidx = randidx[:ntrain]
testidx = randidx[ntrain + 1:]
train_imgs = [imgs[idx] for idx in trainidx]
train_labels = [labels[idx] for idx in trainidx]
test_imgs = [imgs[idx] for idx in testidx]
test_labels = [labels[idx] for idx in testidx]
# TENSORIZE DATA
train_imgs_tensor = np.stack(train_imgs, axis=0)
train_labels_tensor = np.stack(train_labels, axis=0)
test_imgs_tensor = np.stack(test_imgs, axis=0)
test_labels_tensor = np.stack(test_labels, axis=0)
print ("Num train_imgs: %d" % (len(train_imgs)))
print ("Num test_imgs: %d" % (len(test_imgs)))
print_np(train_imgs_tensor, "train_imgs_tensor")
print_np(train_labels_tensor, "train_labels_tensor")
print_np(test_imgs_tensor, "test_imgs_tensor")
print_np(test_labels_tensor, "test_labels_tensor")
# REMOVE LISTS FROM MEMORY
del train_imgs, train_labels, test_imgs, test_labels

#plot data
randidx = np.sort(np.random.randint(ntrain, size=2))-1
for i, j in enumerate(randidx):
    curr_img = train_imgs_tensor[j, :, :, :]
    curr_label = np.argmax(train_labels_tensor[j, :])
    plt.figure(i)
    plt.imshow(curr_img)
    plt.title("TRAIN [" + str(curr_label) + ", " + categories[curr_label] + "]")
    plt.show()
randidx = np.sort(np.random.randint(ntest, size=2))-1
for i, j in enumerate(randidx):
    curr_img = test_imgs_tensor[j, :, :, :]
    curr_label = np.argmax(test_labels_tensor[j, :])
    plt.figure(i)
    plt.imshow(curr_img)
    plt.title("TEST [" + str(curr_label) + ", " + categories[curr_label] + "]")
    plt.show()

#help function use vgg19
with tf.device(device_type):
    # FUNCTIONS FOR USING VGG19
    def conv_layer(input, weights, bias):
        conv = tf.nn.conv2d(input, tf.constant(weights), strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.bias_add(conv, bias)
    def pool_layer(input):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    def preprocess(image, mean_pixel):
        return image - mean_pixel
    print "Functions for VGG ready"

    # Define network
    def vggnet(data_path, input_image):
        layers = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        )
        data = scipy.io.loadmat(data_path) # Read .mat file via scipy.io.loadmat
        mean = data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))
        weights = data['layers'][0]
        net = {}
        current = preprocess(input_image, mean_pixel)
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0 , 2, 3))
                bias = bias.reshape(-1)
                current = conv_layer(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current)
            elif kind == 'pool':
                current = pool_layer(current)
            net[name] = current
        assert len(net) == len(layers)
        return net, mean_pixel, layers
print ("Network for VGG ready")

