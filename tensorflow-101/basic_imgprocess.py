#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 22:15:37 2018

@author: miao
"""
# Import packs
import numpy as np
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform

# import tensorflow as tf
print ("Packs loaded")


def print_typeshape(img):
    print("type is %s" % (type(img)))
    print("shape is %s " % (img.shape,))


# load
cat = imread("/Users/miao/Documents/tensorflow/Tensorflow-101-master/notebooks/images/cats/images (1).jpeg")
print_typeshape(cat)
# plot
# plt.figure(0)
# plt.imshow(cat)
# plt.title("Original Image with imread")
# plt.show()

# load
cat2 = imread("/Users/miao/Documents/tensorflow/Tensorflow-101-master/notebooks/images/cats/images (1).jpeg").astype(
    np.float)
print_typeshape(cat2)
# plot
# plt.figure(0)
# plt.imshow(cat2)
# plt.title("Original Image with imread")
# plt.show()

# load
cat3 = imread("/Users/miao/Documents/tensorflow/Tensorflow-101-master/notebooks/images/cats/images (1).jpeg").astype(
    np.float)
print_typeshape(cat3)
# plot
# plt.figure(0)
# plt.imshow(cat3/255.)
# plt.title("Original Image with imread")
# plt.show()

# Resize
catsmall = imresize(cat, [100, 100, 3])
print_typeshape(catsmall)


# Plot
# plt.figure(1)
# plt.imshow(catsmall)
# plt.title("Resized Image")
# plt.show()

# Grayscale
def rgb2gray(rgb):
    if len(rgb.shape) is 3:
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    else:
        print ("Current Image if GRAY!")
        return rgb


catsmallgray = rgb2gray(catsmall)

print ("size of catsmallgray is %s" % (catsmallgray.shape,))
print ("type of catsmallgray is", type(catsmallgray))

# plt.figure(2)
# plt.imshow(catsmallgray, cmap=plt.get_cmap("gray"))
# plt.title("[imshow] Gray Image")
# plt.colorbar()
# plt.show()

catrowvec = np.reshape(catsmallgray, (1, -1))
print ("size of catrowvec is %s" % (catrowvec.shape,))
print ("type of catrowvec is", type(catrowvec))

catmatrix = np.reshape(catsmallgray, (100, 100))
print ("size of catmatrix is %s" % (catmatrix.shape,))
print ("type of catmatrix is", type(catmatrix))

#load from folder
path = "/Users/miao/Documents/tensorflow/Tensorflow-101-master/notebooks/images/cats"
valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
print ("%d files in %s" % (len(os.listdir(path)), path))

imgs =[]
names = []
for f in os.listdir(path):
    #for all files
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_exts:
        continue
    fullpath = os.path.join(path,f)
    imgs.append(imread(fullpath))
    names.append(os.path.splitext(f)[0] + os.path.splitext(f)[1])
print ("%d images loaded" % (len(imgs)))

#check
nimgs = len(imgs)
randidx = np.sort(np.random.randint(nimgs, size=3))
print ("Type of 'imgs': ", type(imgs))
print ("Length of 'imgs': ", len(imgs))
for curr_img, curr_name, i \
    in zip([imgs[j] for j in randidx]
           , [names[j] for j in randidx]
           , range(len(randidx))):
    print ("[%d] Type of 'curr_img': %s" % (i, type(curr_img)))
    print ("    Name is: %s" % (curr_name))
    print ("    Size of 'curr_img': %s" % (curr_img.shape,))

for curr_img, curr_name, i \
    in zip([imgs[j] for j in randidx]
           , [names[j] for j in randidx], range(len(randidx))):
    plt.figure(i)
    plt.imshow(curr_img)
    plt.title("[" + str(i) + "] ")
    plt.show()







