#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:41:09 2018

@author: miao
"""
import numpy as np
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
print("package loaded")
cwd = os.getcwd()
print ("Current folder is %s" % (cwd) )

#training set folder
paths={"","","",""}

#reshape size
image_size=[64,64]

#grey scale
use_gray=1

#save_name
data_name="./data/custom_data"

for i,path in enumerate(paths):
    print("[%d/%d] %s/%s"%(i,len(paths),cwd,path)) 

#rgb grey function
def rgb2gray(rgb):
    if len(rgb.shape) is 3:
        return np.dot(rgb[...,:3],[0.299, 0.587, 0.114])
    else:
        return rgb

#load images
nclass = len(paths)
valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
imgcnt =0

for i,relpath in zip(range(nclass), paths):
    path = cwd +"/"+relpath
    flist = os.listdir(path)
    for f in flist:
        if os.path.splittext(f)[1].lower() not in valid_exts:
            continue
        fullpath = os.path.join(path,f)
        currimg = imread(fullpath)
        #convert to gray
        if use_gray:
            grayimg = rgb2gray(currimg)
        else:
            grayimg = currimg
        
        #reshape
        graysmall = imresize(grayimg, [imgsize[0], imgsize[1]])/255
        grayvec = np.reshape(graysmall,[-1,1])
        
        #save
        curr_label = np.eye(nclass, nclass)[i:i+1,:]
        if imgcnt is 0:
            totalimg = grayvec
            totallabel = curr_label
            
            
def print_shape(string, x):
    print ("shape of '%s' is %s" % (string, x.shape,))
    
#DIVIDE TOTAL DATA INTO TRAINING AND TEST SET
randidx =np.random.randint(imgcnt, size=imgcnt)
randidx    = np.random.randint(imgcnt, size=imgcnt)
trainidx   = randidx[0:int(3*imgcnt/5)]
testidx    = randidx[int(3*imgcnt/5):imgcnt]
trainimg   = totalimg[trainidx, :]
trainlabel = totallabel[trainidx, :]
testimg    = totalimg[testidx, :]
testlabel  = totallabel[testidx, :]
print_shape("trainimg", trainimg)
print_shape("trainlabel", trainlabel)
print_shape("testimg", testimg)
print_shape("testlabel", testlabel)
        
        
#save to npz
save_path = cwd +"/data/" + data_name + ".npz"
np.savez(savepath, trainimg=trainimg, trainlabel=trainlabel
         , testimg=testimg, testlabel=testlabel, imgsize=imgsize, use_gray=use_gray)
print ("save to %s" %(save_path))


#load to check
#load them
cwd = os.getcwd()
loadpath = cwd + "/data/" + data_name + ".npz"
l = np.load(loadpath)












        
        
        