#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

print ("Packages loaded.")
clsf_path = "/Users/miao/PycharmProjects/tensorflow_prac/tensorflow-nn-practise/data/haarcascade_frontalface_default.xml"
print (clsf_path)
face_cascade = cv2.CascadeClassifier(clsf_path)
print ("face_cascade is %s" % (face_cascade))

imgpath = "/Users/miao/Documents/tensorflow/Tensorflow-101-master/notebooks/images/Lenna.png"
img_bgr = cv2.imread(imgpath)

# convert to rgb
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# detect face
faces = face_cascade.detectMultiScale(img_gray)
print ("%d faces deteced. " % (len(faces)))

# plot detected faces
plt.figure(0)
plt.imshow(img_gray, cmap=plt.get_cmap("gray"))
ca = plt.gca()
for face in faces:
    ca.add_patch(Rectangle((face[0], face[1]), face[2], face[3]
                           , fill=None, alpha=1, edgecolor='blue'))
plt.title("Face detection with Viola-Jones")
plt.show()

#detect faces in the folders
# path = cwd + "/../../img_dataset/celebs/Arnold_Schwarzenegger"
# flist = os.listdir(path)
# valid_exts = [".jpg",".gif",".png",".tga", ".jpeg"]
# for f in flist:
#     if os.path.splitext(f)[1].lower() not in valid_exts:
#         continue
#     fullpath = os.path.join(path, f)
#     img_bgr = cv2.imread(fullpath)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
#     faces = face_cascade.detectMultiScale(img_gray)
#     # PLOT
#     plt.imshow(img_gray, cmap=plt.get_cmap("gray"))
#     ca = plt.gca()
#     for face in faces:
#         ca.add_patch(Rectangle((face[0], face[1]), face[2], face[3]
#                                , fill=None, alpha=1, edgecolor='red'))
#     plt.title("Face detection with Viola-Jones")
#     plt.show()