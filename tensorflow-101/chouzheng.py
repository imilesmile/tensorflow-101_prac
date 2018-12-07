#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from PIL import Image
from ctypes import *

video_path = ""
lib = cdll.LoadLibrary('./libfastsnapshot.so') # 加载解帧库
lib.process(video_path, '/tmp/frame', 256, 256) # 提取视频关键帧输出到指定目录，并指定输出宽高
frames = [Image.frombuffer('RGB', (256, 256), open(frame, mode='rb').read(), 'raw', 'RGB', 0, 1) for frame in
          '/tmp/frame'] # 将提取的关键帧加载到Image列表
