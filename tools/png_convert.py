# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:26:46 2022

@author: Administrator
"""

import os
from tqdm import tqdm
import cv2
from skimage import io
path = r"D:\DDRNET\data\CamVid\test"
#西瓜6的代码
fileList = os.listdir(path)
for i in tqdm(fileList):
    image = io.imread(path+'\\'+i)
    # image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path+'\\'+i,image)
