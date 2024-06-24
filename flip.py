# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 16:27:57 2020

@author: hp
"""
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

os.getcwd()
os.chdir("G:/Downloads/plant_disease_detect/02_maize_dete/train_smp")
print(os.getcwd())
print (sys.version)



class RandomHorizontalFlip(object):


    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
            if random.random() < self.p:
                img = img[:, ::-1, :]

            return img


class HorizontalFlip(object):


    def __init__(self):
        pass

    def __call__(self, img):

        img = img[:, ::-1, :]
        return img
    
    
def read_img(source_imgpath):
    img = cv2.imdecode(np.fromfile(source_imgpath,dtype=np.uint8),cv2.IMREAD_COLOR)
    return img



for i in range(1,5):
     k=809
     for j in range(450,550):
        imgname = str(i)+'_'+str(j)
        
        img = read_img(imgname+".jpg")
        img_name= str(i)+'_'+str(k)
        k+=1
        hor_flip = RandomHorizontalFlip(1)  

        img = hor_flip(img)
        outputdir="./flip"
        cv2.imwrite(outputdir + os.path.sep + img_name + '.jpg', img)
          


    
    
    