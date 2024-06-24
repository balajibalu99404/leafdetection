# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 15:30:48 2020

@author: hp
"""
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import mode

os.getcwd()
os.chdir("G:/Downloads/plant_disease_detect/02_maize_dete/train_smp")
print(os.getcwd())
print (sys.version)


class HorizontalFlip(object):

  

    def __init__(self):
        pass

    def __call__(self, img):
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))

        img = img[:, ::-1, :]


        return img

class RandomShear(object):  
    def __init__(self, shear_factor = 0.2):
        self.shear_factor = shear_factor
        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"   
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)       
        shear_factor = random.uniform(*self.shear_factor)
        
    def __call__(self, img,filled_color=-1):
        
        shear_factor = random.uniform(*self.shear_factor)
        
        if filled_color == -1:
          filled_color = mode([img[0, 0], img[0, -1],
                             img[-1, 0], img[-1, -1]]).mode[0]
        if np.array(filled_color).shape[0] == 2:
          if isinstance(filled_color, int):
            filled_color = (filled_color, filled_color, filled_color)
        else:
          filled_color = tuple([int(i) for i in filled_color])
        w,h = img.shape[1], img.shape[0]
    
        if shear_factor < 0:
            img= HorizontalFlip()(img)
    
        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
    
        nW =  img.shape[1] + abs(shear_factor*img.shape[0])
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]),  borderValue=filled_color)
        if shear_factor < 0:
            img= HorizontalFlip()(img)
        img = cv2.resize(img, (w,h))
        return img

def read_img(source_imgpath):
    img = cv2.imdecode(np.fromfile(source_imgpath,dtype=np.uint8),cv2.IMREAD_COLOR)
    return img



for i in range(1,5):
     k=601
     for j in range(200,300):
        imgname = str(i)+'_'+str(j)
        
        img = read_img(imgname+".jpg")
        shear = RandomShear(0.7)
        img_name= str(i)+'_'+str(k)
        k+=1
        img = shear(img)
        outputdir="./shear"
        cv2.imwrite(outputdir + os.path.sep + img_name + '.jpg', img)
          


          