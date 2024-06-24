# -*- coding: utf-8 -*-
"""
Created on Sat May 18 09:52:15 2019

@author: HP
"""


import cv2
import os
import numpy as np
from scipy.stats import mode
from math import fabs, sin, cos, radians

import os,sys
os.getcwd()
os.chdir("G:/Downloads/leaf detection/02_maize_dete/train_smp")
print(os.getcwd())
print (sys.version)




def read_img(source_imgpath):
    img = cv2.imdecode(np.fromfile(source_imgpath,dtype=np.uint8),cv2.IMREAD_COLOR)
    return img






def crop_img(img, new_x, new_y):
    res = cv2.resize(img, (new_x, new_y), interpolation=cv2.INTER_AREA) #见下
    cv2.imwrite("./" + str(new_x) + '_' + str(new_y) +
                '.jpg', res)



def rotate_img(img, rotate_angle, outputdir, filled_color=-1):

    if not os.path.exists(outputdir) and not os.path.isdir(outputdir):  
        
        os.mkdir(outputdir)  
    if filled_color == -1:
        filled_color = mode([img[0, 0], img[0, -1],
                             img[-1, 0], img[-1, -1]]).mode[0]
    if np.array(filled_color).shape[0] == 2:
        if isinstance(filled_color, int):
            filled_color = (filled_color, filled_color, filled_color)
    else:
        filled_color = tuple([int(i) for i in filled_color])          
    height, width = img.shape[:2]

    height_new = int(width * fabs(sin(radians(rotate_angle))) +
                     height * fabs(cos(radians(rotate_angle))))
    width_new = int(height * fabs(sin(radians(rotate_angle))) +
                    width * fabs(cos(radians(rotate_angle))))
    cols=width
    rows=height       
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_angle, 1)
    M[0, 2] += (width_new - width) / 2
    M[1, 2] += (height_new - height) / 2

    
    img_rotated = cv2.warpAffine(img, M, (width_new, height_new),
                                 borderValue=filled_color)
    cv2.imwrite(outputdir + os.path.sep + imgname + '_pic_' +str(rotate_angle) + '.jpg', img_rotated)




if __name__ == '__main__':
    
    for i in [2]:
     for j in range(218,219):
        imgname = str(i)+'_'+str(j)
        
        img = read_img(imgname+".jpg")
        print(img.shape)
        crop_img(img, 128, 128)
    
        curr_angle = 0
        while curr_angle < 360:
            rotate_img(img, curr_angle, "./bd1")
            curr_angle +=20