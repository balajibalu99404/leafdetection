# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 22:23:24 2019

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 07:11:11 2018

@author: HP
"""


import cv2
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras.layers import Activation, Dense
from matplotlib import pyplot as plt
from skimage import io,data
import time
from keras import layers

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

import os,sys
os.getcwd()
os.chdir("C:/Users/BALAJI BALU/Downloads/mini project/leaf detection")


print(os.getcwd())
print (sys.version)



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"










DATASET_DIR = 'train_smp/'


# ===============================================================================

def read_image(size = None):
     data_x, data_y = [], []
     for i in range(1,5): 
         for j in range(101,909):
             try:
                 im = cv2.imread('train_smp/%s.jpg' % (str(i)+'_'+str(j)))
                 if size is None:
                    size = (224, 224)
                 #im = resize_without_deformation(im, size)
                 data_x.append(np.asarray(im, dtype = np.int8))
                 data_y.append(str(i))
             except IOError as e:
                print(e)
             except:
                print('done!')
     return data_x, data_y
 

from keras import backend as K





from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import  Dense, Dropout, Flatten,LeakyReLU
from keras.optimizers import SGD

IMAGE_SIZE = 224
raw_images, raw_labels = read_image(size=(IMAGE_SIZE, IMAGE_SIZE))
raw_images, raw_labels = np.asarray(raw_images, dtype = np.float32), np.asarray(raw_labels, dtype = np.int32)



from keras.utils import np_utils
ont_hot_labels = np_utils.to_categorical(raw_labels)
##ont_hot_labels = np.delete(ont_hot_labels, 0, axis=1)
ont_hot_labels = ont_hot_labels[:,list(range(1,5))]
#ont_hot_labels = ont_hot_labels[:,[18,26,43,44,45]]


from sklearn.model_selection import  train_test_split
train_input, valid_input, train_output, valid_output =train_test_split(raw_images, 
                  ont_hot_labels,
                  test_size = 0.3,
                  shuffle=True
#                  ,random_state=0                  
                  )
raw_images/=255.0
train_input /= 255.0
valid_input /= 255.0    







#--------VGG19-----------------
#coding=utf-8  
from keras.applications.vgg19 import VGG19
from keras.layers import Dense,Flatten,GlobalAveragePooling2D
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate  
from keras.optimizers import SGD
from keras.models import Model  
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D  
import numpy as np  

from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras.layers import *       
def swish(x):
    return (K.sigmoid(x) * x)
get_custom_objects().update({'swish': Activation(swish)})


def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None   
#    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,name=conv_name)(x)
    x = (swish)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)  
    return x  


def Inception(x,nb_filter):  
    branch1x1 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)  
  
    branch3x3 = Conv2d_BN(x,nb_filter,(3,3), padding='same',strides=(1,1),name=None)  
    branch3x3 = Conv2d_BN(branch3x3,nb_filter,(3,3), padding='same',strides=(1,1),name=None)  
  
    branch5x5 = Conv2d_BN(x,nb_filter,(5,5), padding='same',strides=(1,1),name=None)  
    branch5x5 = Conv2d_BN(branch5x5,nb_filter,(5,5), padding='same',strides=(1,1),name=None)
    branch5x5 = Conv2d_BN(branch5x5,nb_filter,(5,5), padding='same',strides=(1,1),name=None)
  
    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)  
    branchpool = Conv2d_BN(branchpool,nb_filter,(1,1),padding='same',strides=(1,1),name=None)  
  
    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)    
    return x  


base_model = VGG19(weights='G:/Downloads/leaf detection/KerasWeights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
                   , include_top=False)
for layer in base_model.layers:
    layer.trainable = False
base_model.layers.pop()  
base_model.layers.pop()
x = base_model.output
x = Conv2d_BN(x,512,(3,3),strides=(1,1),padding='same')  
x = MaxPooling2D(pool_size=(2,2),strides=(1,1),padding='same')(x) 
x = Inception(x,512)  
x = Inception(x,512)  
x = GlobalAveragePooling2D()(x)
output = Dense(len(ont_hot_labels[0]), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)









learning_rate = 0.0001  
#learning_rate = 0.005
decay = 1e-6
momentum = 0.8
nesterov = True
sgd_optimizer = SGD(lr = learning_rate, decay = decay,
                    momentum = momentum, nesterov = nesterov)
model.compile(loss = 'categorical_crossentropy',
                               optimizer = sgd_optimizer,
                               metrics = ['accuracy'])





batch_size = 5 
epochs = 30
model.fit(train_input, train_output,
                           epochs = epochs,
                           batch_size = batch_size, 
                           shuffle = True,
                           validation_data = (valid_input, valid_output),
                           )


print(model.evaluate(valid_input, valid_output, verbose=0))
MODEL_PATH = 'obj_reco/tst_model.h5'
model.save(MODEL_PATH)



#history.loss_plot('epoch')
from sklearn.metrics import classification_report, confusion_matrix

#Confution Matrix and Classification Report
Y_pred = model.predict(valid_input)
y_pred = np.argmax(Y_pred,axis=1)
print('Confusion Matrix')
valid_output=np.argmax(valid_output,axis=1)
print(confusion_matrix(valid_output, y_pred))
print('Classification Report')
classes = ['Healthy',
            'Gray_leaf Spot',      
        'Northern Leaf Blight',       
            'Common Rust' ]
print(classification_report(valid_output, y_pred, target_names=classes))





