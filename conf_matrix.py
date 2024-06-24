# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 22:22:57 2019

@author: HP
"""

#confusion_matrix
import numpy as np
import matplotlib.pyplot as plt





classes = ['Healthy',
            'Gray_leaf Spot',      
        'Northern Leaf Blight',       
            'Common Rust' ]

confusion_matrix = np.array([(233,0,0,1),(1,225,2,29),(0,2,256,3),(1,36,2,179)],dtype=np.float64)


conf_mat=confusion_matrix  
np.seterr(divide='ignore',invalid='ignore')

def show_confMat(confusion_mat, classes_name, set_name, out_dir):
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes_name)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    cmap = plt.cm.get_cmap('Greys') 
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    
    xlocations = np.array(range(len(classes_name)))
    plt.xticks(xlocations, classes_name, rotation=12, fontsize=6)
    plt.yticks(xlocations, classes_name, fontsize=8)
    plt.xlabel('Predicted label', fontsize=8)
    plt.ylabel('Ground truth', fontsize=8)
    plt.title('Confusion_Matrix' + set_name, fontsize=8)

    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.jpg'),dpi=300)
    plt.show()
    plt.close()

show_confMat(conf_mat, classes, "", "./")
