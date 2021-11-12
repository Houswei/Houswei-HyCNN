
#coding:utf8
import scipy.io as sci
import spectral as spc
from sklearn import preprocessing
import numpy as np
import  matplotlib.pyplot as plt
import cv2

ksc_color =np.array([[255,255,255],
     [184,40,99],
     [74,77,145],
     [35,102,193],
     [238,110,105],
     [117,249,76],
     [114,251,253],
     [126,196,59],
     [234,65,247],
     [141,79,77],
     [183,40,99],
     [0,39,245],
     [90,196,111],
        ])

input_image = sci.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
output_image = sci.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']

img = spc.imshow(input_image, (30, 20, 10), classes = output_image, colors=ksc_color)
img.set_display_mode('overlay')
img.class_alpha = 0.5

ground_truth = spc.imshow(classes = output_image,figsize =(9,9),colors=ksc_color)

spc.view_cube(input_image)

cv2.imshow('tt', input_image[:,:,(30,20,10)])
cv2.waitKey()
# gt = preprocessing.minmax_scale(output_image, (0,255)).astype('int')
# gt = np.tile(np.expand_dims(gt,-1), 3)
# plt.imshow(gt)
# plt.show()