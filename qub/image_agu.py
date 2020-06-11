import cv2
import os
import numpy as np
from glob import glob
import os
import torch
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
import os
import numpy as np
import csv
import traceback
from random import shuffle
from matplotlib.pyplot import show, pause
from skimage.morphology import erosion, square, dilation
import params
import random
import png
from scipy.misc import imread, imsave, imrotate


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGaimrotatemma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


root='C:/Users/3055638/Documents/data/Mostafa/'



images= os.listdir(root+'original/CD3/images/validation/')

for file in images:
    print (file)
    filename=file.split('.tif')[0]
    img=imread(root+'original/CD3/images/validation/'+ filename+'.tif')
    img= cv2.resize(img, (512, 512) , interpolation=cv2.INTER_NEAREST)
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # luv= cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    # xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    # YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl1 = clahe.apply(gray_image)
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    mask=imread(root+'original/CD3/annotations/validation/'+ filename+'.tif')
    mask= cv2.resize(mask, (512, 512) , interpolation=cv2.INTER_NEAREST)

    # imsave(root+'aug/CD3/images/training/'+ filename+'_lab.png',lab)
    
    # imsave(root+'aug/CD3/annotations/training/'+ filename+'_lab.png',mask)

    # for gamma in np.arange(0.0, 1.5, 0.5):             
    #             # apply gamma correction and show the images
    #             gamma = gamma if gamma > 0 else 0.5
    #             adjusted = adjust_gamma(img, gamma=gamma)
    #             cv2.imwrite(root+'train/gama_a/'+ filename+'_'+str(gamma)+'_gama.png',adjusted)
    #             cv2.imwrite(root+'train/gama_b/'+ filename+'_'+str(gamma)+'_gama.png',mask)
    #             after=cv2.imread(root+'train/gama_b/'+ filename+ '_'+str(gamma)+'_gama.png')
    #             print(np.unique(after))

 

    # cv2.imwrite(root+'train/gama_a/'+ filename+'_gama.png',image_lab)
    # cv2.imwrite(root+'train/gama_b/'+ filename+'_gama.png',mask)

    # after=cv2.imread(root+'train/lab_b/'+ filename+'_lab.png')
    # print(np.unique(after))

    flip_img = np.flipud(img)
    flip_mask = np.flipud(mask)
    # for angle in range(30,60,90):
    #     rot_img = imrotate(img, angle, interp='bilinear')
    #     rot_mask = imrotate(mask, angle, interp='nearest')

    # imsave(root+'aug/CD3/images/validation/'+ filename+'_ud_rotate.png',flip_img)

    imsave(root+'aug/CD3/annotations/validation/'+ filename+'_ud_rotate.png',flip_mask)


