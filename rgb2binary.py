import os
import cv2
import skimage.color as color
import numpy as np

data_dir='path to the data directory'
dest_dir='path to the destination directory'

files= os.listdir(data_dir)

for f in files:
    im=cv2.imread(data_dir+f)
    im= color.rgb2gray(im)
    cv2.imwrite(dest_dir+f, im)
    gt=cv2.imread(dest_dir+f)
    unq=np.unique(gt)
    print('after', unq)# for checking the binary values in every image