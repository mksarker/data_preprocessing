import os
import cv2
import skimage.color as color
import numpy as np

data_dir='path to the data directory'
dest_dir='path to the destination directory'

files= os.listdir(data_dir)

for f in files:
    ## read image
    im=cv2.imread(data_dir+f)
    ## conver image
    im= color.rgb2gray(im)

    ## if have any problem to convert using above code then use below functions
    # thresh = threshold_otsu(im)
    # binary= im > thresh
    # im=binary.astype(dtype=np.int8)
    # im=im*255.0

    ## write image
    cv2.imwrite(dest_dir+f, im)
    ## again read and check the conversion
    after=cv2.imread(dest_dir+f)
    unq=np.unique(after)
    print('after', unq)# for checking the binary values in every image