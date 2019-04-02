import os
import cv2
import skimage.color as color
import numpy as np
import png

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
    ### if problem with saving CV2 library then use PNG library
    #png.from_array(img[:], 'L').save(dest_dir+fname+'.png')
    ## again read and check the conversion
    after=cv2.imread(dest_dir+f)
    unq=np.unique(after)
    print('after', unq)# for checking the binary values in every image