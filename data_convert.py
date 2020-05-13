import os
import cv2
import skimage.color as color
import numpy as np
import png
from skimage.filters import threshold_otsu

data_dir='D:/CODE/SKIN/code/outputs/'
dest_dir='D:/CODE/SKIN/outputs/'

files= os.listdir(data_dir)

for f in files:
    print(f)
    file_name = f.split('.')[0]
    ## read image
    im=cv2.imread(data_dir+f)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im = cv2.resize(im, (400, 400), interpolation = cv2.INTER_LINEAR)
    unq=np.unique(im)
    print('before', unq)
    # ## conver image
    # im= color.rgba2rgb(im)

    # ## if have any problem to convert using above code then use below functions
    # thresh = threshold_otsu(im)
    # binary= im > thresh
    # im=binary.astype(dtype=np.int8)
    # im=im*255.0

    # ## write image
    # cv2.imwrite(dest_dir+file_name+'.png', im)
    # ### if problem with saving CV2 library then use PNG library
    png.from_array(im[:], 'L').save(dest_dir+file_name+'.png')
    # ## again read and check the conversion
    after=cv2.imread(dest_dir+file_name+'.png')
    # print('after')
    # print(len(after.shape))
    unq=np.unique(after)
    print('after', unq)# for checking the binary values in every image