import os
import cv2
import skimage.color as color
import numpy as np
import png

import pandas as pd

data_dir='E:/covid-chest-xray/images/'
dest_dir='E:/split/'

files= os.listdir(data_dir)

csv_path = 'E:/covid-chest-xray/metadata.csv'
df = pd.read_csv(csv_path)



for i, j in zip (df['finding'], df['filename']):
    print(i)
    
    try:
        im=cv2.imread(data_dir+j)


        im = cv2.resize(im, (400, 400), interpolation = cv2.INTER_LINEAR)

        if not os.path.exists(dest_dir+ i):
            os.mkdir(dest_dir+ i)
            print("Directory " , i ,  " Created ")
        else:    
            print("Directory " , i ,  " already exists")


        cv2.imwrite(dest_dir+ i + '/' + j, im)
    except:
        print('No valid images')


# img_path = os.path.sep.join([data_dir, 'images', df['filename'].iloc[idx]])


# for f in files:
#     print(f)
#     file_name = f.split('.')[0]
#     ## read image
#     im=cv2.imread(data_dir+f)
#     im = cv2.resize(im, (400, 400), interpolation = cv2.INTER_LINEAR)
#     # print('before')
#     # print(len(im.shape))
#     # ## conver image
#     # im= color.rgba2rgb(im)

#     # ## if have any problem to convert using above code then use below functions
#     # # thresh = threshold_otsu(im)
#     # # binary= im > thresh
#     # # im=binary.astype(dtype=np.int8)
#     # # im=im*255.0

#     # ## write image
#     cv2.imwrite(dest_dir+file_name+'.jpg', im)
#     # ### if problem with saving CV2 library then use PNG library
#     # png.from_array(im[:], 'L').save(dest_dir+file_name+'.png')
#     # ## again read and check the conversion
#     # after=cv2.imread(dest_dir+file_name+'.jpg')
#     # print('after')
#     # print(len(after.shape))
#     # unq=np.unique(after)
#     # print('after', unq)# for checking the binary values in every image