from __future__ import print_function
from skimage import io
from skimage.util import view_as_windows
import os
import numpy as np
from tqdm import tqdm
import argparse
import glob
from pathlib import Path
from PIL import Image
from skimage.transform import resize


img_folder = "C:/Users/3055638/Documents/data/Mostafa/original/CD3/images/training"
# img_folder = '/media/balamurali/NewVolume2/Deep_network/mitosis'
out_folder = "C:/Users/3055638/Documents/data/Mostafa/patch-512/images/training"
# out_folder = '/media/balamurali/NewVolume2/Deep_network/mitosis/output'
img_ext = "tif"
# img_size = (200, 200)

# img_stride = (32, 32)

# stride_list = []
# window_shape_list = []
# for each in img_size:
#     window_shape_list.append((each))

# for each in img_stride:
#     stride_list.append(each)

# print("Settings: \nInput Folder:{}\nOutput Folder:{}\nImg ext:{}\nImg size:{}\n".format(img_folder, out_folder, img_ext,
#                                                                                         window_shape_list))

imgs_path = glob.glob(os.path.join(img_folder, '*.' + img_ext))
# imgs_path = os.listdir(img_folder)
# print(imgs_path)

# for window_shape in tqdm(window_shape_list):
#     print("Window size : {}\n".format(window_shape))
window_shape= 512
stride= 32
# for stride in tqdm(stride_list):
#     print("Stride size : {}\n".format(stride))

#     # out_folder_temp = os.path.join(out_folder, 'size_' + str(window_shape) + '_' + 'stride_' + str(stride))

#     # if not os.path.exists(out_folder_temp):
#     #     print("Directory {} not found,creating one".format(out_folder_temp))
#     #     os.mkdir(out_folder_temp)

    # print("Sampling images\n")
for img_path in tqdm(imgs_path):
    # Read image
    img = io.imread(img_path)
    print(img_path)
    print(img.shape)
    h,w,c = img.shape
    if h<=514:                               
        img= Image.fromarray(img)
        img = img.resize((512,512), Image.ANTIALIAS)
        img.save(os.path.join(out_folder, os.path.basename(img_path)[:-4] + '.png'))               
    elif w<=514:
        # img = resize(img,(200,200))
        img= Image.fromarray(img)
        img = img.resize((512,512), Image.ANTIALIAS)
        img.save(os.path.join(out_folder, os.path.basename(img_path)[:-4] + '.png'))
    else:    
        if len(img.shape) == 3:
            r_channel = img[:, :, 0]
            g_channel = img[:, :, 1]
            b_channel = img[:, :, 2]

            r_sample = view_as_windows(r_channel, window_shape, step=stride)
            g_sample = view_as_windows(g_channel, window_shape, step=stride)
            b_sample = view_as_windows(b_channel, window_shape, step=stride)
        else:
            r_sample = view_as_windows(img, window_shape, step=stride)

        no_of_rows = r_sample.shape[0]
        no_of_cols = r_sample.shape[1]
        cnt = 0
        for row in tqdm(range(no_of_rows)):
            for col in tqdm(range(no_of_cols)):
                if len(img.shape) == 3:
                    sample_r = r_sample[row, col]
                    sample_g = g_sample[row, col]
                    sample_b = b_sample[row, col]
                    img_sample = np.dstack((sample_r, sample_g, sample_b))

                else:
                    img_sample = r_sample[row, col]
                    
                    # print (img_sample.shape)
                cnt += 1
                img_sample_path = (os.path.join(out_folder,
                                                os.path.basename(img_path)[:-4] + '_' + str(cnt) + '.png')).format(
                    cnt)
                img_sample= Image.fromarray(img_sample)
                img_sample = img_sample.resize((512,512), Image.ANTIALIAS)
                img_sample.save(img_sample_path)
                # io.imsave(img_sample_path, img_sample)
                # break
                # break
            # break
        # break

