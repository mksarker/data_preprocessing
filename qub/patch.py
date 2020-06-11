from glob import glob, glob0
# from scipy.io import imread
from skimage.io import imread
import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil
import png
from PIL import Image
import sys
# from scipy.misc import imresize
split = 'train'

source_dir = 'C:/Users/3055638/Documents/data/Mostafa/aug/CD3/%s' % split

out_dir = 'C:/Users/3055638/Documents/data/Mostafa/aug/CD3/%s_split' % split

out_orig_images_dir = os.path.join(out_dir, 'Images')
out_bin_images_dir = os.path.join(out_dir, 'Labels')

shutil.rmtree(out_dir, ignore_errors=True)

os.mkdir(out_dir)

os.mkdir(out_orig_images_dir)
os.mkdir(out_bin_images_dir)

## D:\PathLake\HoverNet\my_data\my_ConSep\train\Images

orig_images_dir = os.path.join(source_dir, 'Images')
bin_images_dir = os.path.join(source_dir, 'Labels')


files = list(filter(lambda filename: filename.endswith(".tif"), os.listdir(orig_images_dir)))
for filename in tqdm(files):

    orig_img = cv2.imread(os.path.join(orig_images_dir, filename))
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    # orig_img = imresize(orig_img, (512, 512), interp='bicubic')
    bin_img = imread(os.path.join(bin_images_dir, filename))
    # bin_img = imresize(bin_img, (512, 512), interp='bicubic')

    dims = orig_img.shape

    h_2 = dims[0] // 2
    w_2 = dims[1] // 2
    n = 1
    for i in range(2):
        for j in range(2):
            im = orig_img[i * h_2: (i + 1) * h_2, j * w_2: (j + 1) * w_2]
            cv2.imwrite(os.path.join(out_orig_images_dir, 'p_%d_%s' % (n, filename)), im)
            img = bin_img[i * h_2: (i + 1) * h_2, j * w_2: (j + 1) * w_2]
            cv2.imwrite(os.path.join(out_bin_images_dir, 'p_%d_%s' % (n, filename)), img)
            pp=imread(out_bin_images_dir+ '/p_%d_%s' % (n, filename))
            unq=np.unique(pp)
            print("after", unq)
            # binary_transform = np.array(img).astype(np.uint8)
            # img = Image.fromarray(binary_transform, 'P')
            # img.save(out_bin_images_dir, 'p_%d_%s' % (n, filename)+ ".png")
            n += 1