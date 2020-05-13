from glob import glob, glob0
from scipy.misc import imread
import cv2
import os
from tqdm import tqdm
import shutil
import sys
import numpy as np

split = 'val'

source_dir = '/home/vivek/Downloads/SPLIT_EYE/RESULT/' 

out_dir = '/home/vivek/Downloads/SPLIT_EYE/test_combined/'


shutil.rmtree(out_dir, ignore_errors=True)
os.mkdir(out_dir)

image_names = []
with open(os.path.join(source_dir, 'names.txt')) as fp:
    image_names = [l.strip() for l in fp]

for im_name in tqdm(image_names):
    imgs = []
    for i in range(1, 5):
        filename = os.path.join(source_dir, 'p_%d_%s' % (i, im_name))
        img = imread(filename)
        imgs.append(img)
    
    img1 = np.concatenate(imgs[:2], axis=1)
    img2 = np.concatenate(imgs[2:], axis=1)
    del imgs
    img = np.concatenate([img1, img2], axis=0)
    del img1, img2
    filename = os.path.join(out_dir, im_name)
    cv2.imwrite(filename, img)
    
    
