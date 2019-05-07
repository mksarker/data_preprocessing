import os
from shutil import copyfile, move
from scipy.misc import imread, imsave

src='annotations/training/'

dst ='filter_mask/'

file= open('filter.txt', 'r')

for f in file:
    f=f.rstrip()
    f=f.split('.jpg')[0]
    file_name= f+'.png'
    print(file_name)
    # image_ori=imread(src+f+'.jpg')
    # imsave(dst+f+'.png',image_ori)
    move(src+file_name, dst+file_name)
