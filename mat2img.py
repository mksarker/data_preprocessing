from scipy.io import loadmat
import cv2
import numpy as np
# from skimage.io import imread, imsave
from scipy.misc import imsave, imread
import os
import png
from PIL import Image

root_dir='D:/PathLake/HoverNet/data/CoNSeP/Train/Labels/'
dest_dir='D:/PathLake/HoverNet/my_data/my_ConSep/train/lbl_map/'

for file in os.listdir(root_dir):
     if file.endswith('mat'):
        main_part=file.split('.mat')[0]
        mat_file= loadmat(root_dir+main_part+'.mat')
        # print(mat_file.keys())
        # print (main_part)
        ### 'inst_map', 'type_map', 'inst_type', 'inst_centroid']
        # print (mat_file['type_map'].shape)
        im=mat_file['inst_map']

        unq=np.unique(im)
        unq=unq[1:]
        # print("before",unq)
        # im=cv2.resize(im,(512,512))
        new_mask=0*im
        for i in range(unq.shape[0]):
            inds=np.where(im==unq[i])
            new_mask[inds]=unq[i]

        # print(np.unique(new_mask))

        # png.from_array(new_mask[:], 'L').save(dest_dir+ main_part + ".png")

        binary_transform = np.array(new_mask).astype(np.uint8)
        img = Image.fromarray(binary_transform, 'P')

        img.save(dest_dir+ main_part + ".png")

        # cv2.imshow('new_mask_before:',new_mask)

        pp=imread(dest_dir+ main_part + ".png")

        # cv2.imshow('new_mask_after:',pp)

        # print(np.max(pp))
        unq=np.unique(pp)
        # print("after", unq)
        # print(pp.shape)

        # print(np.sum(np.abs(pp-new_mask)))

        # # cv2.waitKey(0)




# from scipy.io import loadmat
# import cv2
# import numpy as np
# from scipy.misc import imsave, imread
# import os
# import png
# from PIL import Image

# root_dir='D:/PathLake/HoverNet/my_data/CoNSeP/train/512x512_256x256/'
# dest_dir='D:/PathLake/HoverNet/my_data/my_ConSep/train/512x512_256x256/'


# for file in os.listdir(root_dir):
#      if file.endswith('npy'):
#         main_part=file.split('.npy')[0]
#         mat_file= np.load(root_dir+main_part+'.npy')
#         print(mat_file)
#         im=mat_file[-3:-1:]

#         unq=np.unique(im)
#         unq=unq[1:]
#         print("before",unq)
#         im=cv2.resize(im,(512,512))
#         new_mask=0*im
#         for i in range(unq.shape[0]):
#             inds=np.where(im==unq[i])
#             new_mask[inds]=unq[i]


#         # print(np.unique(new_mask))

#         # png.from_array(new_mask[:], 'L').save(dest_dir+ main_part + ".png")

#         binary_transform = np.array(new_mask).astype(np.uint8)
#         img = Image.fromarray(binary_transform, 'P')

#         img.save(dest_dir+ main_part + ".png")

#         # cv2.imshow('new_mask_before:',new_mask)

#         pp=imread(dest_dir+ main_part + ".png")

#         # cv2.imshow('new_mask_after:',pp)

#         # print(np.max(pp))
#         unq=np.unique(pp)
#         print("after", unq)
#         # print(pp.shape)

#         # print(np.sum(np.abs(pp-new_mask)))

#         # # cv2.waitKey(0)




