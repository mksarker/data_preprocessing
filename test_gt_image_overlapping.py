
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import png
from skimage.morphology import erosion, square, dilation

def imgcolor(img,color,shape):
    img=img.reshape((-1,3))
    img=np.multiply(img, color)
    img=img.reshape((shape[0],shape[1],3))
    return img

# Read the image from the directory
dir_out='/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/output/mobilegan-blend/'

img_list=os.listdir('data/2016/Test/GT/')  #data/2017/Test/GT/
for filename in img_list:
    if filename.endswith('.png'):
        print(filename)
        filename=filename.split('.')[0]
        img_gt=cv2.imread('data/2016/Test/GT/'+filename+'.png')
        #img_gt=cv2.resize(img_gt,(56,96))
        org_img=cv2.imread('data/2016/Test/OR/'+filename+'.jpg')
        #org_img=cv2.resize(org_img,(96,96))

        img_gt=img_gt/255
        img_gt=np.array(img_gt,dtype=np.uint8)

        img_gt[np.where(img_gt<1)]=0

        img_predict=cv2.imread('predict/mobilegan/2016_Test/'+filename+'.jpg')

        # kernel = np.ones((5,5),np.uint8)
        # img_predict = cv2.erode(img_predict,kernel,iterations = 2)

        #img_predict=cv2.resize(img_predict,(565,584))
        kernel = np.ones((5,5),np.uint8)
        img_predict = cv2.morphologyEx(img_predict, cv2.MORPH_CLOSE, kernel)
        img_predict=np.array(img_predict,dtype=np.uint8)
        
        
        # img_predict = cv2.erosion(img_predict,)

        img_predict=img_predict/255
        img_predict=np.array(img_predict,dtype=np.uint8)

        img_predict[np.where(img_predict<1)]=0


        result=img_predict-img_gt

#   Compute the FP, TP, FN, TN *****************************************

        FP=0*img_predict
        FP[np.where(result>0)]=1
        FN=0*img_predict
        FN[np.where(result<0)]=1
        TP=0*img_predict
        TP=cv2.bitwise_and(img_predict,img_gt)
        TN=0*img_predict
        TN=cv2.bitwise_and(1-img_predict,1-img_gt)

        aa=cv2.bitwise_or(img_predict,img_gt)

        #np.multiply(matrix, color)

#    Fill the colors into a mask ********************************************

        colors=[ [231, 76, 60] ,  [248, 196, 113]  , [ 46, 204, 113   ],  [ 250, 51, 212 ]]  
        # colors=[ [0, 0, 255] ,  [ 0, 255,0]  , [255, 255, 0],  [255, 0, 0]]   # Red, Yellow, Green,Blue

        colors=np.array(colors,dtype=np.uint8 )
        shape=img_gt.shape

        img_gt=imgcolor(img_gt,colors[0],shape)
        img_predict=imgcolor(img_predict,colors[1],shape)

        FP=imgcolor(FP,colors[2],shape)
        TP=imgcolor(TP,colors[3],shape)
        

#   Image Blending opearation ********************************************
        dst1 = cv2.addWeighted(FP,0.5,TP,0.5,0)
        # Blend_org = cv2.addWeighted(img_gt,0.8,  img_predict,0.5,0)
        Blend_org=  cv2.addWeighted(org_img,0.8 ,dst1,0.5,0)

        cv2.imwrite(dir_out+filename+'.jpg', Blend_org)

        # cv2.imshow('color',img_gt)
        # cv2.imshow('FP',FP)
        # cv2.imshow('TP',TP)
        # cv2.imshow('img_prediorg_imgct',img_predict)

        # cv2.imshow('dst1',dst1)
        # cv2.imshow('dst2',dst2)
        # cv2.imshow('Blend_org',Blend_org)
        # cv2.waitKey(0)





# org_img



        




import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import png
from skimage.morphology import erosion, square, dilation

def imgcolor(img,color,shape):
    img=img.reshape((-1,3))
    img=np.multiply(img, color)
    img=img.reshape((shape[0],shape[1],3))
    return img

# Read the image from the directory
dir_out='/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/output/mobilegan-blend/'

img_list=os.listdir('data/2016/Test/GT/')  #data/2017/Test/GT/
for filename in img_list:
    if filename.endswith('.png'):
        print(filename)
        filename=filename.split('.')[0]
        img_gt=cv2.imread('data/2016/Test/GT/'+filename+'.png')
        #img_gt=cv2.resize(img_gt,(56,96))
        org_img=cv2.imread('data/2016/Test/OR/'+filename+'.jpg')
        #org_img=cv2.resize(org_img,(96,96))

        img_gt=img_gt/255
        img_gt=np.array(img_gt,dtype=np.uint8)

        img_gt[np.where(img_gt<1)]=0

        img_predict=cv2.imread('predict/mobilegan/2016_Test/'+filename+'.jpg')

        # kernel = np.ones((5,5),np.uint8)
        # img_predict = cv2.erode(img_predict,kernel,iterations = 2)

        #img_predict=cv2.resize(img_predict,(565,584))
        kernel = np.ones((5,5),np.uint8)
        img_predict = cv2.morphologyEx(img_predict, cv2.MORPH_CLOSE, kernel)
        img_predict=np.array(img_predict,dtype=np.uint8)
        
        
        # img_predict = cv2.erosion(img_predict,)

        img_predict=img_predict/255
        img_predict=np.array(img_predict,dtype=np.uint8)

        img_predict[np.where(img_predict<1)]=0


        result=img_predict-img_gt

#   Compute the FP, TP, FN, TN *****************************************

        FP=0*img_predict
        FP[np.where(result>0)]=1
        FN=0*img_predict
        FN[np.where(result<0)]=1
        TP=0*img_predict
        TP=cv2.bitwise_and(img_predict,img_gt)
        TN=0*img_predict
        TN=cv2.bitwise_and(1-img_predict,1-img_gt)

        aa=cv2.bitwise_or(img_predict,img_gt)

        #np.multiply(matrix, color)

#    Fill the colors into a mask ********************************************

        colors=[ [231, 76, 60] ,  [248, 196, 113]  , [ 46, 204, 113   ],  [ 250, 51, 212 ]]  
        # colors=[ [0, 0, 255] ,  [ 0, 255,0]  , [255, 255, 0],  [255, 0, 0]]   # Red, Yellow, Green,Blue

        colors=np.array(colors,dtype=np.uint8 )
        shape=img_gt.shape

        img_gt=imgcolor(img_gt,colors[0],shape)
        img_predict=imgcolor(img_predict,colors[1],shape)

        FP=imgcolor(FP,colors[2],shape)
        TP=imgcolor(TP,colors[3],shape)
        

#   Image Blending opearation ********************************************
        dst1 = cv2.addWeighted(FP,0.5,TP,0.5,0)
        # Blend_org = cv2.addWeighted(img_gt,0.8,  img_predict,0.5,0)
        Blend_org=  cv2.addWeighted(org_img,0.8 ,dst1,0.5,0)

        cv2.imwrite(dir_out+filename+'.jpg', Blend_org)

        # cv2.imshow('color',img_gt)
        # cv2.imshow('FP',FP)
        # cv2.imshow('TP',TP)
        # cv2.imshow('img_prediorg_imgct',img_predict)

        # cv2.imshow('dst1',dst1)
        # cv2.imshow('dst2',dst2)
        # cv2.imshow('Blend_org',Blend_org)
        # cv2.waitKey(0)





# org_img



        



