
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import png

def imgcolor(img,color,shape):
    img=img.reshape((-1,3))
    img=np.multiply(img, color)
    img=img.reshape((shape[0],shape[1],3))
    return img

# Read the image from the directory
dir_out='/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/output/cgan-im/'

img_list=os.listdir('data/2017/Test/OR/')
for filename in img_list:
    if filename.endswith('.jpg'):
        print(filename)
        filename=filename.split('.')[0]
        img_gt=cv2.imread('data/2017/Test/GT/'+filename+'.png')
        org_img=cv2.imread('data/2017/Test/OR/'+filename+'.jpg')
        #org_img=cv2.resize(org_img,(96,96))

        img_gt=img_gt/255
        img_gt=np.array(img_gt,dtype=np.uint8)

        img_gt[np.where(img_gt<1)]=0

        img_predict=cv2.imread('predict/cgan/2017_Test/'+filename+'.jpg')
        # kernel = np.ones((5,5),np.uint8)
        # img_predict = cv2.morphologyEx(img_predict, cv2.MORPH_CLOSE, kernel)
        # img_predict=cv2.resize(img_predict,(565,584))
        img_predict=np.array(img_predict,dtype=np.uint8)

        img_predict=img_predict/255
        img_predict=np.array(img_predict,dtype=np.uint8)

        img_predict[np.where(img_predict<1)]=0


        result=img_predict-org_img

#   Compute the FP, TP, FN, TN *****************************************

        FP=0*img_predict
        FP[np.where(result>0)]=1
        FN=0*img_predict
        FN[np.where(result<0)]=1
        TP=0*img_predict
        TP=cv2.bitwise_and(img_predict,org_img)
        TN=0*img_predict
        TN=cv2.bitwise_and(1-img_predict,1-org_img)

        aa=cv2.bitwise_or(img_predict,org_img)

        #np.multiply(matrix, color)

#    Fill the colors into a mask ********************************************

        #colors=[ [231, 76, 60] ,  [248, 196, 113]  , [ 46, 204, 113   ],  [ 250, 51, 212 ]]  
        colors=[ [255, 255, 255] ,  [ 0, 0,255]  , [255, 127, 0],  [127, 0, 0]]   # Red, Yellow, Green,Blue

        colors=np.array(colors,dtype=np.uint8 )
        shape=org_img.shape

        img_gt=imgcolor(org_img,colors[0],shape)
        img_predict=imgcolor(img_predict,colors[1],shape)

        FP=imgcolor(FP,colors[2],shape)
        TP=imgcolor(TP,colors[3],shape)
        

#   Image Blending opearation ********************************************
        dst1 = cv2.addWeighted(FP,0.9,TP,0.5,0)
        Blend_org = cv2.addWeighted(img_gt,0.5,  img_predict,0.9,0)

        #Blend_org=  cv2.addWeighted(org_img,0.5,dst2,0.5,0)

        cv2.imwrite(dir_out+filename+'.jpg', Blend_org)

        # cv2.imshow('color',img_gt)
        # cv2.imshow('FP',FP)
        # cv2.imshow('TP',TP)
        # cv2.imshow('img_prediorg_imgct',img_predict)

        # cv2.imshow('dst1',dst1)
        # cv2.imshow('dst2',dst2)
        # cv2.imshow('Blend_org',Blend_org)
        # cv2.waitKey(0)





org_img



        



