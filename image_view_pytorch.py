## after apply any leyer of the forward function in pytorch 
img=x1[:,:3,:,].cpu().data.numpy()
img=np.transpose(img,(1,2,0))
cv2.imshow('out',img)
cv2.waitKey(0)
