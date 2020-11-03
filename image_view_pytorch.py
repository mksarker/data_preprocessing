## after apply any leyer of the forward function in pytorch 
iimg=input[:,:3,:,].cpu().data.numpy()
# print(img.shape)
img = np.squeeze(img)
# print(img.shape)
img=np.transpose(img,(1,2,0))
img = cv2.resize(img,(256,256))
cv2.imshow('out',img)
cv2.waitKey(0)
