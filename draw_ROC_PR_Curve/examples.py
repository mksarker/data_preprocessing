from autoMetric import fileList,drawCurve,CollectData



if __name__=="__main__":
	gtlist = fileList('/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/data/2017/Test/GT/', '*.png')
	fcn = fileList('/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/vatiations/BL/', '*.jpg')
	unet = fileList('/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/vatiations/BL+PAM/','*.jpg')
	erfnet = fileList('/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/vatiations/MobileGAN-multiscale/','*.jpg')
	segnet = fileList('/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/vatiations/MobileGan+64x64/','*.jpg')
	size1 = fileList('/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/vatiations/MobileGAN+256x256/','*.jpg')
	loss = fileList('/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/vatiations/MobileGAN+BCE+L1/','*.jpg')
	proposed = fileList('/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/vatiations/BL+PAM+CAM(proposed MobileGAN)/','*.jpg')
	modelName=['BL','BL+PAM','MobileGAN-multiscale','MobileGAN(64x64)','MobileGAN(256x256)','MobileGAN+BCE+L1','Proposed MobileGAN']


	drawCurve(gtlist,[fcn,unet,segnet,erfnet,size1,loss,proposed],modelName,'ISIC 2017 test dataset segmentation')
