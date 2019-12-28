
import moviepy.editor as mpy
import os
import cv2

root = "/media/mostafa/RESEARCH/MICCAI2019/Results/SKIN_MICCAI2019_PAPER_RESULTS/output/mobilegan-blend/"
images = os.listdir(root)

rendered_frames = []
for image in images:
    print(image)
    im = cv2.imread(root+image)
    # cv2.imshow('image', im)
    # cv2.waitKey(500)
    rendered_frames.append(im)

clip = mpy.ImageSequenceClip(rendered_frames, fps=2)
clip.write_videofile('skin_segmentation.mp4')
