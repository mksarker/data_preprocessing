import cv2
import os
import Augmentor
        
p = Augmentor.Pipeline('C:/Users/3055638/Documents/data/Mostafa/original/CD3/images/validation')
p.ground_truth('C:/Users/3055638/Documents/data/Mostafa/original/CD3/annotations/validation')
# Add operations to the pipeline as normal:
# p.rotate(probability=1, max_left_rotation=45, max_right_rotation=45)
p.flip_left_right(probability=0.8)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.8)
# p.histogram_equalisation( probability=0.8)
# p.random_brightness(probability=0.5,min_factor=0.8,max_factor=1.2)
p.sample(400)

#
#