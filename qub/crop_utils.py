import numpy as np
from skimage.color import rgb2gray

def get_seq(lst):
    def f_(lst):
        lst2 = lst[:]
        i = 0
        while i < len(lst2) - 1:
            if lst2[i + 1] - lst2[i] != 1:
                del lst2[i]
            else:
                i += 1  
        return lst2  
    
    l = f_(lst)
    if l == lst:
        return lst
    return get_seq(l)


def cropping_box(img, window_size=200, round_factor=0.4, epsilon=1e4):
    
    assert img.ndim == 2, "Image must be grayscle"
    _, width = img.shape
    
    img = rgb2gray(img)
    bin_img = np.round(img + round_factor)
    
    boxes = []
    w = 0
    b = 0
    while w < width + window_size // 2:
        sub_img = bin_img[:, w: w + window_size]
        
        th = sub_img.sum() / epsilon

        if th >= 1:
            boxes.append(b)
        
        b += 1
        w += window_size

    filtered_boxes = get_seq(boxes)
    l = filtered_boxes[0] * 200
    u = filtered_boxes[-1] * 200
    
    return l, u

def crop_img_and_mask(img, mask, window_size=200, round_factor=0.4, epsilon=1e4):

    l, u = cropping_box(img, window_size, round_factor, epsilon)

    cropped_img = img[:, l: u]
    cropped_mask = mask[:, l: u]

    return cropped_img, cropped_mask