Learn more or give us feedback
import numpy as np
import cv2
import os
import csv

def save_to_exel(lists, filename):
    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename, 'a') as csvfile:
        fwriter = csv.writer(csvfile, delimiter=',',lineterminator='\n')
        fwriter.writerows(lists)
        csvfile.close()


root_file = "b"
exel = [["image name" ,"width", "height","class", "x_min", "Y_min","x_max", "y_max"]]
output_file = "outputs"


for directory, subdirectories, files in os.walk(root_file):
    for file in files:
        name = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1]
        img = os.path.join(directory, file)
        print (name)
        im = cv2.imread(img)
        height, width, channels = im.shape
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x_min,Y_min,x_max,y_max = cv2.boundingRect(cnt)
        #exel.append([name, x, y, w, h])
        x_max=x_max+x_min
        y_max=y_max+Y_min
        exel.append([img ,width,height ,"tumor",x_min, Y_min, x_max, y_max])

save_to_exel(exel, output_file + "/" + "table.csv")
