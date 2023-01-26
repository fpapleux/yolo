#!/usr/bin/python3

import cv2 as cv
import yolo

original = cv.imread('img/IMG_9697.jpg')
myImage = yolo.Image(original)
myImage.process()

print(myImage)


