#!/usr/bin/python3

import cv2 as cv
import yolo

original = cv.imread('img/IMG_9723.jpg')
myImage = yolo.Image(original)
myImage.process()
print(myImage)

cv.imshow("original picture", original)
cv.imshow("Processed Picture", myImage.img)
cv.waitKey(0)

