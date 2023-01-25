#!/usr/bin/python3

import cv2 as cv
import yolo

original = cv.imread('dog.jpg')
myImage = yolo.Image(original)
myImage.process()
print(myImage)

pic = cv.imread(myImage.filename)
cv.imshow("original picture", original)
cv.imshow("Processed Picture", myImage.img)
cv.imshow("Picture from disk", pic)
cv.waitKey(0)

