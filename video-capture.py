#!/usr/bin/python3
import cv2
import acapture

# Checking connected cameras
"""
index = 0
arr = []
while True:
    cap = cv2.VideoCapture(index)
    if not cap.read()[0]:
        break
    else:
        arr.append(index)
    cap.release()
    index += 1

print(arr)
"""

"""
import acapture
import cv2

cap = acapture.open(0)
while True:
    check,frame = cap.read() # non-blocking
    if check:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow("test",frame)
        cv2.waitKey(1)
"""

import cv2
import acapture
import pyglview
viewer = pyglview.Viewer()
cap = acapture.open(0) # Camera 0,  /dev/video0
def loop():
    check,frame = cap.read() # non-blocking
    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if check:
        viewer.set_image(frame)
viewer.set_loop(loop)
viewer.start()
