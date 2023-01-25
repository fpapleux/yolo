# YOLO object detection
import cv2 as cv
import numpy as np
import time

class Object:
    def __init__ (self, name, confidence, x, y, w, h):
        self.name = name
        self.confidence = confidence
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def __str__ (self):
        myString = f"{self.name} ({self.confidence * 100}%)"
        return myString



class Image:
    
    # Self properties
    # self.img: image object to be processed/processed
    # filename: name of the file containing the processed picture
    # processTime: number of milliseconds it required to processed the image
    # objects: array of DetectedObjects

    objects = []

    def __init__ (self, img):
        self.img = img.copy()
        self.filename = ""
        self.processTime = 0
        self.objects = []

    def __str__ (self):
        printForm =  f"\Processed Image:\n------------------\n"
        printForm += f"processed file saved as: {self.filename}\n"
        printForm += f"Number of objects found: {len(self.objects)}\n"
        for object in self.objects:
            printForm += f"{object.__str__()}\n"
        return printForm

    def process (self, ):
        classes = open('coco.names').read().strip().split('\n')
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
        net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        blob = cv.dnn.blobFromImage(self.img, 1/255.0, (416, 416), swapRB=True, crop=False)
        r = blob[0, 0, :, :]
        net.setInput(blob)

        t0 = time.time()
        outputs = net.forward(ln)
        t = time.time()

        r0 = blob[0, 0, :, :]
        r = r0.copy()

        boxes = []
        confidences = []
        classIDs = []
        h, w = self.img.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.5:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        self.objects = []    
        indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                self.objects.append ( Object(classes[classIDs[i]], confidences[i], x, y, w, h))
                color = [int(c) for c in colors[classIDs[i]]]
                cv.rectangle(self.img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv.putText(self.img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        startTime = int(t0 * 1000)
        self.filename = "pic" + str(startTime) + ".jpg"
        cv.imwrite(self.filename, self.img)
