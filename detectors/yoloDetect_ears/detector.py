import cv2, sys, os, numpy as np
# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle

class Detector:
	# Load Yolo
	net = cv2.dnn.readNet("detectors/yoloDetect_ears/yolov3_training_final.weights", "detectors/yoloDetect_ears/yolov3_testing.cfg")

	# Name custom object
	classes = ["Ear"]
	layer_names = net.getLayerNames()
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	
	def detectEars(self, img):
		output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
		
		img = cv2.resize(img, None, fx=0.4, fy=0.4)
		#height, width, channels = img.shape
		
		width = 480
		height = 360

		# Detecting objects
		blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)

		self.net.setInput(blob)
		outs = self.net.forward(output_layers)

		# Showing informations on the screen
		class_ids = []
		confidences = []
		boxes = []
		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.3:
					# Object detected
					#print(detection)
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					# Rectangle coordinates
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)
					
					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)
		
		#indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
		#print(indexes)
		#font = cv2.FONT_HERSHEY_PLAIN
		#for i in range(len(boxes)):
			#if i in indexes:
				#x, y, w, h = boxes[i]
				#label = str(self.classes[class_ids[i]])
				#color = self.colors[class_ids[i]]
				#cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
				#cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
				#print(x, y, w, h, round(confidences[i], 3))
		#cv2.imshow("Image", img)
		#key = cv2.waitKey(0)
		
		return boxes, confidences