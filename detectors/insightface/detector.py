import cv2, sys, os, numpy as np

import insightface
from insightface.app import FaceAnalysis

class Detector:
	app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
	app.prepare(ctx_id=0, det_size=(640, 640))
	
	#detector = insightface.model_zoo.get_model('buffalo_sc.onnx')
	#detector.prepare(ctx_id=0, det_size=(640, 640))

	def detectFaces(self, img):
		faces = self.app .get(img)
		listFaces = []
		for items in faces:
			coordinateList = items.bbox.tolist()
			A = [int(a) for a in coordinateList]
			
			A[2] = A[2] - A[0]
			A[3] =  A[3] - A[1]
			listFaces.append(A)
			
		return listFaces