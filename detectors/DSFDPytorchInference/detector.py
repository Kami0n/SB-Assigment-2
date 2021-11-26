import cv2, sys, os, numpy as np
import glob
import detectors.DSFDPytorchInference.face_detection as face_detection

class Detector:
	
	impaths = "images"
	impaths = glob.glob(os.path.join(impaths, "*.jpg"))
	detector = face_detection.build_detector( "DSFDDetector", max_resolution=1080 )	
	
	def detectFaces(self, img):
		output = self.detector.detect(  img[:, :, ::-1] )
		
		faces = output[:, :4]
		confidence = output[:, 4]
		
		listFaces = []
		for items in faces:
			A = [int(a) for a in items]
			A[2] = A[2] - A[0]
			A[3] =  A[3] - A[1]
			listFaces.append(A)
		return listFaces, confidence