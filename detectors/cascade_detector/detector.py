import cv2, sys, os, numpy as np

class Detector:
	cascadeEarsLeft = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_leftear.xml'))
	cascadeEarsRight = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_rightear.xml'))
	
	cascadeFacesDefault = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_frontalface_default.xml'))
	cascadeFacesAlt = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_frontalface_alt_tree.xml'))
	
	def detectEars(self, img):
		det_listEarsLeft = self.cascadeEarsLeft.detectMultiScale(img, 1.08, 1)
		det_listEarsRight = self.cascadeEarsRight.detectMultiScale(img, 1.08, 1)
		
		if(len(det_listEarsLeft) > 0 and len(det_listEarsRight)>0):
			return np.concatenate((det_listEarsLeft, det_listEarsRight), axis=0), []
		elif(len(det_listEarsRight) == 0):
			return det_listEarsLeft, []
		return det_listEarsRight, []
	
	def detectFaces(self, img):
		#det_listFacesAlt, numDetections = self.cascadeFacesAlt.detectMultiScale2(img, scaleFactor=1.08, minNeighbors=1)
		#return det_listFacesAlt, numDetections
		det_listFacesAlt, rejectLevels, levelWeights = self.cascadeFacesAlt.detectMultiScale3(img, scaleFactor=1.08, minNeighbors=1, outputRejectLevels = True)
		return det_listFacesAlt, levelWeights

if __name__ == '__main__':
	fname = sys.argv[1]
	img = cv2.imread(fname)
	detector = Detector()
	detected_loc = detector.detect(img)
	for x, y, w, h in detected_loc:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(fname + '.detected.jpg', img)