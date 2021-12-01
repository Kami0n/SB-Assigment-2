import mxnet as mx
from detectors.mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector

class Detector:
	detector = MtcnnDetector(model_folder='detectors\mxnet_mtcnn_face_detection\model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)
	def detectFaces(self, img):
		results = self.detector.detect_face(img)
		listFaces = []
		confidence = []
		if(results is not None):
			for faces in results[0]:
				confidence.append(faces[4])
				listFaces.append([int(faces[0]), int(faces[1]), int(faces[2]-faces[0]), int(faces[3]-faces[1]) ])
		return listFaces, confidence