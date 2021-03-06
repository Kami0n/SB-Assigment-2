import argparse
from detectors.yolo_face.utils import *

printArgs = False
#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./detectors/yolo_face/cfg/yolov3-face.cfg',
					help='path to config file')
parser.add_argument('--model-weights', type=str,
					default='./detectors/yolo_face/model-weights/yolov3-wider_16000.weights',
					help='path to weights of model')
parser.add_argument('--image', type=str, default='',
					help='path to image file')
parser.add_argument('--video', type=str, default='',
					help='path to video file')
parser.add_argument('--src', type=int, default=0,
					help='source of the camera')
parser.add_argument('--output-dir', type=str, default='outputs/',
					help='path to the output directory')
args = parser.parse_args()
#####################################################################
# print the arguments
if printArgs:
	print('----- info -----')
	print('[i] The config file: ', args.model_cfg)
	print('[i] The weights of model file: ', args.model_weights)
	print('[i] Path to image file: ', args.image)
	print('[i] Path to video file: ', args.video)
	print('###########################################################\n')

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

class Detector:
	def detectFaces(self, img):
		# Create a 4D blob from a frame.
		blob = cv2.dnn.blobFromImage(img, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
		# Sets the input to the network
		net.setInput(blob)
		# Runs the forward pass to get output of the output layers
		outs = net.forward(get_outputs_names(net))
		# Remove the bounding boxes with low confidence
		faces, confidences = post_process(img, outs, CONF_THRESHOLD, NMS_THRESHOLD)
		return faces, confidences