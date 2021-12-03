import cv2
import numpy as np
np.set_printoptions(suppress=True)
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
import time

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        
        with open('config.json') as config_file:
            config = json.load(config_file)
        
        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']
    
    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [int(i) for i in l_arr]
                    annot.append(l_arr)
            return annot
    
    def run_evaluation(self):
        im_list = sorted(glob.glob(self.images_path + '/*.*', recursive=True))
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()
        
        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        #import detectors.insightface.detector as insightface_detector
        #import detectors.DSFDPytorchInference.detector as DSFDPytorchInference_detector
        #import detectors.yolo_face.detector as yolo_faceDetector
        #import detectors.mxnet_mtcnn_face_detection.detector as mxnet_detector
        
        
        import detectors.yoloDetect_ears.detector as my_yolo_detector
         
        # import detectors.your_super_detector.detector as super_detector
       
        
        cascade_detector = cascade_detector.Detector()
        #insightface_detector = insightface_detector.Detector()
        #DSFDPtI_detector = DSFDPytorchInference_detector.Detector()
        #yolo_faceDetector = yolo_faceDetector.Detector()
        #mxnet_detector = mxnet_detector.Detector()
        my_yolo_detector = my_yolo_detector.Detector()
        
        allImagesNumber = len(im_list)
        counter = 0
        printBar = False
        printVerbose = True
        mAPEnable = False
        showBoxes = False
        
        allTPFP = {} # dict
        for threshold in np.arange(0.5, 0.95, 0.05):
            allTPFP[str(threshold)] = []
        allConfidences = []
        countAllBboxes = 0
        
        if printBar:
            printProgressBar(0, allImagesNumber, prefix = '  Progress:', suffix = 'Complete', length = 120)
        
        for im_name in im_list:
            if(printVerbose):
                print("Analysing image: "+im_name)
            
            img = cv2.imread(im_name) # Read an image
            img = preprocess.allPreprocessing(img) # Apply some preprocessing
            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)
            countAllBboxes += len(annot_list)
            
            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            
            # Faces
            #prediction_list, confidences = cascade_detector.detectFaces(img)
            #prediction_list = insightface_detector.detectFaces(img)
            #prediction_list, confidences = DSFDPtI_detector.detectFaces(img)
            #prediction_list, confidences = yolo_faceDetector.detectFaces(img)
            #prediction_list, confidences = mxnet_detector.detectFaces(img)
            
            # Ears
            prediction_list, confidences = cascade_detector.detectEars(img)
            #prediction_list, confidences = my_yolo_detector.detectEars(img)
            
            # Only for detection:
            p, gt = eval.prepare_for_detection(prediction_list, annot_list)
            iou = eval.iou_compute(p, gt)
            iou_arr.append(iou)
            
            if(mAPEnable):
                predictionsSorted, groundTruthSorted, confidencesSorted = eval.boundigBoxesPairs(prediction_list, annot_list, confidences, showBoxes)
                allConfidences.extend(confidencesSorted)
                for threshold in np.arange(0.5, 0.95, 0.05):
                    TPFP = eval.TPFP_compute(predictionsSorted, groundTruthSorted, threshold/100)
                    allTPFP[str(threshold)].extend(TPFP)
            
            if printBar:
                counter += 1
                printProgressBar(counter, allImagesNumber, prefix = '  Progress:', suffix = 'Complete', length = 120)
        
        print("\n")
        miou = np.average(iou_arr)
        print("Average IOU:", f"{miou:.2%}")
        print("\n")
        
        if(mAPEnable):
            mAP = eval.averagePrecision_compute(allTPFP, allConfidences, countAllBboxes)
            print("mAP:", f"{mAP:.5}")
            print("\n")

# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

if __name__ == '__main__':
    #os.system('cls')
    ev = EvaluateAll()
    start = time.time() # start stopwatch
    ev.run_evaluation()
    end = time.time() # end stopwatch
    timeDiff = (end - start)
    print("Porabljen čas: ", timeDiff)