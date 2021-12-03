import cv2
import numpy as np
from sklearn.metrics import auc

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy
class Evaluation:

    def convert2mask(self, mt, shape):
        # Converts coordinates of bounding-boxes into blank matrix with values set where bounding-boxes are.
        
        t = np.zeros([shape, shape])
        for m in mt:
            x, y, w, h = m
            cv2.rectangle(t, (x,y), (x+w, y+h), 1, -1)
        return t
    
    def prepare_for_detection(self, prediction, ground_truth):
        # For the detection task, convert Bounding-boxes to masked matrices (0 for background, 1 for the target). If you run segmentation, do not run this function
        
        if len(prediction) == 0:
            return [], []
        
        #print(prediction)
        #print(ground_truth)
        
        # Large enough size for base mask matrices:
        shape = 2*max(np.max(prediction), np.max(ground_truth)) 
        
        p = self.convert2mask(prediction, shape)
        gt = self.convert2mask(ground_truth, shape)
        
        return p, gt
    
    def iou_compute(self, p, gt): # Computes Intersection Over Union (IOU)
        if len(p) == 0:
            return 0
        
        intersection = np.logical_and(p, gt)
        union = np.logical_or(p, gt)
        
        iou = np.sum(intersection) / np.sum(union)
        
        return iou
    
    # Add your own metrics here, such as mAP, class-weighted accuracy, ...
    
    def isRectangleOverlap(self, R1org, R2org):
        R1 = copy.deepcopy(R1org)
        R2 = copy.deepcopy(R2org)
        a, b = self.prepare_for_detection([R1], [R2])
        iou = self.iou_compute(a, b)
        if(iou > 0):
            return True
        return False
    
    def boundigBoxesPairs(self, predictions, ground_truth, confidences, displayBoxes=False):
        
        if not len(predictions):
            predictions = []
        elif type(predictions) is np.ndarray:
            predictions = predictions.tolist()
        
        if not len(confidences):
            confidences = []
        elif type(confidences) is np.ndarray:
            confidences = confidences.tolist()
        
        groundTruthOriginal = copy.deepcopy(ground_truth)
        groundTruthOriginalCopy = copy.deepcopy(ground_truth)
        predictionsOriginal = copy.deepcopy(predictions)
        confidencesOriginal = copy.deepcopy(confidences)
        
        groundTruthSorted = []
        predictionsSorted = []
        confidencesSorted = []
        
        for ground in groundTruthOriginal:
            for predict, confid in zip(predictionsOriginal, confidencesOriginal):
                # if any corner inside of ground truth and vice versa -> pair
                if(self.isRectangleOverlap(ground, predict)):
                    groundTruthSorted.append(ground)
                    predictionsSorted.append(predict)
                    confidencesSorted.append(confid)
                    
                    groundTruthOriginalCopy.remove(ground)
                    predictionsOriginal.remove(predict)
                    confidencesOriginal.remove(confid)
                    break
        
        if displayBoxes:
            fig, ax = plt.subplots()
            for ground in groundTruthOriginal:
                ax.add_patch(Rectangle((ground[0], ground[1]), ground[2], ground[3], edgecolor = 'red', fill=False, lw=1, linestyle = 'dashed'))
                
            for predict in predictionsOriginal:
                ax.add_patch(Rectangle((predict[0], predict[1]), predict[2], predict[3], edgecolor = 'blue', fill=False, lw=1))
            
            plt.axis([0, 480, 0, 360])
            plt.show()
        
        extendZero=[0,0,0,0]
        if(len(predictionsOriginal) > 0):
            predictionsSorted.extend(predictionsOriginal)
            for a in predictionsOriginal:
                groundTruthSorted.append(extendZero)
                confidencesSorted.append(0.0)
        if(len(groundTruthOriginalCopy) > 0):
            groundTruthSorted.extend(groundTruthOriginalCopy)
            for a in groundTruthOriginalCopy:
                predictionsSorted.append(extendZero)
                confidencesSorted.append(0.0)
        
        return predictionsSorted, groundTruthSorted, confidencesSorted
    
    def TPFP_compute(self, predictionsSorted, groundTruthSorted, threshold): # Computes Average Precision (AP)
        
        # calculate IoU for every bounding box pair
        iousArr = []
        TPFP = []
        
        for predict, ground in zip( predictionsSorted, groundTruthSorted ):
            p, gt = self.prepare_for_detection([predict], [ground])
            IoUforPair = self.iou_compute(p, gt)
            iousArr.append(IoUforPair)
            
            if(IoUforPair > threshold):
                TPFP.append("TP")
            elif(predict == [0,0,0,0]):
                TPFP.append("FN")
            elif(ground == [0,0,0,0]):
                TPFP.append("FP")
        return TPFP
    
    def averagePrecision_compute(self, allTPFP, allConfidences, allBboxes):
        allAUC = []
        for key in allTPFP:
            
            combined = list(zip(allTPFP[key], allConfidences))
            sortedCombined = sorted(combined, key=lambda x: x[1], reverse=True) # sort tables by confidence (desc)
            
            # true positive
            # false positive
            # false negatives -> did not output bounding box for target bounding box
            # true negative -> we dont really have (didnt output box for box that isnt there)
            
            # precision -> true positive / all predictions -> which predictions were correct
            # precision = TP / TP + FP
            
            # recall -> (number of correctly predicted boundig boxes) true positive / true positive + false negatives -> divide total number of target bounding boxes
            # recall -> of all target bounding boxes, what fraction was corectly detected?
            # recall = TP / TP + FN
            
            precision = 0
            precisionNumerator = 0
            precisionTop = 0
            precisionArray = []
            
            recall = 0
            recallTop = 0
            recallArray = []
            
            for item in sortedCombined:
                if(item[0] == 'TP'):
                    precisionTop += 1
                # FP only bottom ++
                precisionNumerator += 1
                precision = precisionTop / precisionNumerator
                precisionArray.append(precision)
                
                if(item[0] == 'TP'):
                    recallTop += 1
                recall = recallTop / allBboxes
                recallArray.append(recall)
            
            
            
            curAuc = auc(recallArray,precisionArray)
            allAUC.append(curAuc)
        
        fig, ax = plt.subplots()
        ax.plot(recallArray, precisionArray, color='purple')
        #add axis labels to plot
        ax.set_title('Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        plt.axis([0, 1, 0, 1])
        plt.show()
        
        meanAUC = sum(allAUC) / len(allAUC)
        return meanAUC
    
    
if __name__ == '__main__':
    a = [[576, 173, 35, 40], [651, 115, 34, 49], [521, 118, 35, 55], [146, 105, 34, 61]]
    b = [[150, 98, 30, 67], [521, 117, 36, 56], [576, 174, 36, 40], [649, 114, 36, 50]]
    eval = Evaluation()
    eval.averagePrecision_compute(a, b, [0.8, 0.5, 0.6, 0.7])