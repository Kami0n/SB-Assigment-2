import cv2, numpy as np
import os

# import image module
from PIL import Image
from PIL import ImageFilter


class Preprocess:
    
    def allPreprocessing(self, img):
        
        #img = self.toGreyScale(img)
        #img = self.imgNormalize(img)
        
        img = self.histogram_equlization_rgb(img) # This one makes VJ worse
        #img = self.histogramEqualization(img)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        #img = self.edgeEnhancement(img)
        
        return img
    
    def histogram_equlization_rgb(self, img):
        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)
        
        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)
        
        return img
    
    # Add your own preprocessing techniques here.
    def histogramEqualization(self, img):
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        img2 = cdf[img]
        
        return img2
    
    def toGreyScale(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def imgNormalize(self, img):
        norm_img = np.zeros((img.shape[0], img.shape[1]))
        norm_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
        return norm_img
    
    def edgeEnhancement(self, img):
        imageObject = Image.fromarray(img)
        edgeEnahnced = imageObject.filter(ImageFilter.EDGE_ENHANCE)
        moreEdgeEnahnced = imageObject.filter(ImageFilter.EDGE_ENHANCE_MORE)
        open_cv_image = np.array(moreEdgeEnahnced)
        return open_cv_image
    
    
    
    
    

if __name__ == '__main__':
    os.system('cls')
    pre = Preprocess()
    img = cv2.imread("data/faces/wider_award/images/16_Award_Ceremony_Awards_Ceremony_16_419.jpg")
    editedImg = pre.allPreprocessing(img)
    cv2.imwrite("./out.jpg", editedImg)