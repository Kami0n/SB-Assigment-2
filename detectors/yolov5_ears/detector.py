import torch

class Detector:
	model = torch.hub.load('ultralytics/yolov5', 'custom', 'detectors/yolov5_ears/best.pt') 
	
	def detectEars(self, img):
		boxes= []
		confidences= []
		
		predictions = self.model(img)
		
		#predictions.print()  # print results to screen
		#predictions.show()  # display results
		#predictions.save()  # save as results1.jpg, results2.jpg... etc.
		results = predictions.pandas().xyxy[0].to_dict(orient="records")
		for result in results:
			con = result['confidence']
			cs = result['class']
			x1 = int(result['xmin'])
			y1 = int(result['ymin'])
			x2 = int(result['xmax'])
			y2 = int(result['ymax'])
			w = x2-x1
			h = y2-y1
			boxes.append([x1,y1,w,h ])
			confidences.append(con)
		return boxes, confidences