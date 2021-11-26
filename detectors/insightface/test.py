
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
img = cv2.imread("./t1.jpg")
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)

listFaces = []
for items in faces:
    coordinateList = items.bbox.tolist()
    A = [int(a) for a in coordinateList]
    listFaces.append(A)

print(listFaces)