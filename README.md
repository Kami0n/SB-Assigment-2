# Code for Assigment no. 2 for subject Image-based Biometry

This is my code for second assigment for subject Image-based Biometry.

## Goals of second assigment
In this assigment we will be learning about first three steps in biometric recognition pipeline:
1. data acquisition
2. pre-processing
3. detection (segmentation)

## What did I do?

0. tried to run VJ detector for ears, to get the feeling of how this code framework works
1. selected dataset WIDER with annotated faces and from this dataset I selected 181 examples (awards)
2. programmed mAP (mean average precision) metric
3. evaluated VJ haar cascade
3. evaluated insightface
4. evaluated DSFD-Pytorch-Inference
5. evaluated yoloface

### Evaluated detectors:
#### Results:

- VJ haar cascade
  - IoU: 32,44 %
  - mAP: cant get confidences!
- deepinsight / insightface -> https://github.com/deepinsight/insightface
  - IoU: 87,29 %
  - mAP: cant get confidences!
- hukkelas / DSFD-Pytorch-Inference -> https://github.com/hukkelas/DSFD-Pytorch-Inference
  - IoU: 85,34 %
  - mAP: 0.57198
- sthanhng / yoloface -> https://github.com/sthanhng/yoloface
  - IoU: 79,01 %
  - mAP: 0.57183
- YYuanAnyVision / mxnet_mtcnn_face_detection -> https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection
  - IoU: 73,50 %
  - mAP: 0.38264

