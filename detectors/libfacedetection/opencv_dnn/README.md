# Deploy libfacedetection with OpenCV

Example to deploy libfacedetection with the OpenCV's FaceDetectorYN in both Python and C++.

***Important Notes***:
- Install OpenCV >= 4.5.4 to have the API `FaceDetectorYN`.
- Download the ONNX model from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet).

Envrionment tested:
- System: Ubuntu 18.04 LTS / 20.04 LTS
- OpenCV >= 4.5.4
- Python >= 3.6

## Python
1. Install `numpy` and `opencv-python`.
    ```shell
    pip install numpy
    pip install "opencv-python>=4.5.4.58"
    ```
2. Run demo. For more options, run `python python/detect.py --help`.
    ```shell
    # detect on an image
    python python/detect.py --model=/path/to/yunet.onnx --input=/path/to/example/image
    # detect on default camera
    python python/detect.py --model=/path/to/yunet.onnx
    ```
    
python detectors/libfacedetection/opencv_dnn/python/detect.py --model=detectors/libfacedetection/face_detection_yunet_2021sep.onnx --input=data/faces/one_award/images/16_Award_Ceremony_Awards_Ceremony_16_031.jpg
    