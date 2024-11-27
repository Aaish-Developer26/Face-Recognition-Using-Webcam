# keras-facenet
Facenet implementation by Keras2

## Pretrained model
You can quickly start facenet with pretrained Keras model (trained by MS-Celeb-1M dataset).
- Download model from [here](https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn) and save it in model/keras/


You can also create Keras model from pretrained tensorflow model.
- Download model from [here](https://github.com/davidsandberg/facenet) and save it in model/tf/
- Convert model for Keras in [tf_to_keras.ipynb](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb)


## Demo
- [Face vector calculation](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/demo-images.ipynb)
- [Classification with SVM](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/demo-svm.ipynb)
- [Web camera demo](https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/demo-webcam.ipynb)

## Environments
Ubuntu16.04 or Windows10  
python3.6.2  
tensorflow: 1.3.0  
keras: 2.1.2  

## 1. Importing Libraries
The program uses the following libraries:

1. OpenCV (cv2): For video capture and drawing bounding boxes.
2. face_recognition: For face encoding and identification.
3. os: To access and iterate through files in the known_faces directory.
4. mediapipe: For face detection using MediaPipe's FaceDetection module.

## 2. Capturing Video From Webcam
cap = cv2.VideoCapture(0)
cv2.VideoCapture(0) initializes the webcam for capturing live video.
Frames from the webcam will be processed in real-time.

## Face Detection Using MediaPipe
MediaPipe's FaceDetection detects faces in the frame with a minimum confidence threshold of 0.5.
The webcam captures a frame (frame), and it is converted to RGB because MediaPipe and face_recognition work with RGB images.

## Releasing Resources
cap.release()
cv2.destroyAllWindows()
The webcam is released, and all OpenCV windows are closed.
