# Image Face Recognition (LBPH)

A Computer Vision pipeline developed using OpenCV to perform face detection and identification on static images using Haar Feature-based Cascade Classifiers and Local Binary Patterns Histograms (LBPH).

## Features

- **Face Detection:** Leverages OpenCV's pre-trained Haar Cascade (`haarcascade_frontalface_default.xml`) to localize frontal faces.
- **Face Recognition:** Trains an LBPH recognizer (`cv2.face.LBPHFaceRecognizer_create`) on reference images to predict identity along with confidence scores.
- **Visual Output:** Annotates detected faces with green bounding boxes and predicted names directly on the target image, saving the output to `output_result.jpg`.

## Project Structure

```text
image_face_recognition/
├── faces/                          # Training images organized by identity/person
├── haarcascade_frontalface_default.xml # Haar Cascade model file
├── face_recognition.py             # Training and evaluation pipeline
├── test_image.jpeg                 # Target test image for recognition
├── output_result.jpg               # Annotated output result
└── .gitignore
