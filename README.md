# Face Recognition System
This project implements a face recognition system using Python and OpenCV. The system is capable of capturing face images to create a dataset, training a face recognizer, and then using it for real-time face recognition.

## Features
- Dataset Capture: Capture face images using a webcam and save them to create a dataset for training.
- Training: Train a LBPH (Local Binary Patterns Histograms) face recognizer using the captured dataset.
- Real-time Recognition: Recognize faces in real-time using the trained model and activate a GPIO pin based on the recognized identity.
## Prerequisites
- Python 3.x
- OpenCV
- numpy
- Pillow (PIL)
- RPi.GPIO (for Raspberry Pi GPIO control)

### Installation

To install the required dependencies, run the following command:

```bash
pip3 install -r requirements.txt
```

## Usage
- Dataset Capture:
Run dataset_capture.py to capture face images. Enter a unique user ID when prompted. Look at the camera and follow instructions until the required number of images is captured.
- Training:
Run training.py to train the face recognizer using the captured dataset. The trained model will be saved as trainer.yml.
- Real-time Recognition:
Run recognizer.py to start real-time face recognition. The system will continuously capture frames from the webcam, detect faces, and recognize them using the trained model. If a recognized face is below a certain confidence threshold, the associated name will be displayed, and a GPIO pin will be activated.

## Configuration
- Adjust the number of images to capture per user in dataset_capture.py (count >= 30).
- Set the GPIO pin number in recognizer.py (relay = 23).
