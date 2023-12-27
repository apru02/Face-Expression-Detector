# Face Detection and Emotion Recognition

This Python script uses computer vision and deep learning techniques to perform real-time face detection and emotion recognition using a convolutional neural network (CNN) model.

## Requirements

Make sure you have the following libraries installed:

- NumPy
- Matplotlib
- OpenCV (cv2)
- TensorFlow
- PyWavelets (pywt)

You can install these dependencies using the following:

```bash
pip install numpy matplotlib opencv-python tensorflow pywavelets
```
## Usage
1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```
2. Run the face_detect.py script:
   ```bash
   python face_detect.py
   ```
The script will open your webcam and perform face detection and emotion recognition in real-time.

### Emotion Classes
The model classifies emotions into the following classes:
- Angry
- Fear
- Happy
- Neutral
- Sad
- Surprise

Additional Notes:
The emotion prediction is displayed in real-time on the webcam feed.
Press 'q' to exit the application.

- Download the training and validation dataset from [here](https://www.kaggle.com/datasets/jangedoo/utkface-new)

