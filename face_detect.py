import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pywt

new_model = tf.keras.models.load_model('face_model.keras')

# print(new_model.summary())
def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
video_capture = cv2.VideoCapture(0)
roi_color = []
last_emotion = ""
face_dict= {
    0 : 'Angry',
    1 : 'Fear',
    2 : 'Happy',
    3 : 'Neutral',
    4 : 'Sad',
    5 : 'Surprise'
}
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Draw the rectangle around each face
    X = 0
    Y = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]
        X = x
        Y = y
    
    if len(roi_color) != 0:
        roi_color = cv2.resize(roi_color, (48, 48), interpolation=cv2.INTER_CUBIC)
        roi_color = w2d(roi_color, 'db2', 5)
        roi_color = np.array(roi_color).reshape(-1,48, 48,1)/255.0 
        y = new_model.predict(roi_color)
        y_pred = np.argmax(tf.nn.softmax(y[0]))
        last_image = roi_color
        emotion = face_dict[y_pred]
        
        # For saving images uncomment the following lines
        # emotion_dir = "new_train/" + emotion
        # cv2.imwrite(emotion_dir + "/" + str(len(os.listdir(emotion_dir))) + ".jpg", last_image[0])

        cv2.putText(frame, emotion, (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
    # Display
    cv2.imshow("Video", frame)
    roi_color = []
    # Exit when escape is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


