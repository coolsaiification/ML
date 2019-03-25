# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

# Init camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('../../Datasets/haarcascade_frontalface_alt.xml')

# Save every 10th image
skip = 0
face_data = []
dataset_path = "./data/"

file_name = input("Enter your name")

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if ret==False:
        continue
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    # Sort ascending using area (width * height)
    faces = sorted(faces, key=lambda face:face[2]*face[3])

    # Pick only the last face(max area)
    for (x,y,w,h) in faces[-1:]:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        
        skip += 1
        # Add face data once in 10 iterations
        if skip%10 != 0:
            continue

        # Add padding
        offset = 10
        # Using gray image to save space
        # y , x
        face_section = gray_frame[y-offset:y+h+offset, x-offset:x+w+offset]
        # Resize in 100x100 grid
        face_section = cv2.resize(face_section,(100,100))

        face_data.append(face_section)
        print(len(face_data))

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Convert face list into numpy array
face_data = np.asarray(face_data)
print(face_data.shape)
face_data = face_data.reshape((face_data.shape[0],-1))
# 10000 columns = 100x100 for each face
# For frame (colored) it will be 3x10000 as 1 layer to each RGB
print(face_data.shape)

# Save data into file system
np.save(dataset_path+file_name+".npy", face_data)
print("Data successfully saved at "+dataset_path+file_name+".npy")

cap.release()
cv2.destroyAllWindows()