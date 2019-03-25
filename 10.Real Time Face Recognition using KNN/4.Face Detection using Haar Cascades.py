import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("../Datasets/haarcascade_frontalface_alt.xml")

while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue
    # @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
    # @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have
    # to retain it.
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    # faces -> array of tuples(x_left, y_top, width, height)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Video Frame", frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if(key_pressed==ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()

    