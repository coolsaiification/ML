# Read video from camera frame by frame
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret == False:
        continue
    
    cv2.imshow("Video Frame", frame)

    # Wait for user input-'q', then stop the loop
    # 1 -> millisecond
    # 0xff -> for 8 bits
    key_pressed = cv2.waitKey(1) & 0xFF
    # ord -> gives ASCII
    if key_pressed == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()