import cv2
import numpy as np

cap = cv2.VideoCapture("./hands.mp4")
for i in range(10):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    if i ==9:
    	break
cap.release()
cv2.destroyAllWindows() 