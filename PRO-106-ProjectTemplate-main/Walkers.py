import cv2


# Create our body classifier
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')


# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    bodies= body_cascade.detectMultiScale(cap)


    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(cap,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('img',cap) 

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break


    cap.release()
    cv2.destroyAllWindows()
