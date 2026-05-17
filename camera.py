import cv2
#LOAD PRE_TRAINED FACE DETECTION MODEL
face_cascade=cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
)
#START WEBCAM
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    #convert to grayscale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #DETECT FACES
    faces=face_cascade.detectMultiScale(
        gray,scaleFactor=1.3,
        minNeighbors=5
    )
    #DRAW RECTANGLE AROUND FACES
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #SHOW OUTPUT
    cv2.imshow("FACE DETECTION",frame)
    #press 'q' to exit
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
