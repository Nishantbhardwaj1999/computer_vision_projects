import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import cvzone

# Load the video file
cap = cv2.VideoCapture(0)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

# Initialize the face mesh detector
detector = FaceMeshDetector(maxFaces=1)

#eye detection from edge
idList=[22,23,24,26,110,157,158,159,160,161,130,243]

ratioList=[]


blinkCounter=0
counter=0
color=(255,0,255)

plotY=LivePlot(640,360,[20,50])

# Main loop to process each frame of the video
while True:
    # Read a frame from the video
    success, img = cap.read()

    # Check if the frame is read successfully
    if not success:
        print("Error: Failed to read a frame.")
        break

    # Find face mesh in the frame
    img, faces = detector.findFaceMesh(img,draw=False)

    if faces:
        face=faces[0]
        for id in idList:
            cv2.circle(img,face[id],5,color,cv2.FILLED)

        leftUP=face[159]
        leftDown=face[23]
        leftleft=face[130]
        leftright=face[243]
        lengthVer,_=detector.findDistance(leftUP,leftDown)
        lengthHor,_=detector.findDistance(leftleft,leftright)
        cv2.line(img,leftUP,leftDown,(0,200,0),3)
        #print(lengthVer)
        ratio=int((lengthVer/lengthHor)*100)
        ratioList.append(ratio)
        if len(ratioList)>10:
            ratioList.pop(0)
        ratioAvg=sum(ratioList)/len(ratioList)
        if ratioAvg<35 and counter ==0:
            blinkCounter +=1
            color=(0,200,0)
            counter=1
        if counter !=0:
            counter+=1
        if counter>10:
            counter=0
            color=(255,0,255)
        cvzone.putTextRect(img,f'Blink Count: {blinkCounter}', (50,100),colorR=color)
        imgPlot=plotY.update(ratioAvg,color)
        # Resize the frame for display
        img = cv2.resize(img, (640, 360))
        imgStack=cvzone.stackImages([img,imgPlot],2,1)

    else:
         img = cv2.resize(img, (640, 360))
         imgStack=cvzone.stackImages([img,img],2,1)

        

    
    # Display the frame
    cv2.imshow("Image", imgStack)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
