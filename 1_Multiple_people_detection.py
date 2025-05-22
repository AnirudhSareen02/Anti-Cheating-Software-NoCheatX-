import cv2 as cv

haar_cascade = cv.CascadeClassifier(r"C:\Users\ANIRUDH\Documents\Projects\Anti_Cheating_Software\haar_face.xml")

img = cv.VideoCapture(0)
flag = False

while True:
    istrue, frame = img.read()

    if not istrue:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors= 5, minSize= (30,30))

    for(x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    num_faces = len(faces)
    if num_faces >=2:
        flag = True
        break
    else:
        flag = False

    cv.imshow("Video", frame)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

img.release()
cv.destroyAllWindows()
if flag == True:
    print("More than 1 student")
else:
    print("1 student")