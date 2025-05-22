import cv2
import dlib
import face_recognition
import os
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load the passport image and extract face encoding
passport_image_path = r'C:\Users\ANIRUDH\Downloads\WhatsApp Image 2025-05-12 at 10.10.25_a90097cd.jpg'  # Updated path to passport image
passport_image = face_recognition.load_image_file(passport_image_path)
passport_image_encodings = face_recognition.face_encodings(passport_image)

if not passport_image_encodings:
    raise ValueError("No faces found in the passport image.")
    
passport_image_encoding = passport_image_encodings[0]

# Flag to check if a match is found
flag = False

# Create directories to store detected face images and encodings
output_dir = 'faces'
encoding_dir = 'face_encodings'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(encoding_dir):
    os.makedirs(encoding_dir)

# Face ID counter for naming detected faces
face_id = 0

while True:
    ret, frame = cap.read()  # Capture each frame
    if not ret:
        break
    
    # Convert the frame to grayscale (used by dlib for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the current frame
    faces = detector(gray)

    # Iterate over all detected faces in the frame
    for face in faces:
        # Get the coordinates of the face
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the region of interest (ROI) for the detected face
        face_roi = frame[y:y + h, x:x + w]
        
        # Save the detected face image
        face_filename = os.path.join(output_dir, f"face_{face_id}.jpg")
        cv2.imwrite(face_filename, face_roi)

        # Extract the face encoding for the detected face
        face_encodings = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])

        if face_encodings:
            face_encoding = face_encodings[0]
            
            # Save the encoding of the detected face
            encoding_filename = os.path.join(encoding_dir, f"face_{face_id}.npy")
            np.save(encoding_filename, face_encoding)
            
            # Compare the face encoding with the passport image encoding
            matches = face_recognition.compare_faces([passport_image_encoding], face_encoding)
            
            if matches[0]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                flag = True  # Match found
                print(f"Match found: Face ID {face_id} matches the passport image.")
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # No match
                print(f"No match: Face ID {face_id} does not match the passport image.")

        face_id += 1  # Increment face ID for the next detected face
    
    # Show the frame with rectangles drawn around the faces
    cv2.imshow('Video Frame', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop, check if a match was found and display the result
if flag:
    print("Video Person is the Same as Passport Size Person")
else:
    print("Different Person")

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
