import cv2
import dlib
import numpy as np

# Load the face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\ANIRUDH\Downloads\shape_predictor_68_face_landmarks.dat")

# Function to calculate the mouth aspect ratio (MAR)
def get_mouth_aspect_ratio(mouth_points):
    A = np.linalg.norm(mouth_points[2] - mouth_points[9])  # Vertical distance
    B = np.linalg.norm(mouth_points[4] - mouth_points[7])  # Vertical distance
    C = np.linalg.norm(mouth_points[0] - mouth_points[6])  # Horizontal distance
    mar = (A + B) / (2.0 * C)
    return mar

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (dlib expects RGB, not BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    faces = detector(rgb_frame)

    for face in faces:
        landmarks = predictor(rgb_frame, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Mouth points (68 landmarks, mouth is 48 to 67)
        mouth_points = landmarks[48:68]
        
        # Calculate MAR
        mar = get_mouth_aspect_ratio(mouth_points)
        
        # Draw circles on the mouth points
        for (x, y) in mouth_points:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Check if MAR indicates talking with a more sensitive threshold (e.g., 0.4)
        if mar > 0.4:  # Lowered threshold for more sensitivity
            cv2.putText(frame, "Talking", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            count = 1
        else:
            cv2.putText(frame, "Not Talking", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count == 1:
        break

cap.release()
cv2.destroyAllWindows()

# Print the status based on MAR
if count == 1:
    print("Student is Talking")
else:
    print("Student is Not Talking")
