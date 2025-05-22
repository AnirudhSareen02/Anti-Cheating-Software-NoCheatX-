from inference_sdk import InferenceHTTPClient
import cv2 as cv

# Initialize the RoboFlow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="8G2COb6cc9JcTO6vsmKq"
)

cap = cv.VideoCapture(1)

def resize_frame(frame, target_size=(640, 480)):
    return cv.resize(frame, target_size)

while True:
    istrue, frame = cap.read()
    if not istrue:
        break

    resized_frame = resize_frame(frame)

    # Send the resized frame to the model for inference
    result = CLIENT.infer(resized_frame, model_id="mobile-phone-detection-mtsje/1")

    # Checking if any phones were detected
    if len(result['predictions']) == 0:
        print("No phone detected")
    else:
        print("Phone detected")
        for pred in result['predictions']:
            # Get center x, center y, width, and height
            center_x = pred['x'] * frame.shape[1] / resized_frame.shape[1]
            center_y = pred['y'] * frame.shape[0] / resized_frame.shape[0]
            width = pred['width'] * frame.shape[1] / resized_frame.shape[1]
            height = pred['height'] * frame.shape[0] / resized_frame.shape[0]

            # Calculate top-left and bottom-right points
            x_min = int(center_x - width / 2)
            y_min = int(center_y - height / 2)
            x_max = int(center_x + width / 2)
            y_max = int(center_y + height / 2)

            # Draw rectangle around detected phone
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            label = f"{pred['class']} {pred['confidence']*100:.1f}%"
            cv.putText(frame, label, (x_min, y_min - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv.imshow('Phone Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()