import cv2
import torch

# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

brands = ['renault', 'mercedes', 'volvo', 'bmw', 'volkswagen', 'ford', 'citroen', 'toyota', 'kia', 'fiat', 'honda',
          'skoda', 'plate', 'nissan', 'seat', 'peugeot', 'opel', 'audi', 'hyundai', 'rover']
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/Users/canrollas/Projects/OCR/yolov5/runs/train/exp4/weights/best.pt')

# load car.xml as Cascade model
car_model = cv2.CascadeClassifier('cars.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was read successfully, display it
    results = model(frame)
    for result in results.pred:
        for *xyxy, conf, cls in result:
            if brands[int(cls.item())] == 'plate':
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                cv2.putText(frame, 'Plate', (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
            if brands[int(cls.item())] != 'plate':
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, brands[int(cls.item())], (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

    if ret:
        # Display the frame
        cv2.imshow('Webcam', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
