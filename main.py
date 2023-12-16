import cv2
import torch
from vidgear.gears import CamGear

stream = CamGear(source='https://www.youtube.com/watch?v=lfYYtQ1Ah04', stream_mode=True, logging=True).start()

brands = ['renault', 'mercedes', 'volvo', 'bmw', 'volkswagen', 'ford', 'citroen', 'toyota', 'kia', 'fiat', 'honda',
          'skoda', 'plate', 'nissan', 'seat', 'peugeot', 'opel', 'audi', 'hyundai', 'rover']
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/Users/canrollas/Projects/OCR/yolov5/runs/train/exp4/weights/best.pt')  # custom model

# Neden efektif degil ?

while True:
    # Capture frames
    frames = stream.read()
    results = model(frames)
    print(results.pred)

    # Draw bounding boxes and labels of detections per each frame in the video
    for result in results.pred:
        for *xyxy, conf, cls in result:
            if brands[int(cls.item())] == 'plate':
                # Draw bounding box and label for the plate
                cv2.rectangle(frames, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                cv2.putText(frames, 'Plate', (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)
            else:
                # Draw bounding box and label for the brand
                cv2.rectangle(frames, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frames, brands[int(cls.item())], (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

    # Display output
    frames = cv2.resize(frames, (640, 480))  # Adjust the size as needed
    cv2.imshow('Car Detection', frames)

    if cv2.waitKey(1) == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()
