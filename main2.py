import logging

import cv2
import numpy as np
import torch
from sort import Sort


def initialize_sort_tracker():
    return Sort()


def detect_plates(model, frame):
    results = model(frame)
    detections = []

    for result in results.pred:
        for *xyxy, conf, cls in result:
            if int(cls.item()) == 12:
                detections.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), 0])

    return detections


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='/Users/canrollas/Projects/OCR/yolov5/runs/train/exp4/weights/best.pt')

    sort_tracker = initialize_sort_tracker()
    sort_tracker_car = initialize_sort_tracker()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        detections = detect_plates(model, frame)

        try:
            trackers = sort_tracker.update(np.array(detections))
        except Exception as e:
            logging.error("Error updating plate tracker:", exc_info=True)
            trackers = []

        for d in trackers:
            left, top, right, bottom, track_id = map(int, d)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, str("Plate"), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
