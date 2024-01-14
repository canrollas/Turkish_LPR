import cv2
from concurrent.futures import ThreadPoolExecutor

import re
import time
from difflib import SequenceMatcher

import imutils
import torch
from queue import Queue

from threading import Thread, Lock
import pytesseract

start_time = time.time()
fps_counter = 0

video_path = 'temp1.mp4'  # Update with the path to your video file
cap = cv2.VideoCapture(video_path)

brands = ['renault', 'mercedes', 'volvo', 'bmw', 'volkswagen', 'ford', 'citroen', 'toyota', 'kia', 'fiat', 'honda',
          'skoda', 'plate', 'nissan', 'seat', 'peugeot', 'opel', 'audi', 'hyundai', 'rover']
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/canrollas/Projects/OCR/exp4/weights/best.pt')
brand_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/canrollas/Projects/OCR/brands.pt')

white_list = ["34ACN264", "66ACE441", "16ASG040", "06CCD368", "28YH184"]

lock = Lock()
text_result = None
frame_queue = Queue()

def find_similar_text(text):
    for i in white_list:
        sim_measure = SequenceMatcher(None, text, i).ratio()
        if sim_measure > 0.8:
            return i
    return ""

def extract_text(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresholded_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    resized_roi = cv2.resize(thresholded_roi, (0, 0), fx=0.5, fy=0.5)

    CONFIG = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    crop_img = resized_roi[0:resized_roi.shape[0], 0:resized_roi.shape[1] - 10]

    for i in range(-15, 15, 3):
        rotated_roi = imutils.rotate(crop_img, i)
        text = pytesseract.image_to_string(rotated_roi, config=CONFIG)
        if text_validate_plate(text) != "Not found":
            return text
    return "Not found"

def text_validate_plate(text):
    turkish_plate_pattern1 = re.compile(r'^\d{2}[A-Z]{2,3}\d{2,3}$')
    turkish_plate_pattern2 = re.compile(r'^\d{2}[A-Z]{2,3}\d{2,4}$')
    if turkish_plate_pattern1.match(text):
        return turkish_plate_pattern1.match(text).group()
    elif turkish_plate_pattern2.match(text):
        return turkish_plate_pattern2.match(text).group()
    else:
        return "Not found"

def process_frame(frame):
    global fps_counter, start_time, fps

    results = model(frame)
    for result in results.pred:
        for *xyxy, conf, cls in result:
            if brands[int(cls.item())] == 'plate':
                text_img = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                text = extract_text(text_img)
                text = re.sub('[^A-Za-z0-9]+', '', text)
                better_text = find_similar_text(text)
                if better_text != "":
                    text = better_text
                if text != "Not found":
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, 'Plate:' + text, (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
                    cv2.putText(frame, 'Plate:' + "Camera Position error not extracted",
                                (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
    fps_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = fps_counter / elapsed_time
        fps_counter = 0
        start_time = time.time()

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    frame = cv2.resize(frame, (640, 480))

    return frame

with ThreadPoolExecutor() as executor:
    while True:
        ret, frames = cap.read()
        if not ret:
            break

        processed_frames = list(executor.map(process_frame, [frames]))

        for processed_frame in processed_frames:
            cv2.imshow('Car Detection', processed_frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
