import re
from difflib import SequenceMatcher

import cv2
import imutils
import torch
from vidgear.gears import CamGear
from queue import Queue

from threading import Thread, Lock
import pytesseract

stream = CamGear(source='https://www.youtube.com/watch?v=2pRTWZXuoN8', stream_mode=True, logging=True).start()

brands = ['renault', 'mercedes', 'volvo', 'bmw', 'volkswagen', 'ford', 'citroen', 'toyota', 'kia', 'fiat', 'honda',
          'skoda', 'plate', 'nissan', 'seat', 'peugeot', 'opel', 'audi', 'hyundai', 'rover']
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/Users/canrollas/Projects/OCR/exp4/weights/best.pt')  # custom model

brand_model = torch.hub.load('ultralytics/yolov5', 'custom',
                             path='/Users/canrollas/Projects/OCR/brands.pt')  # custom model

white_list = ["34ACN264","66ACE441","16ASG040","06CCD368","28YH184"]

# Use threading and Lock
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

    # Rotate and find the best match
    CONFIG = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    crop_img = resized_roi[0:resized_roi.shape[0], 0:resized_roi.shape[1] - 10]

    # rotate the image -15 degreees to 15 degrees with 1 degree increments
    for i in range(-30, 30, 3):
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
def process_frames():
    while True:
        frames = stream.read()
        results = model(frames)
        brand_results = brand_model(frames)
        print(results.pred)

        # Draw bounding boxes and labels of detections per each frame in the video
        for result in results.pred:
            for *xyxy, conf, cls in result:
                if brands[int(cls.item())] == 'plate':
                    # Draw bounding box and label for the plate
                    text_img = frames[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    text = extract_text(text_img)
                    print("Text is: ", text)
                    # remove all non-alphanumeric characters from the text
                    text = re.sub('[^A-Za-z0-9]+', '', text)
                    better_text = find_similar_text(text)
                    if better_text != "":
                        text = better_text
                    if text != "Not found":
                        cv2.rectangle(frames, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                                      (0, 255, 0), 2)
                        cv2.putText(frames, 'Plate:' + text, (int(xyxy[0]), int(xyxy[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2,
                                    cv2.LINE_AA)
                    else:
                        cv2.rectangle(frames, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                                      (0, 0, 255), 2)
                        cv2.putText(frames, 'Plate:' + "Camera Position error not extracted",
                                    (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2,
                                    cv2.LINE_AA)

        # Display output
        frames = cv2.resize(frames, (640, 480))  # Adjust the size as needed
        frame_queue.put(frames)


# Start the processing thread
process_thread = Thread(target=process_frames)
process_thread.start()


while True:
    frames = frame_queue.get()
    cv2.imshow('Car Detection', frames)

    if cv2.waitKey(1) == ord('q'):
        break

    cv2.destroyAllWindows()





# Wait for both threads to finish
process_thread.join()
