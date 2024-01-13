import concurrent.futures
import os
import re
import threading
import cv2
import numpy as np
import torch
from sort import Sort
from queue import Queue
import pytesseract
import logging

# Constants
CAMERA_INDEX = 0
PLATE_MODEL_PATH = '/Users/canrollas/Projects/OCR/yolov5/runs/train/exp4/weights/best.pt'
PLATES_FOLDER = 'plates'

# Color Codes
GREEN = (0, 255, 0)

def apply_filters(image):
    # Apply filters or preprocessing steps as needed
    # Example: Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return blurred

def find_edges(image):
    # Find edges using Canny edge detector
    edges = cv2.Canny(image, 50, 150)

    return edges

def find_contours(image):
    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def apply_mask(image, mask):
    # Apply mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image

def create_blank_mask(image):
    # Create a blank mask with the same size as the input image
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    return mask

def draw_contours_on_mask(mask, contours):
    # Draw contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    return mask

def initialize_sort_tracker():
    return Sort()

def detect_objects(model, frame, class_id):
    results = model(frame)
    detections = []

    for result in results.pred:
        for *xyxy, conf, cls in result:
            if int(cls.item()) == class_id:
                detections.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), 0])

    return detections

def text_validate_plate(text):
    # Turkish license plate format: 2 or 3 letters, 2 or 3 digits without space
    turkish_plate_pattern = re.compile(r'^\d{2}[A-Z]{2,3}\d{2,3}$')

    # Check if the text matches the Turkish license plate pattern
    if turkish_plate_pattern.match(text):
        # return the matched text as the plate string
        return turkish_plate_pattern.match(text).group()
    else:
        return "Not found"

def process_plate_extraction(cap, plate_model, sort_tracker, frame_queue):
    tracked_ids = set()
    tracked_ids_map = {}
    tracked_ids_validation = {}

    while True:
        ret, frame = cap.read()

        if not ret:
            logging.error("Error reading frame from the camera.")
            break

        # Apply filters and find edges
        processed_frame = apply_filters(frame)
        edges = find_edges(processed_frame)

        # Find contours
        contours = find_contours(edges)

        # Create a blank mask and draw contours on it
        mask = create_blank_mask(frame)
        mask_with_contours = draw_contours_on_mask(mask, contours)

        # Apply the mask to the original frame
        result_frame = apply_mask(frame, mask_with_contours)

        # Detect plates using YOLO model
        detections = detect_objects(plate_model, result_frame, class_id=12)

        if not detections:
            frame_queue.put(frame)
            continue

        try:
            trackers = sort_tracker.update(np.array(detections))
        except Exception as e:
            logging.error("Error updating plate tracker:", exc_info=True)
            trackers = []

        for d in trackers:
            left, top, right, bottom, track_id = map(int, d)

            # Check if the object is not already tracked
            if track_id not in tracked_ids or track_id not in tracked_ids_map:
                roi = frame[top:bottom, left:right]  # Extract ROI
                # Show the particular frame
                text = extract_text_from_roi(roi)

                if text_validate_plate(text) != "Not found":
                    print(f"Text from Track ID {track_id}: {text_validate_plate(text)}")
                    tracked_ids_map[track_id] = text_validate_plate(text)
                    tracked_ids_validation[track_id] = 0

                    # Save the image to the disk in the plates folder
                    save_plate_image(roi, text)

                tracked_ids.add(track_id)
            else:
                tracked_ids_validation[track_id] += 1
                if tracked_ids_validation[track_id] == 5:
                    # Validate the extracted text from the plate image and update the map
                    roi = frame[top:bottom, left:right]  # Extract ROI
                    # Show the particular frame
                    text = extract_text_from_roi(roi)
                    if text_validate_plate(text) != "Not found":
                        # Update the map
                        tracked_ids_map[track_id] = text_validate_plate(text)
                        tracked_ids_validation[track_id] = 0

            draw_rectangle_and_text(frame, left, top, right, bottom, track_id, tracked_ids_map)

        frame_queue.put(frame)

    cap.release()

def extract_text_from_roi(roi):
    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance contrast
    _, thresholded_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform OCR using Tesseract with custom configurations 2 digits space 2 or 3 letters space 2 or 3 4 digits
    custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    text = pytesseract.image_to_string(thresholded_roi, config=custom_config)
    return text

def draw_rectangle_and_text(frame, left, top, right, bottom, track_id, tracked_ids_map):
    cv2.rectangle(frame, (left, top), (right, bottom), GREEN, 2)
    if track_id in tracked_ids_map:
        cv2.putText(frame, f"Plate: {tracked_ids_map[track_id]}", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Plate", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA)

def save_plate_image(roi, text):
    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance contrast
    _, thresholded_roi = cv2.threshold(gray_roi, 25, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask
    mask = np.zeros_like(gray_roi)

    # Extract the biggest white area as the plate
    contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the thresholded ROI
    thresholded_roi = cv2.bitwise_and(thresholded_roi, thresholded_roi, mask=mask)

    # Save the image to the disk in the plates folder
    if not os.path.exists(PLATES_FOLDER):
        os.makedirs(PLATES_FOLDER)
    cv2.imwrite(f"{PLATES_FOLDER}/{text_validate_plate(text)}.jpg", thresholded_roi)

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        logging.error("Error: Could not open camera.")
        exit()

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=PLATE_MODEL_PATH)

    sort_tracker = initialize_sort_tracker()
    frame_queue = Queue()

    # Create a separate thread for processing frames
    process_thread = threading.Thread(target=process_plate_extraction, args=(cap, model, sort_tracker, frame_queue))
    process_thread.start()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            frame = frame_queue.get()

            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    process_thread.join()  # Wait for the processing thread to finish
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Keyboard Interrupted")
        exit(0)
