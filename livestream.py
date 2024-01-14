import os
import re
import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QListWidget, QHBoxLayout
from queue import Queue
from threading import Thread
import torch
from pytesseract import pytesseract
from vidgear.gears import CamGear

from sort import Sort
from yolov5.utils.torch_utils import select_device


class PlateRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cap = CamGear(source='https://www.youtube.com/watch?v=lfYYtQ1Ah04', stream_mode=True, logging=True).start()
        self.device = select_device('')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='/Users/canrollas/Projects/OCR/exp4/weights/best.pt')
        self.vehicle_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                            path='/Users/canrollas/Projects/OCR/brands.pt')

        self.sort_tracker = Sort()
        self.vehicle_sort_tracker = Sort()
        self.frame_queue = Queue()
        self.registered_plates = set()
        self.unknown_plate = dict()
        self.dropdown_label = QLabel("Registered Plates:")
        self.detected_plates_label = QLabel("Detected Plates:")
        self.detected_plates = QListWidget()
        self.dropdown_widget = QListWidget()
        self.registered_plates.add("06COG401")
        self.registered_plates.add("06COG402")
        self.registered_plates.add("06COG403")
        self.detected_pixels1 = QLabel("Detected Plate Text: None")
        self.detected_pixels2 = QLabel("Detected Plate Image:None")
        for plate in self.registered_plates:
            self.dropdown_widget.addItem(plate)
        self.init_ui()
        self.init_threads()

    def init_ui(self):
        self.central_widget = QWidget()
        self.central_widget.setLayout(QHBoxLayout())
        self.setCentralWidget(self.central_widget)
        inner_layout = QVBoxLayout()
        inner_layout.addWidget(self.dropdown_label)
        inner_layout.addWidget(self.dropdown_widget)
        inner_layout.addWidget(self.detected_plates_label)
        inner_layout.addWidget(self.detected_plates)
        inner_layout.addWidget(self.detected_pixels1)
        inner_layout.addWidget(self.detected_pixels2)
        self.central_widget.layout().addLayout(inner_layout)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.central_widget.layout().addWidget(self.video_label)
        self.video_label.setScaledContents(True)
        self.brand_labels = ['mercedes', 'peugeot', 'renault', 'toyota', 'volkswagen']
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.setWindowTitle("Plate Recognition App")

    def init_threads(self):
        process_thread = Thread(target=self.process_plate_extraction)
        process_thread.start()

    def update_frame(self):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            pixmap = QPixmap.fromImage(q_image)

            self.video_label.setPixmap(pixmap)


    def draw_vehicle_rectangle_and_text(self, frame, left, top, right, bottom, class_id):
        # Draw the bounding box and the text on the frame is Vehicle
        GREEN = (255, 0, 0)
        # find the brand name 0: mercedes, 1: peugeot, 2: renault, 3: toyota, 4: volkswagen
        if class_id < 0 or class_id > 4:
            brand_name = "Unknown"
        else:
            brand_name = self.brand_labels[class_id]
        cv2.rectangle(frame, (left, top), (right, bottom), GREEN, 2)
        cv2.putText(frame, f"Vehicle: {brand_name}", (left - 20, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2,
                    cv2.LINE_AA)

    def process_plate_extraction(self):
        tracked_ids = set()
        tracked_ids_map = {}
        brand_tracktion_map = {}
        tracked_ids_validation = {}

        while True:
            frame = self.cap.read()



            # Apply filters and find edges
            processed_frame = self.apply_filters(frame)
            edges = self.find_edges(processed_frame)

            # Find contours
            contours = self.find_contours(edges)

            # Create a blank mask and draw contours on it
            mask = self.create_blank_mask(frame)
            mask_with_contours = self.draw_contours_on_mask(mask, contours)

            # Apply the mask to the original frame
            result_frame = self.apply_mask(frame, mask_with_contours)

            # Detect plates using YOLO model
            detections = self.detect_objects(result_frame, class_id=12)

            brand_detections = self.detect_brand(frame, model=self.vehicle_model,track_map=brand_tracktion_map)
            if not detections:
                self.frame_queue.put(frame)
                continue

            try:
                trackers = self.sort_tracker.update(np.array(detections))
            except Exception as e:
                print("Error updating plate tracker:", e)
                trackers = []

            try:
                if brand_detections:
                    vehicle_trackers = self.vehicle_sort_tracker.update(np.array(brand_detections))
                else:
                    vehicle_trackers = []
            except Exception as e:
                print("Error updating vehicle tracker:", e)
                vehicle_trackers = []

            for d in trackers:
                left, top, right, bottom, track_id = map(int, d)
                # Check if the object is not already tracked
                if track_id not in tracked_ids or track_id not in tracked_ids_map:
                    roi = frame[top:bottom, left:right]  # Extract ROI
                    # Show the particular frame
                    text = self.extract_text_from_roi(roi)

                    if self.text_validate_plate(text) != "Not found":
                        print(f"Text from Track ID {track_id}: {self.text_validate_plate(text)}")
                        tracked_ids_map[track_id] = self.text_validate_plate(text)
                        tracked_ids_validation[track_id] = 0

                        # Save the image to the disk in the plates folder
                        self.save_plate_image(roi, text)
                        # show on a pixmap live feed
                        self.save_normal_plate_image(roi, text)
                        self.detected_pixels1.setText(f"Detected Plate: {text}")
                        self.detected_pixels2.setPixmap(QPixmap(f"normal_plates/{text}.jpg"))

                        self.detected_plates.addItem(text)

                    tracked_ids.add(track_id)
                else:
                    tracked_ids_validation[track_id] += 1
                    if tracked_ids_validation[track_id] == 5:
                        # Validate the extracted text from the plate image and update the map
                        roi = frame[top:bottom, left:right]  # Extract ROI
                        # Show the particular frame
                        text = self.extract_text_from_roi(roi)
                        if self.text_validate_plate(text) != "Not found":
                            # Update the map
                            tracked_ids_map[track_id] = self.text_validate_plate(text)
                            tracked_ids_validation[track_id] = 0

                self.draw_rectangle_and_text(frame, left, top, right, bottom, track_id, tracked_ids_map)
            for d in vehicle_trackers:
                left, top, right, bottom, track_id = map(int, d)

                # Check if the object is not already tracked
                if track_id not in tracked_ids or track_id not in tracked_ids_map:
                    # draw the bounding box and the text on the frame with brand name
                    roi = frame[top:bottom, left:right]

                    self.draw_vehicle_rectangle_and_text(frame, left, top, right, bottom, brand_tracktion_map[0])
                    tracked_ids.add(track_id)




                else:
                    tracked_ids_validation[track_id] += 1
                    if tracked_ids_validation[track_id] == 5:
                        # Validate the extracted text from the plate image and update the map
                        roi = frame[top:bottom, left:right]

            self.frame_queue.put(frame)

    @staticmethod
    def apply_filters(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    @staticmethod
    def find_edges(image):
        edges = cv2.Canny(image, 50, 150)
        return edges

    @staticmethod
    def find_contours(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def apply_mask(image, mask):
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image

    @staticmethod
    def create_blank_mask(image):
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        return mask

    @staticmethod
    def draw_contours_on_mask(mask, contours):
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
        return mask

    def detect_objects(self, frame, class_id):
        results = self.model(frame)
        detections = []

        for result in results.pred:
            for *xyxy, conf, cls in result:
                if int(cls.item()) == class_id:
                    detections.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), 0])

        return detections

    def detect_brand(self, frame, model,track_map):
        results = model(frame)
        detections = []

        for result in results.pred:
            for *xyxy, conf, cls in result:
                detections.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), 0])
                track_map[0] = int(cls.item())
        return detections


    @staticmethod
    def text_validate_plate(text):
        turkish_plate_pattern1 = re.compile(r'^\d{2}[A-Z]{2,3}\d{2,3}$')
        turkish_plate_pattern2 = re.compile(r'^\d{2}[A-Z]{2,3}\d{2,4}$')
        if turkish_plate_pattern1.match(text):
            return turkish_plate_pattern1.match(text).group()
        elif turkish_plate_pattern2.match(text):
            return turkish_plate_pattern2.match(text).group()
        else:
            return "Not found"

    def extract_text_from_roi(self, roi):
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresholded_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        resized_roi = cv2.resize(thresholded_roi, (0, 0), fx=0.5, fy=0.5)

        # Rotate and find the best match
        CONFIG = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        text = pytesseract.image_to_string(resized_roi, config=CONFIG)
        return text

    def draw_rectangle_and_text(self, frame, left, top, right, bottom, track_id, tracked_ids_map):
        GREEN = (0, 255, 0)
        RED = (0, 0, 255)
        if track_id in tracked_ids_map:
            cv2.rectangle(frame, (left, top), (right, bottom), GREEN, 2)
            plate_text_spaced_vers = tracked_ids_map[track_id][:2] + " "
            if tracked_ids_map[track_id][4].isdigit():
                plate_text_spaced_vers += tracked_ids_map[track_id][2:]
            else:
                plate_text_spaced_vers += tracked_ids_map[track_id][2:5] + " " + tracked_ids_map[track_id][5:]
            if track_id in tracked_ids_map and tracked_ids_map[track_id] in self.registered_plates:
                cv2.putText(frame, f"Plate: {plate_text_spaced_vers} (Registered)", (left - 20, top - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Plate: {plate_text_spaced_vers} (Not Registered)", (left - 20, top - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, GREEN, 2, cv2.LINE_AA)
        else:
            if self.unknown_plate.get(track_id) is None:
                self.unknown_plate[track_id] = 0
            else:
                self.unknown_plate[track_id] += 1
            cv2.rectangle(frame, (left, top), (right, bottom), RED, 2)
            if self.unknown_plate[track_id] >= 10:

                cv2.putText(frame, "Plate: Low res...", (left, top - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Plate:..extracting..", (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2,
                            cv2.LINE_AA)

    @staticmethod
    def save_plate_image(roi, text):
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresholded_roi = cv2.threshold(gray_roi, 25, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray_roi)
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        thresholded_roi = cv2.bitwise_and(thresholded_roi, thresholded_roi, mask=mask)
        if not os.path.exists("plates"):
            os.makedirs("plates")
        cv2.imwrite(f"plates/{text}.jpg", thresholded_roi)

    @staticmethod
    def save_normal_plate_image(roi, text):
        if not os.path.exists("normal_plates"):
            os.makedirs("normal_plates")
        cv2.imwrite(f"normal_plates/{text}.jpg", roi)

    def close(self):
        self.cap.release()
        self.timer.stop()
        super().close()
        exit(0)


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = PlateRecognitionApp()
        window.show()
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        window.close()
        sys.exit(0)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)