import os
import cv2
import numpy as np
import torch


def apply_filters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def detect_plates(model, frame):
    results = model(frame)
    detections = []

    for result in results.pred:
        for *xyxy, conf, cls in result:
            if int(cls.item()) == 12:
                detections.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])

    return detections


def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='/Users/canrollas/Projects/OCR/yolov5/runs/train/exp4/weights/best.pt')
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Extract plate image
            filtered_image = apply_filters(image)
            detections = detect_plates(model, filtered_image)

            for i, detection in enumerate(detections):
                x_min, y_min, x_max, y_max = detection
                plate_image = image[y_min:y_max, x_min:x_max]

                """
                    # save the image to the disk in plates folder
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
                    if not os.path.exists("plates"):
                        os.makedirs("plates")
                    cv2.imwrite(f"plates/{text_validate_plate(text)}.jpg", thresholded_roi)
                """
                gray_plate_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                _, thresholded_plate_image = cv2.threshold(gray_plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresholded_plate_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask = np.zeros_like(gray_plate_image)
                contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
                thresholded_plate_image = cv2.bitwise_and(thresholded_plate_image, thresholded_plate_image, mask=mask)
                cv2.imwrite(os.path.join(output_dir, filename), thresholded_plate_image)



            if not detections:
                print("No plate found in image {}".format(filename))


if __name__ == "__main__":
    input_directory = "/Users/canrollas/Projects/OCR/dataset/images/train"
    output_directory = "/Users/canrollas/Projects/OCR/plates"

    process_images(input_directory, output_directory)
