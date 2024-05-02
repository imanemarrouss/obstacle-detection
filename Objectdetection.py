import cv2
import os
import matplotlib.pyplot as plt

# Load the image and labels
image_directory = 'C:\\Users\\Dell\\Downloads\\dataset_Ti\\train\\images\\'
labels_directory = 'C:\\Users\\Dell\\Downloads\\dataset_Ti\\train\\labels\\'

def read_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        labels = []
        for line in lines:
            label = line.strip().split(' ')
            labels.append(label)
        return labels

def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Equalize histogram to improve contrast
    equalized_image = cv2.equalizeHist(blurred_image)

    return equalized_image

def detect_contours(image):
    # Detect contours using a combination of thresholding and morphology
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def is_contour_inside_labels(contour, labels):
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    contour_area = cv2.contourArea(contour)

    # Iterate through each label and check if the contour is inside
    for label in labels:
        label_x_min = float(label[1])
        label_y_min = float(label[2])
        label_x_max = float(label[3])
        label_y_max = float(label[4])

        # Calculate the intersection area between the contour and the label
        intersection_area = max(0, min(x + w, label_x_max) - max(x, label_x_min)) * max(0,
                                                                                        min(y + h, label_y_max) - max(y,
                                                                                                                      label_y_min))

        # Calculate the ratio of intersection area to contour area
        intersection_ratio = intersection_area / contour_area

        # If the intersection ratio is above a threshold, consider the contour inside the label
        if intersection_ratio > 0.5:  # You may adjust this threshold based on your requirements
            return True

    return True

def draw_rectangles(image, contours):
    # Make a copy of the original image
    image_with_rectangles = image.copy()

    # Iterate through each contour and draw a rectangle around it
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_with_rectangles

for filename in os.listdir(image_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Detect contours
        contours = detect_contours(preprocessed_image)

        # Read labels
        labels_path = os.path.join(labels_directory, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        labels = read_labels(labels_path)

        # Filter contours based on size and intersection with labels
        min_contour_area = 100  # Define the minimum size of a contour to be considered an obstacle
        filtered_contours = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > min_contour_area and is_contour_inside_labels(contour, labels):
                filtered_contours.append(contour)

        # Draw rectangles around filtered contours
        image_with_rectangles = draw_rectangles(image, filtered_contours)

        # Display the result
        plt.imshow(cv2.cvtColor(image_with_rectangles, cv2.COLOR_BGR2RGB))
        plt.title('Detected Obstacles')
        plt.axis('off')
        plt.show()
