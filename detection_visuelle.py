import os
import cv2
import numpy as np

def detect_object(query_image_path, target_image_path):
    # Load query image and target image
    query_img = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    query_keypoints, query_descriptors = orb.detectAndCompute(query_img, None)
    target_keypoints, target_descriptors = orb.detectAndCompute(target_img, None)

    # Initialize matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(query_descriptors, target_descriptors)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top matches
    img_matches = cv2.drawMatches(query_img, query_keypoints, target_img, target_keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display result
    cv2.imshow('Object Detection Result', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the dataset directory
dataset_dir = 'C:\\Users\\Lenovo\\Desktop\\ti\\train\\images'

# Iterate over all images in the dataset directory
for query_image_name in os.listdir(dataset_dir):
    query_image_path = os.path.join(dataset_dir, query_image_name)
    
    # Compare the query image against all other images in the dataset directory
    for target_image_name in os.listdir(dataset_dir):
        target_image_path = os.path.join(dataset_dir, target_image_name)
        
        # Perform object detection
        detect_object(query_image_path, target_image_path)
