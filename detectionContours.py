import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Fonction pour lire les labels à partir d'un fichier
def read_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        labels = []
        for line in lines:
            label = line.strip().split(' ')
            labels.append(label)
        return labels

# Fonction pour vérifier si un contour est à l'intérieur des labels
def is_contour_inside_labels(contour, labels):
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    contour_area = cv2.contourArea(contour)

    # Check if the contour is inside any of the labeled regions
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


# Chemin vers le répertoire contenant les images
image_directory = 'C:\\Users\\Lenovo\\Desktop\\ti\\train\\images\\'
labels_directory = 'C:\\Users\\Lenovo\\Desktop\\ti\\train\\labels\\'

# Parcourir tous les fichiers dans le répertoire
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Vérifier si le fichier est une image
        # Charger l'image en niveaux de gris
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path, 0)

        # Appliquer un filtre gaussien pour réduire le bruit
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Appliquer une égalisation d'histogramme pour améliorer le contraste
        equalized_image = cv2.equalizeHist(blurred_image)

        # Appliquer une opération de morphologie (ouverture) pour éliminer les petits détails indésirables
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed_image = cv2.morphologyEx(equalized_image, cv2.MORPH_OPEN, kernel)

        # Appliquer un seuillage adaptatif pour détecter les obstacles
        _, obstacles = cv2.threshold(morphed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Appliquer une deuxième seuillage spécifique pour les ombres
        _, shadow_threshold = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY)

        # Inverser le seuillage des ombres
        shadow_threshold_inv = cv2.bitwise_not(shadow_threshold)

        # Supprimer les ombres des obstacles
        obstacles_no_shadows = cv2.bitwise_and(obstacles, obstacles, mask=shadow_threshold_inv)

        # Appliquer la détection des contours en utilisant le Laplacien du Gaussien (LoG)
        laplacian = cv2.Laplacian(obstacles_no_shadows, cv2.CV_64F)
        log_edges = np.uint8(np.absolute(laplacian))

        # Trouver les contours dans l'image seuillée
        contours, _ = cv2.findContours(log_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lire les labels correspondant à l'image actuelle
        labels_path = os.path.join(labels_directory, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        labels = read_labels(labels_path)

        # Filtrer les contours en fonction de leur taille et de leur position par rapport aux labels
        min_contour_area = 100  # Définir la taille minimale d'un contour pour être considéré comme un obstacle
        filtered_contours = []
        for cnt in contours:
            contour_area = cv2.contourArea(cnt)
            if contour_area > min_contour_area and is_contour_inside_labels(cnt, labels):
                filtered_contours.append(cnt)

        # Dessiner les contours filtrés sur l'image originale
        image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_with_contours, filtered_contours, -1, (0, 255, 0), 2)

        # Afficher l'image avec les contours détectés
        plt.imshow(image_with_contours)
        plt.title('Filtered Obstacles detected')
        plt.axis('off')
        plt.show()
