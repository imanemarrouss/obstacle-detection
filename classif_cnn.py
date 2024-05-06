import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Chemin vers le dossier contenant les images d'entraînement
train_images_folder = 'C:\\Users\\Lenovo\\Desktop\\ti\\train\\images\\'
train_annotations_folder = 'C:\\Users\\Lenovo\\Desktop\\ti\\train\\labels\\'

class_names = {
    0: "Obstacle",
    1: "Road",
    2: "Sidewalk"
}
# Parcourir tous les fichiers d'images dans le dossier d'entraînement
for filename in os.listdir(train_images_folder):
    if filename.endswith(".jpg"):
        # Charger l'image
        image_path = os.path.join(train_images_folder, filename)
        image = plt.imread(image_path)

        # Charger les annotations correspondantes
        annotation_filename = os.path.splitext(filename)[0] + ".txt"
        annotation_path = os.path.join(train_annotations_folder, annotation_filename)

        annotations = []
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                annotation = {
                    'class': int(parts[0]),
                    'x_min': float(parts[1]),
                    'y_min': float(parts[2]),
                    'x_max': float(parts[3]),
                    'y_max': float(parts[4])
                }
                annotations.append(annotation)

        # Créer une figure et un axe
        fig, ax = plt.subplots()

        # Afficher l'image
        ax.imshow(image)

        # Dessiner les annotations sur l'image
        for annotation in annotations:
            class_id = annotation['class']
            x_min = annotation['x_min'] * image.shape[1]
            y_min = annotation['y_min'] * image.shape[0]
            x_max = annotation['x_max'] * image.shape[1]
            y_max = annotation['y_max'] * image.shape[0]

            # Créer un rectangle englobant
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='g', facecolor='none')

            # Ajouter le rectangle à l'axe
            ax.add_patch(rect)

            # Ajouter une étiquette de classe
            class_name = class_names[class_id]
            ax.text(x_min, y_min - 0.01 * image.shape[0], f"{class_id}: {class_name}", color='g')

        # Afficher l'image avec les annotations
        plt.title("Annotated Image")
        plt.axis('off')
        plt.show()