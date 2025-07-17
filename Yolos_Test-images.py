from ultralytics import YOLO
import cv2
import numpy as np
import math

# Charger le modèle YOLOv8
model = YOLO(r"C:\Users\Blaadia\Desktop\PFE\segementation1\detect\train7\weights\best.pt")

# Charger l'image de test
image_path = r"C:\Users\Blaadia\Desktop\PFE\segementation1\Hamdi-1\test\images\image_1743466566_jpg.rf.92f638a97a70b2a58caaded7fc89d7d1.jpg"
image = cv2.imread(image_path)

# Vérifier que l'image est bien chargée
if image is None:
    raise FileNotFoundError(f"Image non trouvée à l'emplacement : {image_path}")

# Faire la prédiction avec YOLOv8 (mode segmentation)
results = model(image)

# `results` est une liste, accéder au premier élément
result = results[0]

# Afficher l'image annotée avec les résultats
result.show()

# Paramètres physiques
diametre_seringue = 28# mm
hauteur_seringue = 94.4  # mm
rayon_seringue = diametre_seringue / 2
volume_total = math.pi * (rayon_seringue ** 2) * hauteur_seringue
volume_total_ml = volume_total / 1000
print(f"Volume total de la seringue : {volume_total_ml:.2f} ml")

# Détection de la seringue et du liquide
for bbox, class_id, score in zip(result.boxes.xyxy.cpu(), result.boxes.cls.cpu(), result.boxes.conf.cpu()):
    if int(class_id) == 0:  # seringue
        x1, y1, x2, y2 = bbox
        hauteur_detectee_seringue_pixels = y2 - y1
        facteur_conversion_pixels_mm = hauteur_seringue / hauteur_detectee_seringue_pixels
        print(f"Facteur de conversion : {facteur_conversion_pixels_mm:.4f} mm/pixel")

        # Recherche du liquide
        for bbox_liquide, class_id_liquide, score_liquide in zip(result.boxes.xyxy.cpu(), result.boxes.cls.cpu(), result.boxes.conf.cpu()):
            if int(class_id_liquide) == 1:  # liquide
                x1_l, y1_l, x2_l, y2_l = bbox_liquide
                hauteur_liquide_pixels = y2_l - y1_l
                hauteur_liquide_mm = hauteur_liquide_pixels * facteur_conversion_pixels_mm
                volume_liquide = math.pi * (rayon_seringue ** 2) * hauteur_liquide_mm
                volume_liquide_ml = volume_liquide / 1000
                print(f"Hauteur liquide détectée : {hauteur_liquide_mm:.2f} mm")
                print(f"Volume du liquide dans la seringue : {volume_liquide_ml:.2f} ml")
                # Écrire le volume sur l'image
                text = f"{volume_liquide_ml:.1f} mL"
                org = (int(x1_l), int(y1_l) - 10)  # position du texte au-dessus de la boîte
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (0, 255, 0)  # Vert
                thickness = 2

                cv2.putText(image, text, org, font, font_scale, color, thickness)
                cv2.rectangle(image, (int(x1_l), int(y1_l)), (int(x2_l), int(y2_l)), color, 2)  # dessiner boîte

# Afficher l'image avec volume
cv2.imshow("Volume de la seringue", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sauvegarder l'image si besoin
cv2.imwrite("seringue_volume_annotée.jpg", image)
