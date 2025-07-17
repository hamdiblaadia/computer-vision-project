import cv2
from ultralytics import YOLO
import math
import torch

# Charger le modèle YOLOv8
model = YOLO(r"C:\Users\Blaadia\Desktop\PFE\segmentation\runs\segment\train_Yolov8n_seg\weights\bestn.pt")

# Paramètres physiques de la seringue
diametre_seringue = 28.5  # mm
hauteur_seringue = 94.4  # mm
rayon_seringue = diametre_seringue / 2
volume_total = math.pi * (rayon_seringue ** 2) * hauteur_seringue
volume_total_ml = volume_total / 1000

# Initialiser la capture vidéo depuis la webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Détection avec YOLOv8
    results = model(frame)
    result = results[0]

    # Variables pour les hauteurs
    hauteur_detectee_seringue_pixels = None
    hauteur_liquide_pixels = None

    # Parcourir les détections
    for bbox, class_id in zip(result.boxes.xyxy.cpu(), result.boxes.cls.cpu()):
        x1, y1, x2, y2 = map(int, bbox)
        if int(class_id) == 0:  # seringue
            hauteur_detectee_seringue_pixels = y2 - y1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif int(class_id) == 1:  # liquide
            hauteur_liquide_pixels = y2 - y1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Calcul volume si les deux sont détectés
    if hauteur_detectee_seringue_pixels and hauteur_liquide_pixels:
        facteur_conversion = hauteur_seringue / hauteur_detectee_seringue_pixels
        hauteur_liquide_mm = hauteur_liquide_pixels * facteur_conversion
        volume_liquide = math.pi * (rayon_seringue ** 2) * hauteur_liquide_mm
        volume_liquide_ml = volume_liquide / 1000

        # Ajouter le texte du volume sur l'image
        text = f"Volume: {volume_liquide_ml:.1f} ml"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Afficher la frame annotée
    cv2.imshow("YOLOv8 - Volume Seringue Temps Réel", frame)

    # Quitter avec la touche q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
