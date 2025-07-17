from ultralytics import YOLO

if __name__ == '__main__':
    # Charger le modèle
    model = YOLO(r"C:\Users\Blaadia\Desktop\PFE\segmentation\runs\segment\train_Yolov8n_seg\weights\best.pt")

    # Personnaliser les paramètres de validation
    validation_results = model.val(
        data=r"C:\Users\Blaadia\Desktop\PFE\segmentation\datasets\Data.yaml",
        imgsz=640,
        batch=4,
        conf=0.25,
        iou=0.6,
        device="0"
    )
