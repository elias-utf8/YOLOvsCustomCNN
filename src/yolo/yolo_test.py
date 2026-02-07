import configparser
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

# Config
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = configparser.ConfigParser()
CONFIG.read(PROJECT_ROOT / 'config.ini')

# Chemins depuis config.ini
DATASET_DIR = PROJECT_ROOT / CONFIG.get('data', 'DATASET_DIR')
DATASET_YAML = PROJECT_ROOT / CONFIG.get('data', 'DATASET_YAML')
TEST_IMAGES_DIR = DATASET_DIR / 'images' / 'test'
MODELS_DIR = PROJECT_ROOT / CONFIG.get('yolo', 'MODELS_DIR')
RUNS_DIR = PROJECT_ROOT / CONFIG.get('yolo', 'RUNS_DIR')

MODEL_NAME = 'yolo11n'
CONFIANCE = 0.5

if __name__ == "__main__":
    # Charger le meilleur modele entraine
    best_model = RUNS_DIR / MODEL_NAME / 'weights' / 'best.pt'
    yolo_model = YOLO(str(best_model))

    # 1. Évaluer sur le test set
    print(" Évaluation de YOLO sur le TEST SET...\n")
    test_metrics = yolo_model.val(
        data=str(DATASET_YAML),
        split='test'
    )

    print("\n Métriques YOLO sur le TEST SET :")
    print(f"  mAP@0.5     : {test_metrics.box.map50:.3f}")
    print(f"  mAP@0.5:0.95: {test_metrics.box.map:.3f}")
    print(f"  Precision   : {test_metrics.box.mp:.3f}")
    print(f"  Recall      : {test_metrics.box.mr:.3f}")

    # 2. Prédire sur quelques images du test set pour visualisation
    test_images = sorted(list(TEST_IMAGES_DIR.glob('*.jpg')))[:20]

    print(f"\n Prédiction sur {len(test_images)} images de test...")

    for img_path in test_images:
        results = yolo_model.predict(
            source=str(img_path),
            conf=CONFIANCE,
            save=True,
            project=str(PROJECT_ROOT / 'data/yolo/runs/detect'),
            name='yolo_test_predictions'
        )

        # Afficher les détections pour cette image
        print(f"\n {img_path.name}")
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = results[0].names[class_id]
                coords = box.xyxy[0].tolist()
                print(f" {class_name} (conf: {confidence:.2f}) - bbox: [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")
        else:
            print(f"  Aucune détection")

        # Afficher l'image avec les bounding boxes
        img_with_boxes = results[0].plot()
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title(f"Détections YOLO - {img_path.name}")
        plt.axis('off')
        plt.show()

    print(f"\n rédictions sauvegardées dans : data/yolo/runs/detect/yolo_test_predictions/")
