"""Détection YOLO en temps réel sur flux vidéo."""
import configparser
from pathlib import Path
import cv2
from ultralytics import YOLO

# Config
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = configparser.ConfigParser()
CONFIG.read(PROJECT_ROOT / 'config.ini')

# Paramètres caméra depuis config.ini
CAMERA_ID = CONFIG.getint('camera', 'CAMERA_ID')
IMAGE_WIDTH = CONFIG.getint('camera', 'IMAGE_WIDTH')
IMAGE_HEIGHT = CONFIG.getint('camera', 'IMAGE_HEIGHT')

# Modèle YOLO
TRAINED_MODEL = PROJECT_ROOT / CONFIG.get('yolo', 'TRAINED_MODEL')
PRETRAINED_MODEL = PROJECT_ROOT / CONFIG.get('yolo', 'PRETRAINED_MODEL')

# Paramètres de détection
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45


def load_model():
    """Charge le modèle YOLO (entraîné ou pré-entraîné)."""
    if TRAINED_MODEL.exists():
        print(f" Chargement du modèle entraîné : {TRAINED_MODEL}")
        return YOLO(str(TRAINED_MODEL))
    else:
        print(f"  Modèle entraîné non trouvé : {TRAINED_MODEL}")
        print(f"   Utilisation du modèle pré-entraîné : {PRETRAINED_MODEL}")
        return YOLO(str(PRETRAINED_MODEL))


def run_realtime_detection():
    """Lance la détection en temps réel."""
    model = load_model()

    # Ouvrir la caméra
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    if not cap.isOpened():
        print(f" Impossible d'ouvrir la caméra {CAMERA_ID}")
        return

    print(f"\n Détection en temps réel - Caméra {CAMERA_ID}")
    print(f"   Résolution : {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"   Confiance  : {CONFIDENCE_THRESHOLD}")
    print(f"\n   Appuie sur 'q' pour quitter\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Erreur de lecture de la caméra")
            break

        # Détection YOLO
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False  # Pas de logs pour chaque frame
        )

        # Dessiner les détections sur la frame
        annotated_frame = results[0].plot()

        # Afficher les infos de détection
        detections = results[0].boxes
        if len(detections) > 0:
            for box in detections:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = results[0].names[class_id]

                # Coordonnées de la bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Afficher dans le terminal (optionnel)
                print(f"   {class_name}: {confidence:.2f} @ ({center_x:.0f}, {center_y:.0f})")

        # Afficher la frame annotée
        cv2.imshow('YOLO Detection', annotated_frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n Détection terminée")


def run_on_video_file(video_path: str, output_path: str = None):
    """Lance la détection sur un fichier vidéo."""
    model = load_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Impossible d'ouvrir la vidéo : {video_path}")
        return

    # Récupérer les propriétés de la vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n Traitement vidéo : {video_path}")
    print(f"   {width}x{height} @ {fps}fps - {total_frames} frames")

    # Writer pour sauvegarder si output_path spécifié
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"   Sortie : {output_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Détection YOLO
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        annotated_frame = results[0].plot()

        # Sauvegarder si writer actif
        if writer:
            writer.write(annotated_frame)

        # Afficher
        cv2.imshow('YOLO Detection', annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   Traité : {frame_count}/{total_frames} frames")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\n Terminé - {frame_count} frames traitées")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Mode fichier vidéo
        video_input = sys.argv[1]
        video_output = sys.argv[2] if len(sys.argv) > 2 else None
        run_on_video_file(video_input, video_output)
    else:
        # Mode temps réel (caméra)
        run_realtime_detection()
