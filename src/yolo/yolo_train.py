import argparse
import configparser
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO

# Config
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = configparser.ConfigParser()
CONFIG.read(PROJECT_ROOT / 'config.ini')

# Chemins depuis config.ini
DATASET_YAML = PROJECT_ROOT / CONFIG.get('data', 'DATASET_YAML')
DATASET_DIR = PROJECT_ROOT / CONFIG.get('data', 'DATASET_DIR')
PRETRAINED_MODEL = PROJECT_ROOT / CONFIG.get('yolo', 'PRETRAINED_MODEL')
MODELS_DIR = PROJECT_ROOT / CONFIG.get('yolo', 'MODELS_DIR')
TRAINED_MODEL = PROJECT_ROOT / CONFIG.get('yolo', 'TRAINED_MODEL')
RUNS_DIR = PROJECT_ROOT / CONFIG.get('yolo', 'RUNS_DIR')
CLASSES = CONFIG.get('data', 'CLASSES').split(',')

# Paramètres d'entraînement
EPOCH = 15
IMGSZ = 224
BATCH = 2
PATIENCE = 10


def update_dataset_yaml():
    """Met à jour dataset.yaml avec le chemin absolu correct."""
    config = {
        'path': str(DATASET_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    with open(DATASET_YAML, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f" dataset.yaml mis à jour avec : {DATASET_DIR.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int, default=EPOCH)
    parser.add_argument("--batch",   type=int, default=BATCH)
    args = parser.parse_args()

    # Mettre à jour dataset.yaml avec le bon chemin
    update_dataset_yaml()

    # Charger YOLOv11n pré-entraîné
    model = YOLO(str(PRETRAINED_MODEL))

    # Entrainer sur votre dataset
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=IMGSZ,
        batch=args.batch,
        name=PRETRAINED_MODEL.stem,
        patience=PATIENCE,
        device='cpu',
        project=str(RUNS_DIR),
        exist_ok=True  # Ecrase le dossier existant au lieu d'incrementer
    )

    # Copier le meilleur modèle vers data/yolo/models/best.pt
    best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_weights.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights, TRAINED_MODEL)
        print(f"\n Modèle copié vers : {TRAINED_MODEL}")

    print(" Entraînement terminé !")
