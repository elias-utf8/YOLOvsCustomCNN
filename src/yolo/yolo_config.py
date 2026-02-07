import yaml
import configparser
from pathlib import Path
import torch
import shutil
from torch.utils.data import random_split

# Config
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = configparser.ConfigParser()
CONFIG.read(PROJECT_ROOT / 'config.ini')

# Chemins depuis config.ini
RAW_DIR = PROJECT_ROOT / CONFIG.get('data', 'RAW_DIR')
RAW_IMAGES_DIR = RAW_DIR / 'images'
RAW_LABELS_DIR = RAW_DIR / 'labels'
CLASSES_FILE = PROJECT_ROOT / CONFIG.get('data', 'CLASSES_FILE')
DATASET_DIR = PROJECT_ROOT / CONFIG.get('data', 'DATASET_DIR')


def prepare_yolo_from_existing_dataset(
    images_dir,
    labels_dir,
    classes_file,
    output_dir,
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15
    # test_ratio = 1 - train_ratio - val_ratio
):

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    classes_file = Path(classes_file)

    print(f"Préparation du dataset YOLO depuis : {images_dir}")

    # 1. Créer la structure de dossiers pour YOLO
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 2. Récupérer toutes les images
    image_files = sorted(list(images_dir.glob('*.jpg')) +
                        list(images_dir.glob('*.jpeg')) +
                        list(images_dir.glob('*.png')))

    print(f"{len(image_files)} images trouvées")

    if len(image_files) == 0:
        print("Aucune image trouvée ! Vérifiez le chemin.")
        return None, []

    # 3. Split
    total_size = len(image_files)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_indices, val_indices, test_indices = random_split(
        range(len(image_files)),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f" Split (seed={seed}) : {train_size} train, {val_size} val, {test_size} test")

    # 4. Copier les fichiers dans les bons dossiers
    splits = {
        'train': train_indices.indices,
        'val': val_indices.indices,
        'test': test_indices.indices
    }

    for split_name, indices in splits.items():
        print(f"\n Préparation du split '{split_name}'...")

        copied_count = 0
        for idx in indices:
            img_file = image_files[idx]
            label_file = labels_dir / f"{img_file.stem}.txt"

            # Copier l'image
            dest_img = output_dir / 'images' / split_name / img_file.name
            shutil.copy(img_file, dest_img)

            # Copier le label correspondant (si existe)
            if label_file.exists():
                dest_label = output_dir / 'labels' / split_name / label_file.name
                shutil.copy(label_file, dest_label)
                copied_count += 1
            else:
                print(f"  Label manquant pour : {img_file.name}")

        print(f" {copied_count} images + labels copiés")

    # 5. Copier le fichier classes.txt à la racine
    shutil.copy(classes_file, output_dir / 'classes.txt')

    # 6. Charger les classes pour le fichier YAML
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    print(f"\n Classes ({len(classes)}) : {classes}")

    print(f"\n Dataset YOLO préparé dans : {output_dir}")
    print(f"   Structure : images/{{train,val,test}} + labels/{{train,val,test}}")

    return output_dir, classes


def create_yolo_yaml(output_dir, classes, yaml_filename='dataset.yaml'):
    """
    Crée le fichier YAML de configuration pour YOLO.

    Args:
        output_dir: Dossier racine du dataset YOLO (ex: 'data_yolo')
        classes: Liste des noms de classes (ex: ['cube', 'bouteille'])
        yaml_filename: Nom du fichier YAML (défaut: 'dataset.yaml')

    Returns:
        Path vers le fichier YAML créé
    """
    output_dir = Path(output_dir)
    yaml_path = output_dir / yaml_filename

    # Configuration YOLO avec chemins relatifs
    config = {
        'path': str(output_dir.absolute()),  # Chemin absolu vers la racine
        'train': 'images/train',             # Chemin relatif vers images train
        'val': 'images/val',                 # Chemin relatif vers images val
        'test': 'images/test',               # Chemin relatif vers images test

        'nc': len(classes),                  # Nombre de classes
        'names': classes                     # Noms des classes
    }

    # Écrire le fichier YAML
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n Fichier YAML créé : {yaml_path}")
    print(f"   Contenu :")
    print(f"      - path: {config['path']}")
    print(f"      - train: {config['train']}")
    print(f"      - val: {config['val']}")
    print(f"      - test: {config['test']}")
    print(f"      - nc: {config['nc']}")
    print(f"      - names: {config['names']}")

    return yaml_path


if __name__ == "__main__":
    # Utilisation des chemins depuis config.ini
    output_path, classes = prepare_yolo_from_existing_dataset(
        images_dir=RAW_IMAGES_DIR,
        labels_dir=RAW_LABELS_DIR,
        classes_file=RAW_DIR / 'classes.txt',
        output_dir=DATASET_DIR,
        seed=42
    )

    if output_path and classes:
        yaml_path = create_yolo_yaml(output_path, classes)
        print(f"\nFichier de configuration pret : {yaml_path}")
    else:
        print("Erreur : dataset non prepare correctement")
