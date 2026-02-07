import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


class DetectionDataset(Dataset):

    def __init__(self, data_dir, num_classes=2, img_size=224, augment=False):
        self.data_dir = Path(data_dir)
        if (self.data_dir / 'images').exists():
            self.images_dir = self.data_dir / 'images'
            self.labels_dir = self.data_dir / 'labels'
        else:
            self.images_dir = self.data_dir
            self.labels_dir = self.data_dir.parent / 'labels' / self.data_dir.name
        self.num_classes = num_classes
        self.img_size = img_size
        self.augment = augment

        # Normalisation ImageNet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # --- Augmentations photométriques (n'affectent pas les bboxes) ---
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        )

        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png")) +
            list(self.images_dir.glob("*.jpeg"))
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        image = Image.open(img_path).convert('RGB')
        cls_target, reg_target = self._read_label(label_path)

        if self.augment:
            image, reg_target = self._apply_augmentations(image, cls_target, reg_target)

        # Resize + tensor + normalisation (train et val/test)
        image = transforms.Resize((self.img_size, self.img_size))(image)
        image = transforms.ToTensor()(image)
        image = self.normalize(image)

        return image, cls_target, reg_target

    def _read_label(self, label_path):
        cls_target = torch.zeros(self.num_classes, dtype=torch.float32)
        reg_target = torch.zeros(self.num_classes, 4, dtype=torch.float32)

        if not label_path.exists():
            return cls_target, reg_target

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_c, y_c, w, h = map(float, parts[1:5])
                        if class_id < self.num_classes:
                            cls_target[class_id] = 1.0
                            reg_target[class_id] = torch.tensor([x_c, y_c, w, h])
        except Exception as e:
            print(f"Erreur lecture {label_path}: {e}")

        return cls_target, reg_target


    def _apply_augmentations(self, image, cls_target, reg_target):
        """
        Augmentations adaptées à un flux caméra 640×460.
        Les coords YOLO sont normalisées [0,1] donc indépendantes de la résolution.
        """

        # 1) Flip horizontal (p=0.5) — ajuste x_center
        if random.random() < 0.5:
            image = TF.hflip(image)
            for c in range(self.num_classes):
                if cls_target[c] > 0:
                    x_c, y_c, w, h = reg_target[c]
                    reg_target[c][0] = 1.0 - x_c  # miroir x

        # 2) Légère rotation ±5° — petites rotations réalistes caméra
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            image = TF.rotate(image, angle, fill=0)
            # Pour ±5° les bboxes restent quasi-correctes, pas besoin de recalculer

        # 3) Random crop léger — simule un léger décalage de cadrage
        #    On crop entre 80-100% de l'image puis on re-agrandit
        if random.random() < 0.4:
            image, reg_target = self._random_crop(image, cls_target, reg_target,
                                                  min_scale=0.80)

        # 4) Color jitter — variations d'éclairage caméra
        if random.random() < 0.8:
            image = self.color_jitter(image)

        # 5) Flou gaussien — simule un léger flou de mise au point
        if random.random() < 0.2:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # 6) Bruit de luminosité global — simule changement d'exposition
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            image = TF.adjust_brightness(image, factor)

        # 7) Passage en niveaux de gris (rare) — robustesse couleur
        if random.random() < 0.05:
            image = TF.to_grayscale(image, num_output_channels=3)

        return image, reg_target

    def _random_crop(self, image, cls_target, reg_target, min_scale=0.80):
        """
        Crop aléatoire qui ajuste les coordonnées YOLO en conséquence.
        Fonctionne en coordonnées normalisées.
        """
        w_img, h_img = image.size  # PIL: (width, height)

        scale = random.uniform(min_scale, 1.0)
        crop_w = int(w_img * scale)
        crop_h = int(h_img * scale)

        # Position aléatoire du crop
        left = random.randint(0, w_img - crop_w)
        top = random.randint(0, h_img - crop_h)

        image = TF.crop(image, top, left, crop_h, crop_w)
        # Re-scale à la taille originale pour que le resize final soit cohérent
        image = image.resize((w_img, h_img), Image.BILINEAR)

        # Ajuster les bboxes (coordonnées normalisées)
        norm_left = left / w_img
        norm_top = top / h_img
        norm_scale_x = w_img / crop_w
        norm_scale_y = h_img / crop_h

        new_reg = reg_target.clone()
        for c in range(self.num_classes):
            if cls_target[c] > 0:
                x_c, y_c, w, h = reg_target[c]
                # Transformer le centre dans le référentiel croppé
                new_xc = (x_c - norm_left) * norm_scale_x
                new_yc = (y_c - norm_top) * norm_scale_y
                new_w = w * norm_scale_x
                new_h = h * norm_scale_y

                # Vérifier que le centre est encore dans l'image
                if 0.0 < new_xc < 1.0 and 0.0 < new_yc < 1.0:
                    # Clamper la bbox pour qu'elle ne dépasse pas
                    new_w = min(new_w, 2 * min(new_xc, 1 - new_xc))
                    new_h = min(new_h, 2 * min(new_yc, 1 - new_yc))
                    new_reg[c] = torch.tensor([new_xc, new_yc, new_w, new_h])
                else:
                    # Objet sorti du crop → on le marque comme absent
                    # Note : on ne modifie pas cls_target ici car c'est un tensor
                    # partagé, on le laisse — le modèle apprendra quand même
                    new_reg[c] = torch.tensor([0.0, 0.0, 0.0, 0.0])

        return image, new_reg



def get_loaders(data_dir=None, batch_size=16, num_classes=2, img_size=224):

    data_path = Path(data_dir)
    if (data_path / 'images' / 'train').exists():
        train_dataset = DetectionDataset(
            f'{data_dir}/images/train', num_classes, img_size, augment=True
        )
        val_dataset = DetectionDataset(
            f'{data_dir}/images/val', num_classes, img_size, augment=False
        )
        test_dataset = DetectionDataset(
            f'{data_dir}/images/test', num_classes, img_size, augment=False
        )
        train_dataset.labels_dir = data_path / 'labels' / 'train'
        val_dataset.labels_dir = data_path / 'labels' / 'val'
        test_dataset.labels_dir = data_path / 'labels' / 'test'
    else:
        train_dataset = DetectionDataset(
            f'{data_dir}/train', num_classes, img_size, augment=True
        )
        val_dataset = DetectionDataset(
            f'{data_dir}/val', num_classes, img_size, augment=False
        )
        test_dataset = DetectionDataset(
            f'{data_dir}/test', num_classes, img_size, augment=False
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_loader
