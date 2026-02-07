"""Detection model manager (YOLO / CNN)."""
import cv2
import numpy as np
import configparser
from pathlib import Path
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG = configparser.ConfigParser()
CONFIG.read(PROJECT_ROOT / 'config.ini')


@dataclass
class Detection:
    """Data structure for an object detection."""
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    world_coords: tuple | None = None  # (X, Y, Z) in meters

    @property
    def center(self) -> tuple:
        """Returns the center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class ModelManager:
    """Unified manager for YOLO/CNN models."""

    YOLO_PATH = PROJECT_ROOT / CONFIG.get('yolo', 'TRAINED_MODEL')
    CNN_PATH = PROJECT_ROOT / CONFIG.get('cnn', 'TRAINED_MODEL')

    def __init__(self, confidence: float = 0.5):
        """Initializes the model manager."""
        self.confidence = confidence
        self.model = None
        self.model_type = None
        self.device = None
        self._cnn_transform = None

    def is_loaded(self) -> bool:
        """Checks if a model is currently loaded."""
        return self.model is not None

    def load_yolo(self) -> bool:
        """Loads the YOLO model."""
        if not self.YOLO_PATH.exists():
            print(f"[x] YOLO introuvable: {self.YOLO_PATH}")
            return False
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.YOLO_PATH))
            self.model_type = "yolo"
            print(f"[+] YOLO charge")
            return True
        except Exception as e:
            print(f"[x] Erreur YOLO: {e}")
            return False

    def load_cnn(self) -> bool:
        """Loads the custom CNN model."""
        if not self.CNN_PATH.exists():
            print(f"[x] CNN introuvable: {self.CNN_PATH}")
            return False
        try:
            import torch
            from torchvision import transforms
            from .cnn.model import MultiObjectDetector

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = MultiObjectDetector(num_classes=2).to(self.device)
            model.load_state_dict(torch.load(self.CNN_PATH, map_location=self.device))
            model.eval()

            self.model = model
            self.model_type = "cnn"
            self._cnn_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print(f"[+] CNN charge")
            return True
        except Exception as e:
            print(f"[x] Erreur CNN: {e}")
            return False

    def unload(self):
        """Unloads the current model."""
        self.model = None
        self.model_type = None

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        """Runs inference without drawing."""
        if not self.is_loaded():
            return []
        if self.model_type == "yolo":
            return self._yolo_detect(frame_rgb)
        return self._cnn_detect(frame_rgb)

    def _yolo_detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        """Runs YOLO inference."""
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(frame_bgr, conf=self.confidence, verbose=False)

        detections = []
        for box in results[0].boxes:
            cid = int(box.cls[0])
            detections.append(Detection(
                class_id=cid,
                class_name=results[0].names[cid],
                confidence=float(box.conf[0]),
                x1=float(box.xyxy[0][0]),
                y1=float(box.xyxy[0][1]),
                x2=float(box.xyxy[0][2]),
                y2=float(box.xyxy[0][3]),
            ))
        return detections

    def _cnn_detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        """Runs CNN inference."""
        import torch

        classes = CONFIG.get('data', 'CLASSES').split(',')
        h, w = frame_rgb.shape[:2]

        tensor = self._cnn_transform(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cls_pred, reg_pred = self.model(tensor)
            cls_pred = torch.sigmoid(cls_pred[0]).cpu()
            reg_pred = reg_pred[0].cpu()

        detections = []
        for c in range(len(classes)):
            if cls_pred[c] > self.confidence:
                xc, yc, bw, bh = reg_pred[c].numpy()
                detections.append(Detection(
                    class_id=c,
                    class_name=classes[c],
                    confidence=float(cls_pred[c]),
                    x1=(xc - bw/2) * w,
                    y1=(yc - bh/2) * h,
                    x2=(xc + bw/2) * w,
                    y2=(yc + bh/2) * h,
                ))
        return detections
