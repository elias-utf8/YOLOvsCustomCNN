"""Calibration parameters display tab."""
import json
import configparser
from pathlib import Path

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt6.QtCore import Qt

# Config
PROJECT_ROOT = Path(__file__).parents[3]
CONFIG = configparser.ConfigParser()
CONFIG.read(PROJECT_ROOT / 'config.ini')

CALIBRATION_JSON = PROJECT_ROOT / CONFIG.get('calibration', 'OUTPUT_DIR') / 'calibration.json'


class CalibrageTab(QWidget):
    """Tab displaying calibration parameters from calibration.json."""

    def __init__(self, parent=None):
        """Initializes the calibration tab."""
        super().__init__(parent)
        self._setup_ui()
        self._load_calibration()

    def _setup_ui(self):
        """Sets up the tab layout and widgets."""
        layout = QVBoxLayout(self)

        title = QLabel("Paramètres de calibration")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        btn_refresh = QPushButton("Rafraîchir")
        btn_refresh.clicked.connect(self._load_calibration)
        layout.addWidget(btn_refresh)

    def _load_calibration(self):
        """Reads and displays the content of calibration.json."""
        if not CALIBRATION_JSON.exists():
            self.results_text.setPlainText(
                f"Fichier de calibration non trouvé :\n{CALIBRATION_JSON}"
            )
            return

        try:
            with open(CALIBRATION_JSON) as f:
                data = json.load(f)

            lines = [
                f"RMS Error : {data['rms_error']:.4f} pixels",
                "",
                f"Focal : fx={data['focal']['fx']:.2f}, fy={data['focal']['fy']:.2f}",
                f"Centre optique : cx={data['optical_center']['cx']:.2f}, cy={data['optical_center']['cy']:.2f}",
                "",
                "Distorsion radiale :",
                f"  k1={data['distortion']['radial']['k1']:.6f}",
                f"  k2={data['distortion']['radial']['k2']:.6f}",
                f"  k3={data['distortion']['radial']['k3']:.6f}",
                "Distorsion tangentielle :",
                f"  p1={data['distortion']['tangential']['p1']:.6f}",
                f"  p2={data['distortion']['tangential']['p2']:.6f}",
            ]
            self.results_text.setPlainText("\n".join(lines))
        except Exception as e:
            self.results_text.setPlainText(f"Erreur lecture fichier : {e}")
