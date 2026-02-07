import sys
import configparser
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox

from src.ui.tabs.detection_tab import DetectionTab
from src.ui.tabs.training_tab import TrainingTab
from src.ui.tabs.calibrage_tab import CalibrageTab


CONFIG_PATH = Path(__file__).parent / "config.ini"

config = configparser.ConfigParser()
config.read(CONFIG_PATH)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("sae-s5.a.01-7")
        self.setMinimumSize(800, 600)

        camera_id = config.getint("camera", "CAMERA_ID", fallback=0)

        # Onglets
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.detection_tab = DetectionTab(camera_id)
        self.training_tab = TrainingTab()
        self.calibrage_tab = CalibrageTab()

        self.tabs.addTab(self.detection_tab, "Détection")
        self.tabs.addTab(self.training_tab, "Entraînement")
        self.tabs.addTab(self.calibrage_tab, "Calibrage")


    def closeEvent(self, event):
        self.detection_tab.cleanup()
        event.accept()


def main():
    app = QApplication(sys.argv)

    if not DetectionTab.is_camera_connected() and not config.getboolean("dev", "DEBUG"):
        QMessageBox.critical(
            None,
            "Erreur",
            "Aucune caméra RealSense détectée.\nVeuillez connecter une caméra et relancer l'application."
        )
        sys.exit(1)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
