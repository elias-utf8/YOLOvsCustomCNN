"""Training tab — lance yolo_train.py ou cnn/train.py en sous-processus."""
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSpinBox, QPushButton, QTextEdit, QGroupBox,
)
from PyQt6.QtCore import Qt, QProcess

PROJECT_ROOT = Path(__file__).parents[3]
YOLO_SCRIPT = PROJECT_ROOT / "src" / "yolo" / "yolo_train.py"
CNN_SCRIPT  = PROJECT_ROOT / "src" / "cnn"  / "train.py"

DEFAULTS = {
    "YOLO": {"epochs": 15, "batch": 2},
    "CNN Custom":  {"epochs": 50, "batch": 16},
}


class TrainingTab(QWidget):
    """Tab pour lancer l'entraînement YOLO ou CNN."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Titre
        title = QLabel("Entraînement de modèle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        config_box = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_box)

        # Sélection du modèle
        row = QHBoxLayout()
        row.addWidget(QLabel("Modèle :"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLO", "CNN Custom"])
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        row.addWidget(self.model_combo)
        config_layout.addLayout(row)

        # Nombre d'epochs
        row = QHBoxLayout()
        row.addWidget(QLabel("Epochs :"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(DEFAULTS["YOLO"]["epochs"])
        row.addWidget(self.epochs_spin)
        config_layout.addLayout(row)

        # Taille du batch
        row = QHBoxLayout()
        row.addWidget(QLabel("Batch size :"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(DEFAULTS["YOLO"]["batch"])
        row.addWidget(self.batch_spin)
        config_layout.addLayout(row)

        layout.addWidget(config_box)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Démarrer")
        self.btn_start.clicked.connect(self._start_training)
        self.btn_stop = QPushButton("Arrêter")
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        btn_clear = QPushButton("Nettoyer")
        btn_clear.clicked.connect(self._clear)
        btn_layout.addWidget(btn_clear)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)


    def _on_model_changed(self, model_name: str):
        """Met à jour epochs/batch avec les valeurs par défaut du modèle."""
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            return  # pas de changement pendant un entraînement en cours
        defaults = DEFAULTS[model_name]
        self.epochs_spin.setValue(defaults["epochs"])
        self.batch_spin.setValue(defaults["batch"])

    def _start_training(self):
        """Lance le script d'entraînement en sous-processus."""
        model_name = self.model_combo.currentText()
        epochs = self.epochs_spin.value()
        batch  = self.batch_spin.value()

        # Choisir le script et le répertoire de travail
        if model_name == "YOLO":
            script      = str(YOLO_SCRIPT)
            working_dir = str(PROJECT_ROOT)
        else:  # CNN
            script      = str(CNN_SCRIPT)
            working_dir = str(PROJECT_ROOT / "src" / "cnn")

        cmd_args = [script, "--epochs", str(epochs), "--batch", str(batch)]

        # Réinitialiser les logs
        self.log.clear()
        self._log(f"uv run {' '.join(cmd_args)}\n")

        # Créer le processus
        self.process = QProcess(self)
        self.process.setWorkingDirectory(working_dir)
        # stderr + stdout (tqdm/ultralytics sont dans stderr)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Lancer
        self.process.start(sys.executable, cmd_args)

    def _stop_training(self):
        """Tue le sous-processus en cours."""
        if self.process:
            self.process.kill()
            self.process.waitForFinished()

    def _on_output(self):
        """Appelé dès qu'il y a de la sortie à lire."""
        text = self.process.readAllStandardOutput().data().decode()
        self._log(text)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """Appelé quand le processus se termine."""
        if exit_code == 0:
            self._log("\nEntraînement terminé avec succès.")
        else:
            self._log(f"\nProcessus terminé avec le code {exit_code}.")

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _log(self, text: str):
        """Ajoute du texte au panneau de logs et scrolle vers le bas."""
        self.log.insertPlainText(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _clear(self):
        """Réinitialise l'interface à l'état initial."""
        self.log.clear()
        self.model_combo.setCurrentIndex(0)
        self._on_model_changed(self.model_combo.currentText())
