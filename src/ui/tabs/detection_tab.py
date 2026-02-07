"""Object detection tab with YOLO and CNN."""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QGroupBox, QSizePolicy, QTextEdit, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

import cv2
import numpy as np
import time

from ...realsense.RealSense import RealSenseCamera
from ...depth.DepthProcessor import DepthProcessor
from ...depth.transform import convert_intrinsics_to_K_dist
from ...ModelManager import ModelManager, Detection
from ..Renderer import Renderer


class DetectionTab(QWidget):
    """Main tab for real-time object detection."""

    @staticmethod
    def is_camera_connected() -> bool:
        """Checks if a RealSense camera is connected."""
        return RealSenseCamera.is_connected()

    def __init__(self, camera_id: int = 0, parent=None):
        """Initializes the detection tab."""
        super().__init__(parent)
        self.camera = RealSenseCamera()
        self._model_manager = ModelManager()
        self._renderer = Renderer()
        self._depth_processor = DepthProcessor()
        self._last_time = 0.0
        self._fps = 0.0
        self._current_detections = []
        self._depth_frame = None
        self._intrinsics = None
        self.corners = None
        self.ids = None

        self._setup_ui()
        self._setup_camera()
        self._setup_timer()

    def _setup_ui(self):
        """Sets up the main UI layout and widgets."""
        layout = QVBoxLayout(self)

        groupbox = QGroupBox("Configuration")
        groupbox.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout.addWidget(groupbox)
        vbox = QVBoxLayout(groupbox)

        # Camera
        camera_bar = QHBoxLayout()
        camera_bar.addWidget(QLabel("Camera :"))
        self.camera_combo = QComboBox()
        self._populate_cameras()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        camera_bar.addWidget(self.camera_combo)
        camera_bar.addStretch()
        vbox.addLayout(camera_bar)

        model_bar = QHBoxLayout()
        model_bar.addWidget(QLabel("Modele :"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Aucun", "YOLO", "CNN Custom"])
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_bar.addWidget(self.model_combo)
        model_bar.addStretch()
        vbox.addLayout(model_bar)
        content = QHBoxLayout()
        layout.addLayout(content)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e;")
        content.addWidget(self.video_label, stretch=2)

        right_panel = QVBoxLayout()

        self.detection_text = QTextEdit()
        self.detection_text.setReadOnly(True)
        self.detection_text.setMaximumWidth(300)
        self.detection_text.setPlaceholderText("Les detections apparaitront ici...")
        right_panel.addWidget(self.detection_text, stretch=1)

        self.options_groupbox = QGroupBox("Options d'affichage")
        self.options_groupbox.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        options_layout = QVBoxLayout(self.options_groupbox)

        self.checkbox_3d = QCheckBox("Localisation 3D")
        self.checkbox_3d.stateChanged.connect(self._on_3d_changed)
        options_layout.addWidget(self.checkbox_3d)

        self.checkbox_fps = QCheckBox("Afficher les FPS")
        self.checkbox_fps.setChecked(False)
        options_layout.addWidget(self.checkbox_fps)

        self.checkbox_resolution = QCheckBox("Afficher la résolution")
        self.checkbox_resolution.setChecked(False)
        options_layout.addWidget(self.checkbox_resolution)

        right_panel.addWidget(self.options_groupbox)
        content.addLayout(right_panel, stretch=1)

    def _populate_cameras(self):
        """Populates the camera combo box with available cameras."""
        self.camera_combo.clear()
        for name, data in RealSenseCamera.list_cameras():
            self.camera_combo.addItem(name, data)
        if self.camera_combo.count() == 0:
            self.camera_combo.addItem("Aucune camera", None)

    def _setup_camera(self):
        """Starts the camera if one is available."""
        if self.camera_combo.count() > 0:
            self._start_camera(self.camera_combo.currentData())

    def _setup_timer(self):
        """Sets up the frame update timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(30)

    def _start_camera(self, camera_data):
        """Starts the camera with the given stream data."""
        self.camera.stop()
        if camera_data and isinstance(camera_data, tuple):
            stream_type, serial = camera_data
            if self.checkbox_3d.isChecked():
                self.camera.start_rgbd(serial)
            else:
                self.camera.start(stream_type, serial)

    def _on_camera_changed(self, index: int):
        """Handles camera selection change."""
        data = self.camera_combo.currentData()
        if data:
            self._start_camera(data)

    def _on_model_changed(self, index: int):
        """Loads the selected model."""
        self._model_manager.unload()
        self._current_detections = []

        if index == 1:
            self._model_manager.load_yolo()
        elif index == 2:
            self._model_manager.load_cnn()

    def _on_3d_changed(self, state: int):
        """Restarts camera in appropriate mode when 3D toggled."""
        if not self.checkbox_3d.isChecked():
            self.corners = None
            self.ids = None
        data = self.camera_combo.currentData()
        if data:
            self._start_camera(data)

    def _update_frame(self):
        """Main frame update loop."""
        if self.checkbox_3d.isChecked():
            # RGBD mode
            frame, self._depth_frame, self._intrinsics = self.camera.read_rgbd_frame()
            if frame is None:
                return

            # Continuous calibration with ArUco
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            corners, ids, _ = self._depth_processor.calibrate(frame_bgr, self._intrinsics)
            if corners is not None:
                self.corners = corners
                self.ids = ids
            else:
                self.corners = None
                self.ids = None
        else:
            # Single stream mode
            frame = self.camera.read_frame(apply_undistort=False)
            if frame is None:
                return

        now = time.monotonic()
        if self._last_time > 0:
            self._fps = 0.9 * self._fps + 0.1 / (now - self._last_time)
        self._last_time = now

        if self._model_manager.is_loaded():
            self._current_detections = self._model_manager.detect(frame)

            # Compute 3D coordinates if enabled and calibrated
            if self.checkbox_3d.isChecked() and self._depth_processor.is_calibrated:
                for det in self._current_detections:
                    det.world_coords = self._depth_processor.compute_3d(
                        det.center, self._depth_frame, self._intrinsics
                    )
        if self._intrinsics is not None:
            camera_matrix, dist_coeffs = convert_intrinsics_to_K_dist(self._intrinsics)
        else:
            camera_matrix, dist_coeffs = None, None
        self._renderer.show_fps = self.checkbox_fps.isChecked()
        self._renderer.show_resolution = self.checkbox_resolution.isChecked()
        display = self._renderer.render(frame.copy(), self._current_detections, self._fps, self.corners, self.ids, camera_matrix, dist_coeffs, self._depth_processor.rotation_vec, self._depth_processor.translation_vec)

        self._display_label(display)
        self._display_detections(self._current_detections)

    def _display_label(self, frame: np.ndarray):
        """Displays frame informations in the video label."""
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        scaled = img.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def _display_detections(self, detections: list[Detection]):
        """Displays the list of detections in the text panel."""
        if not detections:
            self.detection_text.setPlainText("Aucune detection")
            return

        lines = []
        for d in detections:
            cx, cy = d.center
            text = (
                f"{d.class_name}\n"
                f"  Confiance: {d.confidence:.2%}\n"
                f"  Position: ({cx:.0f}, {cy:.0f})\n"
            )
            if d.world_coords is not None:
                x, y, z = d.world_coords
                text += f"  Monde: X={x:.2f} Y={y:.2f} Z={z:.2f}\n"
            lines.append(text)
        self.detection_text.setPlainText("\n".join(lines))

    def cleanup(self):
        """Stops the timer and camera."""
        self.timer.stop()
        self.camera.stop()
        self._model_manager.unload()
