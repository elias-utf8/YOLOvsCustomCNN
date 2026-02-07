"""Visual overlay renderer for detection frames."""
import cv2
import numpy as np

from ..ModelManager import Detection


class Renderer:
    """Draws all visual overlays on camera frames."""

    BBOX_COLORS = [(255, 0, 0), (0, 0, 255)]

    def __init__(self):
        self.show_fps = False
        self.show_resolution = False

    def render(self, frame: np.ndarray, detections: list[Detection], fps: float, corners=None, ids=None, camera_matrix = None, distCoeffs = None,rotation_vec = None, transition_vec = None) -> np.ndarray:
        """Draws detections and HUD on a frame."""
        self._draw_detections(frame, detections, corners, ids,camera_matrix , distCoeffs, rotation_vec, transition_vec)
        self._draw_hud(frame, fps)
        return frame

    def _draw_detections(self, frame: np.ndarray, detections: list[Detection], corners=None, ids=None, camera_matrix = None, distCoeffs = None, rotation_vec = None, transition_vec = None):
        """Draws bounding boxes and labels."""
        for det in detections:
            color = self.BBOX_COLORS[det.class_id % len(self.BBOX_COLORS)]
            pt1 = (int(det.x1), int(det.y1))
            pt2 = (int(det.x2), int(det.y2))
            cv2.rectangle(frame, pt1, pt2, color, 2)
            label = f"{det.class_name} {det.confidence:.0%}"
            cv2.putText(frame, label, (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw world coordinates if available
            if det.world_coords is not None:
                x, y, z = det.world_coords
                coord_text = f"X={x:.2f} Y={y:.2f} Z={z:.2f}"
                cv2.putText(frame, coord_text, (pt1[0], pt2[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.circle(frame, (pt1[0], pt2[1]), 5, (0, 255, 0), -1)

        if corners is not None and ids is not None and rotation_vec is not None and transition_vec is not None and camera_matrix is not None and distCoeffs is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, camera_matrix, distCoeffs, rotation_vec, transition_vec, 0.01)

    def _draw_hud(self, frame: np.ndarray, fps: float):
        """Draws HUD overlay (FPS, resolution)."""
        h, w = frame.shape[:2]
        lines = []
        if self.show_fps:
            lines.append(f"FPS: {fps:.1f}")
        if self.show_resolution:
            lines.append(f"{w}x{h}")
        for i, text in enumerate(lines):
            y = 25 + i * 22
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
