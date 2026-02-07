"""Depth processing and 3D localization."""
import numpy as np

from utils.arucoMarkerDetection import findArucoMarkers
from .transform import calibration_extrinsic, transform_from_cam_to_world
from .deprojection import deproject_en_3D


class DepthProcessor:
    """Handles ArUco calibration and 3D coordinate computation."""

    MARKER_ID = None
    MARKER_SIZE = 0.045  # mètres

    def __init__(self):
        """Initializes the depth processor."""
        self.rotation_vec = None
        self.translation_vec = None

    @property
    def is_calibrated(self) -> bool:
        """Returns True if extrinsic calibration is done."""
        return self.rotation_vec is not None

    def calibrate(self, frame_bgr: np.ndarray, intrinsics) -> tuple:
        """Calibrates using ArUco marker."""
        corners, ids = findArucoMarkers(frame_bgr, target_id=self.MARKER_ID)
        if ids is None or len(corners) == 0:
            return None, None, False

        success, self.rotation_vec, self.translation_vec, _, _ = \
            calibration_extrinsic(self.MARKER_SIZE, intrinsics, corners[0][0])
        return corners, ids, success

    def compute_3d(self, center: tuple, depth_frame, intrinsics) -> tuple | None:
        """Computes world coordinates for a detection center point."""
        if not self.is_calibrated:
            return None

        cx, cy = int(center[0]), int(center[1])
        depth = self.get_depth_of_pixel(depth_frame, cx, cy)
        if depth <= 0:
            return None

        X_cam = deproject_en_3D(intrinsics, cx, cy, depth)
        if X_cam is None:
            return None

        X_world = transform_from_cam_to_world(X_cam, self.rotation_vec, self.translation_vec)
        return tuple(X_world)

    def get_depth_of_pixel(self, depth_frame, u: int, v: int, patch: int = 5) -> float:
        """Returns median depth at pixel (u,v) in meters."""
        # w = depth_frame.get_width()
        # h = depth_frame.get_height()
        # r = patch // 2

        # zs = []
        # for dv in range(-r, r + 1):
        #     for du in range(-r, r + 1):
        #         uu = int(np.clip(u + du, 0, w - 1))
        #         vv = int(np.clip(v + dv, 0, h - 1))
        #         z = depth_frame.get_distance(uu, vv)
        #         if 0.1 <= z <= 3.0:
        #             zs.append(z)

        # return float(np.median(zs)) if zs else 0.0

        if depth_frame is None:
            raise RuntimeError("[DepthProcessor] depth_frame is None")
        return depth_frame.get_distance(u, v)

