"""Intel RealSense camera management module."""
import cv2
import numpy as np
import pickle
import configparser
from pathlib import Path

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# Config
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = configparser.ConfigParser()
CONFIG.read(PROJECT_ROOT / 'config.ini')
CALIBRATION_FILE = PROJECT_ROOT / CONFIG.get('calibration', 'OUTPUT_DIR') / 'calibration.pkl'


class StreamType:
    """Available RealSense stream types."""
    COLOR = "rs_color"
    DEPTH = "rs_depth"
    IR_LEFT = "rs_ir_left"
    IR_RIGHT = "rs_ir_right"


class RealSenseCamera:
    """Manages an Intel RealSense camera."""

    def __init__(self):
        """Initializes the camera and loads calibration."""
        self.pipeline = None
        self.stream_type = None
        self.colorizer = None
        self.serial = None

        # RGBD mode
        self._align = None
        self._profile = None

        # Calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self._load_calibration()

    def _load_calibration(self):
        """Loads calibration parameters from calibration.pkl."""
        if not CALIBRATION_FILE.exists():
            print(f"[-] Calibration non trouvée : {CALIBRATION_FILE}")
            return

        try:
            with open(CALIBRATION_FILE, 'rb') as f:
                data = pickle.load(f)
            self.camera_matrix = np.array(data['camera_matrix'])
            self.dist_coeffs = np.array(data['dist_coeffs'])
            print(f"[+] Calibration chargée : {CALIBRATION_FILE}")
        except Exception as e:
            print(f"[x] Erreur chargement calibration : {e}")

    def _get_calibration(self):
        """Returns the calibration parameters."""
        return self.camera_matrix, self.dist_coeffs

    @staticmethod
    def is_available() -> bool:
        """Checks if the RealSense SDK is available."""
        return REALSENSE_AVAILABLE

    @staticmethod
    def is_connected() -> bool:
        """Checks if a RealSense camera is connected."""
        if not REALSENSE_AVAILABLE:
            return False
        ctx = rs.context()
        return len(ctx.devices) > 0

    @staticmethod
    def list_cameras() -> list:
        """Lists available RealSense cameras and their streams."""
        cameras = []
        if not REALSENSE_AVAILABLE:
            return cameras

        ctx = rs.context()
        for device in ctx.devices:
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            cameras.extend([
                (f"{name} - RGB", (StreamType.COLOR, serial)),
                (f"{name} - Depth", (StreamType.DEPTH, serial)),
                (f"{name} - IR Gauche", (StreamType.IR_LEFT, serial)),
                (f"{name} - IR Droite", (StreamType.IR_RIGHT, serial)),
            ])
        return cameras

    def start(self, stream_type: str, serial: str, width: int = 640, height: int = 480, fps: int = 30):
        """Starts a RealSense stream."""
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("SDK RealSense non disponible")

        self.stop()

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)

        if stream_type == StreamType.COLOR:
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        elif stream_type == StreamType.DEPTH:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.colorizer = rs.colorizer()
        elif stream_type == StreamType.IR_LEFT:
            config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        elif stream_type == StreamType.IR_RIGHT:
            config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
        else:
            raise ValueError(f"Type de flux inconnu: {stream_type}")

        self.stream_type = stream_type
        self.serial = serial
        self.pipeline.start(config)

    def stop(self):
        """Stops the RealSense stream."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.stream_type = None
        self.colorizer = None
        self._align = None
        self._profile = None

    def is_running(self) -> bool:
        """Checks if the stream is active."""
        return self.pipeline is not None

    def is_calibrated(self) -> bool:
        """Checks if calibration is loaded."""
        return self.camera_matrix is not None and self.dist_coeffs is not None

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Applies distortion correction to a frame."""
        if not self.is_calibrated():
            return frame
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def read_frame(self, timeout_ms: int = 100, apply_undistort: bool = False) -> np.ndarray | None:
        """Reads a frame from the active stream."""
        if self.pipeline is None:
            return None

        frame = None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)

            if self.stream_type == StreamType.COLOR:
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame = np.asanyarray(color_frame.get_data())
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            elif self.stream_type == StreamType.DEPTH:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    colorized = self.colorizer.colorize(depth_frame)
                    frame = np.asanyarray(colorized.get_data())
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            elif self.stream_type in (StreamType.IR_LEFT, StreamType.IR_RIGHT):
                ir_frame = frames.get_infrared_frame()
                if ir_frame:
                    frame = np.asanyarray(ir_frame.get_data())
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        except Exception:
            return None

        if frame is not None and apply_undistort:
            frame = self.undistort(frame)

        return frame

    def start_rgbd(self, serial: str, width: int = 640, height: int = 480, fps: int = 30):
        """Starts both color and depth streams with alignment."""
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("SDK RealSense non disponible")

        self.stop()

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self._profile = self.pipeline.start(config)
        self._align = rs.align(rs.stream.color)
        self.stream_type = "rgbd"
        self.serial = serial

    def read_rgbd_frame(self, timeout_ms: int = 100) -> tuple:
        """Reads aligned color and depth frames from the RGBD stream.
        It returns 2 frames.
        """
        if self.pipeline is None or self.stream_type != "rgbd":
            return None, None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            aligned = self._align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None, None

            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Get intrinsics parameters
            intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()

            return color_image, depth_frame, intrinsics
        except Exception:
            return None, None, None

    def __del__(self):
        """Cleans up by stopping the stream."""
        self.stop()
