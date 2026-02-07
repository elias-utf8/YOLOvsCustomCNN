"""Capture chessboard images for camera calibration."""
import cv2
import os
import configparser
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = configparser.ConfigParser()
CONFIG.read(PROJECT_ROOT / 'config.ini')

CAMERA_ID = CONFIG.getint('camera', 'CAMERA_ID')
RESOLUTION = (CONFIG.getint('camera', 'IMAGE_WIDTH'), CONFIG.getint('camera', 'IMAGE_HEIGHT'))
CHESSBOARD = (CONFIG.getint('calibration', 'CHESSBOARD_ROWS'), CONFIG.getint('calibration', 'CHESSBOARD_COLS'))
IMAGES_DIR = PROJECT_ROOT / CONFIG.get('calibration', 'CALIBRATION_IMAGES_DIR')


def capture():
    """Capture chessboard images. Press 'c' to capture, 'q' to quit."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

    if not cap.isOpened():
        return print(f"Cannot open camera {CAMERA_ID}")

    print("Press 'c' to capture, 'q' to quit")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        found, corners = cv2.findChessboardCorners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), CHESSBOARD)
        if found:
            cv2.drawChessboardCorners(display, CHESSBOARD, corners, found)

        cv2.putText(display, f"Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Calibration', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            path = os.path.join(IMAGES_DIR, f"calib_{count:02d}.jpg")
            cv2.imwrite(path, frame)
            print(f"Saved: {path}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture()
