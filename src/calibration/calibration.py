"""Camera calibration using chessboard images."""
import numpy as np
import cv2
import glob
import os
import json
import pickle
import configparser
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = configparser.ConfigParser()
CONFIG.read(PROJECT_ROOT / 'config.ini')

CHESSBOARD = (CONFIG.getint('calibration', 'CHESSBOARD_ROWS'), CONFIG.getint('calibration', 'CHESSBOARD_COLS'))
SQUARE_SIZE = CONFIG.getfloat('calibration', 'SQUARE_SIZE')
IMAGES_DIR = PROJECT_ROOT / CONFIG.get('calibration', 'CALIBRATION_IMAGES_DIR')
OUTPUT_DIR = PROJECT_ROOT / CONFIG.get('calibration', 'OUTPUT_DIR')


def calibrate():
    """Calibrate camera from captured images."""
    images = glob.glob(f'{IMAGES_DIR}/*.jpg')
    if not images:
        return print(f"No images in {IMAGES_DIR}")

    # Prepare 3D points
    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints, imgpoints = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        gray = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD)
        if found:
            objpoints.append(objp)
            imgpoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
            print(f"[OK] {os.path.basename(fname)}")
        else:
            print(f"[SKIP] {os.path.basename(fname)}")

    if not objpoints:
        return print("No chessboard detected")

    # Calibrate
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Extract parameters
    fx, fy = float(mtx[0, 0]), float(mtx[1, 1])
    cx, cy = float(mtx[0, 2]), float(mtx[1, 2])
    k1, k2, p1, p2, k3 = [float(x) for x in dist[0][:5]]

    calibration_data = {
        'rms_error': float(rms),
        'focal': {'fx': fx, 'fy': fy},
        'optical_center': {'cx': cx, 'cy': cy},
        'distortion': {
            'radial': {'k1': k1, 'k2': k2, 'k3': k3},
            'tangential': {'p1': p1, 'p2': p2}
        },
        'camera_matrix': mtx.tolist(),
        'dist_coeffs': dist.tolist()
    }

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'calibration.pkl'), 'wb') as f:
        pickle.dump(calibration_data, f)
    with open(os.path.join(OUTPUT_DIR, 'calibration.json'), 'w') as f:
        json.dump(calibration_data, f, indent=2)

    # Display results
    print(f"\n{'='*40}")
    print(f"RMS Error: {rms:.4f} pixels")
    print(f"Focal: fx={fx:.1f}, fy={fy:.1f}")
    print(f"Center: cx={cx:.1f}, cy={cy:.1f}")
    print(f"Radial: k1={k1:.6f}, k2={k2:.6f}, k3={k3:.6f}")
    print(f"Tangential: p1={p1:.6f}, p2={p2:.6f}")
    print(f"{'='*40}")
    print(f"Saved to {OUTPUT_DIR}/calibration.[pkl|json]")


if __name__ == "__main__":
    calibrate()