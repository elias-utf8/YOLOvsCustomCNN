import cv2
import pyrealsense2 as rs
import numpy as np

def deproject_en_3D(intrinsics, u, v, z_m) -> np.array:
    """
    Convertir le pixel 2D + profondeur en coordonnées 3D dans le repère caméra
    
    Paramètres
    ----------
        u, v : int
            Coordonnées pixel (image RGB alignée).
        z_m : float
            Profondeur en mètres.
        intr : Intrinsics
            Paramètres intrinsèques (RealSense ou OpenCV).

    Retour
    ------
        X_cam : np.ndarray shape (3,)
            Coordonnées 3D (X,Y,Z) en mètres dans le repère caméra.
            None si échec.
    """
    if z_m <= 0:
        return None
    res = rs.rs2_deproject_pixel_to_point(intrinsics, [u,v], z_m)
    return np.array(res, dtype=np.float64)

