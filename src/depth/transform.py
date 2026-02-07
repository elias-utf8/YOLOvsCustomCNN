"""Camera-to-world coordinate transformation utilities."""
import cv2
import numpy as np

def calibration_extrinsic(size_marker, camera_instrinsics, marker_position_in_camera):
    # la position du repère (ici le marqueur) de référence en 3D dans le repère lui-même, 
    # avec le point (0,0,0) est son centre 
    size = float(size_marker)
    marker_position = np.array([
                [-size/2, +size/2, 0],
                [+size/2, +size/2, 0],
                [+size/2, -size/2, 0],
                [-size/2, -size/2, 0]
            ], dtype = np.float32)
    
    # la position du repère (ici le marquer) de référence
    # dans le repère de la caméra.
    image_marker_pos = np.array(marker_position_in_camera, dtype=np.float32).reshape(-1, 1, 2)
    
    camera_matrix, dist_coeffs = convert_intrinsics_to_K_dist(camera_instrinsics)
    
    # sucess, rotation_vect, translation_vect = cv2.solvePnP(marker_position,image_marker_pos, camera_matrix, dist_coeffs)   
    sucess, rotation_vect, translation_vect = cv2.solvePnP(
        marker_position, image_marker_pos, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE 
    )

    return sucess, rotation_vect, translation_vect, camera_matrix, dist_coeffs
    
def convert_intrinsics_to_K_dist(camera_intrinsics):
    # configurer la camera maxtrix
    camera_matrix = np.array([[camera_intrinsics.fx, 0,                      camera_intrinsics.ppx],
                                [0,                     camera_intrinsics.fy , camera_intrinsics.ppy],
                                [0,                   0,                    1]], dtype = np.float64)
    
    # récupérer les coefficients de distorsion
    dist_coeffs = np.array(camera_intrinsics.coeffs).reshape(-1,1)
    
    return camera_matrix, dist_coeffs

def transform_from_cam_to_world(position_3D_object_camera, rotation_vector, translation_vector):
    """
    Transformer la position (3D) de l'objet from du repère Camera au repère "monde"
    X_world = R^T * (X_cam - t)
    -----
    Paramètres:
    position_3D_object_camera: la position (vector 3D) de l'objet dans le repère de la caméra
    rotation_vector: le vecteur de rotation du repère par rapport au repère caméra
    translation_vector: le vecteur de translation du repère par rapport au repère caméra
    """
    if position_3D_object_camera is None:
        return None
    
    # # Convertir le vect de rotation en matrix 3x3 
    R, _ = cv2.Rodrigues(rotation_vector)
    
    # # Assurer le vecteur de transition en 3x1
    # t = translation_vector.reshape(3)

    # return R.T @ (position_3D_object_camera - t)

    t = translation_vector.reshape(3, 1)
    X_cam_col = np.array(position_3D_object_camera).reshape(3, 1)
    
    # Appliquer la transformation inverse
    X_world = R.T @ (X_cam_col - t)
    
    return X_world.flatten()

