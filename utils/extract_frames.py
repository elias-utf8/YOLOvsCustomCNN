import cv2
import os

def extraire_frames_espacees(video_path, output_dir, intervalle=10):
    """
    Extrait 1 frame tous les N frames.
    
    Args:
        video_path: chemin vers la vidéo
        output_dir: dossier de sortie
        intervalle: extraire 1 frame tous les N frames (ex: 10)
    """

    os.makedirs(output_dir, exist_ok=True)

    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not capture.isOpened():
        print(f"❌ Erreur : impossible d'ouvrir {video_path}")
        return
    
    print(f"📹 Extraction de 1 frame tous les {intervalle} frames")
    print(f"   Total attendu : ~{total_frames // intervalle} images")
    
    frame_count = 0
    nb_saved_frames = 0

    while True:
        achieve_read, frame = capture.read()

        if not achieve_read:
            print("Pas de frame: fin de vidéo ou erreur")
            break

        if frame_count % intervalle == 0:
            output_path = os.path.join(output_dir, f'frame_{nb_saved_frames:05d}.jpg')
            cv2.imwrite(output_path, frame)
            nb_saved_frames += 1

        frame_count += 1

    capture.release()

extraire_frames_espacees('../output.mp4', '../data/frames/', intervalle=5)