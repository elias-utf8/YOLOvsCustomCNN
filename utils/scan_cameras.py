"""
Scanner de caméras disponibles
===============================
Trouve toutes les caméras disponibles sur le système.
"""

import cv2

def find_available_cameras(max_cameras=10):
    """
    Scanne les indices de caméra de 0 à max_cameras-1.
    
    Args:
        max_cameras: Nombre maximum d'indices à tester
        
    Returns:
        Liste des indices de caméras disponibles
    """
    available_cameras = []
    
    print(f"\n{'='*60}")
    print("SCAN DES CAMÉRAS DISPONIBLES")
    print(f"{'='*60}\n")
    print(f"Test de {max_cameras} indices de caméra...\n")
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                print(f"✅ Caméra {i} trouvée:")
                print(f"   Résolution: {width}x{height}")
                print(f"   FPS: {fps}")
                
                available_cameras.append(i)
            cap.release()
        else:
            print(f"❌ Caméra {i}: non disponible")
    
    print(f"\n{'='*60}")
    print(f"Résumé: {len(available_cameras)} caméra(s) disponible(s)")
    print(f"{'='*60}\n")
    
    return available_cameras


def test_camera(camera_id):
    """
    Test une caméra spécifique.
    
    Args:
        camera_id: Index de la caméra à tester
    """
    print(f"\n{'='*60}")
    print(f"TEST DE LA CAMÉRA {camera_id}")
    print(f"{'='*60}\n")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la caméra {camera_id}")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Caméra ouverte")
    print(f"   Résolution: {width}x{height}")
    print(f"\nAppuyez sur 'q' pour quitter\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Erreur de lecture")
                break
            
            cv2.imshow(f'Camera {camera_id} - Appuyez sur q pour quitter', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"❌ Erreur lors de l'affichage: {e}")
        import traceback
        traceback.print_exc()
    
    cap.release()
    cv2.destroyAllWindows()
    
    return True


def main():
    """Menu principal."""
    # Scanner les caméras
    available = find_available_cameras(max_cameras=40)
    
    if len(available) == 0:
        print("❌ Aucune caméra disponible")
        return
    
    # Si une seule caméra, la tester directement
    if len(available) == 1:
        print(f"Une seule caméra trouvée (ID {available[0]})")
        test_camera(available[0])
        return
    
    # Sinon, proposer de choisir
    print(f"Caméras disponibles: {available}")
    
    try:
        choice = input(f"\nQuelle caméra tester? ({'/'.join(map(str, available))}): ").strip()
        camera_id = int(choice)
        
        if camera_id in available:
            test_camera(camera_id)
        else:
            print(f"❌ Caméra {camera_id} non disponible")
    except:
        print("❌ Choix invalide")


if __name__ == "__main__":
    main()
