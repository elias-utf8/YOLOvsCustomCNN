import cv2
import cv2.aruco as aruco
import numpy as np

# # Open the default camera
# cam = cv2.VideoCapture(8)


# # Set the resolution to 640x480
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# # Other possible resolutions: 1920x1080, 640 x 360

# # Get the frame width and height
# frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (frame_width, frame_height))



def findCenter(cornersList, centers):
        for corners in cornersList:
            pts = corners[0]   
            cx, cy = pts.mean(axis=0)
            cx , cy = int(cx), int(cy)
            centers.append((cx, cy))


def findArucoMarkers(frame_color, target_id = None):
    arucoDic = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    arucoParam = aruco.DetectorParameters()
    
    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    (corners , ids, _) = aruco.ArucoDetector(arucoDic, arucoParam).detectMarkers(gray)
    
    if ids is None or len(corners) == 0:
        return [], None
    
    if target_id is None:
        return corners, ids

    # Filtrer avec ID 
    filtered = []
    filtered_ids = []
    for c, i in zip(corners, ids):
        if int(i[0]) == int(target_id):
            filtered.append(c)
            filtered_ids.append([int(i[0])])

    if len(filtered) == 0:
        return [], None
    return filtered, np.array(filtered_ids, dtype=np.int32)



# while True:
#     centers = []
#     ret, frame = cam.read()

#     findArucoMarkers(frame)

#     # Press 'q' to exit the loop
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Release the capture and writer objects
# cam.release()
# out.release()
# cv2.destroyAllWindows()


