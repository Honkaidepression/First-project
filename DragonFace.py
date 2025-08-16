import cv2
import faceMesh as fmd

# Initialize
detector = fmd.FaceMesh()
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

# Load overlays
left_eye = cv2.imread("eye1.png")
right_eye = cv2.imread("eye2.png")
smoke_animation = cv2.VideoCapture("smoke_animation.mp4")
smoke_frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ret_smoke, smoke_frame = smoke_animation.read()
    if not ret_smoke:
        smoke_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_smoke, smoke_frame = smoke_animation.read()
        smoke_frame_counter = 0

    smoke_frame_counter += 1
    if smoke_frame_counter == smoke_animation.get(cv2.CAP_PROP_FRAME_COUNT):
        smoke_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
        smoke_frame_counter = 0

    frame = cv2.flip(frame, 1)

    # Detect landmarks
    frame_face_mesh, face_mesh_results = detector.detectFacialLandmarks(frame, detector.faceMeshVideos)

    if face_mesh_results.multi_face_landmarks:
        # Check open/close
        _, mouth_status = detector.isOpen(frame, face_mesh_results, 'MOUTH', threshold=15)
        _, left_eye_status = detector.isOpen(frame, face_mesh_results, 'LEFT EYE', threshold=4.5)
        _, right_eye_status = detector.isOpen(frame, face_mesh_results, 'RIGHT EYE', threshold=4.5)

        # Apply filters
        for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
            if left_eye_status.get(face_num) == 'OPEN':
                frame = detector.masking(frame, left_eye, face_landmarks,
                                         'LEFT EYE', detector.mpFaceMesh.FACEMESH_LEFT_EYE)
            if right_eye_status.get(face_num) == 'OPEN':
                frame = detector.masking(frame, right_eye, face_landmarks,
                                         'RIGHT EYE', detector.mpFaceMesh.FACEMESH_RIGHT_EYE)
            if mouth_status.get(face_num) == 'OPEN':
                frame = detector.masking(frame, smoke_frame, face_landmarks,
                                         'MOUTH', detector.mpFaceMesh.FACEMESH_LIPS)

    cv2.imshow('frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
smoke_animation.release()
cv2.destroyAllWindows()