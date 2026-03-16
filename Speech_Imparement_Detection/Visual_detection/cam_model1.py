import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

prev_nose = None
lip_history = deque(maxlen=10)

HEAD_JERK_THRESHOLD = 25
TREMOR_THRESHOLD = 3
FREEZE_FRAMES = 15

freeze_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        nose = face.landmark[1]
        upper_lip = face.landmark[13]
        lower_lip = face.landmark[14]
        left_lip = face.landmark[61]
        right_lip = face.landmark[291]

        nose_xy = np.array([nose.x*w, nose.y*h])
        upper_xy = np.array([upper_lip.x*w, upper_lip.y*h])
        lower_xy = np.array([lower_lip.x*w, lower_lip.y*h])
        left_xy = np.array([left_lip.x*w, left_lip.y*h])
        right_xy = np.array([right_lip.x*w, right_lip.y*h])

        # ---- Head jerk detection ----
        if prev_nose is not None:
            movement = np.linalg.norm(nose_xy - prev_nose)

            if movement > HEAD_JERK_THRESHOLD:
                cv2.putText(frame,"HEAD JERK",(30,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        prev_nose = nose_xy

        # ---- Mouth opening ----
        mouth_open = np.linalg.norm(upper_xy - lower_xy)

        cv2.putText(frame,f"Mouth: {int(mouth_open)}",(30,100),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)

        # ---- Lip tremor detection ----
        lip_width = np.linalg.norm(left_xy - right_xy)
        lip_history.append(lip_width)

        if len(lip_history) > 5:
            tremor = np.std(lip_history)

            if tremor > TREMOR_THRESHOLD:
                cv2.putText(frame,"LIP TREMOR",(30,150),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        # ---- Mouth freeze detection ----
        if mouth_open > 10:
            if len(lip_history) > 2:
                change = abs(lip_history[-1] - lip_history[-2])

                if change < 0.5:
                    freeze_counter += 1
                else:
                    freeze_counter = 0

                if freeze_counter > FREEZE_FRAMES:
                    cv2.putText(frame,"BLOCK DETECTED",(30,200),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        # draw landmarks (optional)
        for landmark in face.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame,(x,y),1,(0,255,0),-1)

    cv2.imshow("Visual Stutter Detection",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()