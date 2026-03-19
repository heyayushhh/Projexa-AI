import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh


class VisualStutterDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

    def detect(self, video_path):
        cap = cv2.VideoCapture(video_path)

        frame_id = 0
        fps = cap.get(cv2.CAP_PROP_FPS)

        prev_points = None
        prev_velocity = 0

        events = []
        in_event = False
        event_start = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]

                # 🔥 USE 3 KEY POINTS
                nose = face.landmark[1]
                forehead = face.landmark[10]
                chin = face.landmark[152]

                current_points = np.array([
                    [nose.x, nose.y],
                    [forehead.x, forehead.y],
                    [chin.x, chin.y]
                ])

                if prev_points is not None:
                    # 🔥 AVERAGE MOVEMENT
                    movement = np.linalg.norm(current_points - prev_points, axis=1)
                    avg_movement = np.mean(movement)

                    # 🔥 VELOCITY
                    velocity = avg_movement

                    # 🔥 ACCELERATION
                    acceleration = abs(velocity - prev_velocity)

                    time = frame_id / fps

                    # 🎯 HEAD JERK CONDITION
                    if velocity > 0.02 and acceleration > 0.01:
                        if not in_event:
                            in_event = True
                            event_start = time
                    else:
                        if in_event:
                            events.append({
                                "start": round(event_start, 2),
                                "end": round(time, 2),
                                "type": "head_jerk"
                            })
                            in_event = False

                    prev_velocity = velocity

                prev_points = current_points

            frame_id += 1

        cap.release()

        # 🔥 CLOSE LAST EVENT
        if in_event:
            events.append({
                "start": round(event_start, 2),
                "end": round(frame_id / fps, 2),
                "type": "head_jerk"
            })

        return events