import cv2
import numpy as np
import mediapipe as mp
import os
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


class VisualStutterDetector:
    def __init__(self, model_path=None):
        """Initialize MediaPipe Face Landmarker for video mode."""

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        if model_path is None:
            model_path = os.path.join(BASE_DIR, "face_landmarker.task")

        print("MODEL PATH:", model_path)
        print("EXISTS:", os.path.exists(model_path))

        base_options = BaseOptions(model_asset_path=model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Minimum event duration in seconds to filter noise
        self.min_event_duration = 0.1

        # Frame skip factor for performance (process every Nth frame)
        self.frame_skip = 2

    def detect(self, video_path):
        """
        Detect head jerks in video using MediaPipe face landmarks.
        Uses adaptive thresholds computed from the first few seconds of video.
        Returns list of events with start, end, type, confidence, source.
        Preserves original return format and extends with confidence + source.
        """
        cap = cv2.VideoCapture(video_path)

        frame_id = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # fallback

        prev_points = None
        prev_velocity = 0

        events = []
        in_event = False
        event_start = None
        event_max_velocity = 0

        # Collect movement data for adaptive thresholds
        calibration_movements = []
        calibration_duration = 2.0  # seconds to calibrate
        is_calibrating = True

        # Default thresholds (will be overridden by adaptive calculation)
        velocity_threshold = 0.02
        acceleration_threshold = 0.01

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for performance
            if frame_id % self.frame_skip != 0:
                frame_id += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert frame to mediapipe format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp required for VIDEO mode (in ms)
            timestamp = int((frame_id / fps) * 1000)

            result = self.detector.detect_for_video(mp_image, timestamp)

            if result.face_landmarks:
                face = result.face_landmarks[0]

                # 🔥 SAME LANDMARKS as original
                nose = face[1]
                forehead = face[10]
                chin = face[152]

                current_points = np.array([
                    [nose.x, nose.y],
                    [forehead.x, forehead.y],
                    [chin.x, chin.y]
                ])

                if prev_points is not None:
                    movement = np.linalg.norm(current_points - prev_points, axis=1)
                    avg_movement = np.mean(movement)

                    velocity = avg_movement
                    acceleration = abs(velocity - prev_velocity)

                    time = frame_id / fps

                    # Adaptive threshold calibration phase
                    if is_calibrating:
                        calibration_movements.append(velocity)
                        if time >= calibration_duration and len(calibration_movements) > 5:
                            is_calibrating = False
                            mean_vel = np.mean(calibration_movements)
                            std_vel = np.std(calibration_movements)
                            # Set thresholds at mean + 2*std (catches outlier movements)
                            velocity_threshold = max(mean_vel + 2 * std_vel, 0.01)
                            acceleration_threshold = max(velocity_threshold * 0.5, 0.005)
                            print(f"[INFO] Visual adaptive thresholds: vel={velocity_threshold:.4f}, acc={acceleration_threshold:.4f}")

                    # 🎯 HEAD JERK DETECTION (using adaptive thresholds)
                    if velocity > velocity_threshold and acceleration > acceleration_threshold:
                        if not in_event:
                            in_event = True
                            event_start = time
                            event_max_velocity = velocity
                        else:
                            event_max_velocity = max(event_max_velocity, velocity)
                    else:
                        if in_event:
                            event_end = time
                            event_duration = event_end - event_start

                            # Filter short noise events
                            if event_duration >= self.min_event_duration:
                                # Calculate confidence based on how much velocity exceeded threshold
                                confidence = min(1.0, event_max_velocity / (velocity_threshold * 3))
                                events.append({
                                    "start": round(event_start, 2),
                                    "end": round(event_end, 2),
                                    "type": "head_jerk",
                                    "confidence": round(confidence, 3),
                                    "source": "visual"
                                })
                            in_event = False
                            event_max_velocity = 0

                    prev_velocity = velocity

                prev_points = current_points

            frame_id += 1

        cap.release()

        # Close last event
        if in_event:
            event_end = frame_id / fps
            event_duration = event_end - event_start
            if event_duration >= self.min_event_duration:
                confidence = min(1.0, event_max_velocity / (velocity_threshold * 3))
                events.append({
                    "start": round(event_start, 2),
                    "end": round(event_end, 2),
                    "type": "head_jerk",
                    "confidence": round(confidence, 3),
                    "source": "visual"
                })

        return events