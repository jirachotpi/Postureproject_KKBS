import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import json
import os
from datetime import datetime

# ------------------------------
# Statistics & Logging Engine
# ------------------------------
class PostureStatistics:
    def __init__(self, filename="posture_stats.json"):
        self.filename = filename
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.data = self._load_data()

    def _load_data(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    full_data = json.load(f)
                    # Return data for today or init new if date changed
                    return full_data.get(self.current_date, self._init_day_structure())
            except:
                return self._init_day_structure()
        return self._init_day_structure()

    def _init_day_structure(self):
        return {
            "total_sitting_seconds": 0,
            "states": {
                "GOOD": 0,
                "WARNING": 0,
                "BAD": 0,
                "CRITICAL": 0
            },
            "last_updated": str(datetime.now())
        }

    def update(self, state, duration_sec):
        # Update total time and specific state time
        self.data["total_sitting_seconds"] += duration_sec
        if state in self.data["states"]:
            self.data["states"][state] += duration_sec
        
        self.data["last_updated"] = str(datetime.now())
        self._save_to_disk()

    def _save_to_disk(self):
        # Read all historical data first
        all_history = {}
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                try: all_history = json.load(f)
                except: pass
        
        # Merge current day and save
        all_history[self.current_date] = self.data
        with open(self.filename, 'w') as f:
            json.dump(all_history, f, indent=4)

# ------------------------------
# Geometry & Math Utilities
# ------------------------------
def dist(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def angle_vertical(p1, p2):
    # Calculate angle relative to absolute vertical (gravity) assuming camera is horizontal
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    ang = np.degrees(np.arctan2(dy, dx))
    return abs(90 - abs(ang))

# ------------------------------
# Advanced Smoothers
# ------------------------------
class AdaptiveKeypointEMA:
    def __init__(self, base_alpha=0.2, fast_alpha=0.7, diff_thresh=30.0):
        self.base_alpha = base_alpha
        self.fast_alpha = fast_alpha
        self.diff_thresh = diff_thresh
        self.state = {}

    def update(self, idx, x, y, visible=True):
        if not visible:
            return self.state.get(idx)

        if idx not in self.state:
            self.state[idx] = (x, y)
        else:
            px, py = self.state[idx]
            d = dist((x, y), (px, py))
            # If moving fast, adapt alpha to catch up quickly and reduce lag
            alpha = self.fast_alpha if d > self.diff_thresh else self.base_alpha
            new_x = alpha * x + (1 - alpha) * px
            new_y = alpha * y + (1 - alpha) * py
            self.state[idx] = (new_x, new_y)

        return (int(self.state[idx][0]), int(self.state[idx][1]))

class MedianSmoother:
    def __init__(self, window_size=5):
        self.q = deque(maxlen=window_size)
    
    def update(self, val):
        self.q.append(val)
        return np.median(self.q)

# ------------------------------
# Posture State Machine
# ------------------------------
class ErgonomicStateMachine:
    def __init__(self):
        self.state = "GOOD"
        self.bad_posture_start = None
        
        # Thresholds in seconds (Time-over-Threshold)
        self.WARN_SEC = 15.0
        self.BAD_SEC = 30.0
        self.CRIT_SEC = 60.0

    def update(self, is_bad_posture):
        now = time.monotonic()

        if is_bad_posture:
            if self.bad_posture_start is None:
                self.bad_posture_start = now
            
            elapsed = now - self.bad_posture_start
            if elapsed >= self.CRIT_SEC:
                self.state = "CRITICAL"
            elif elapsed >= self.BAD_SEC:
                self.state = "BAD"
            elif elapsed >= self.WARN_SEC:
                self.state = "WARNING"
            else:
                self.state = "GOOD" # Still buffering
        else:
            self.bad_posture_start = None
            self.state = "GOOD"

        return self.state

# ------------------------------
# Core Application Logic
# ------------------------------
class PostureMonitorApp:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.stats = PostureStatistics()
        self.last_tick = time.monotonic()
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # Increased for better accuracy on side views
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True
        )
        
        self.kp_smoother = AdaptiveKeypointEMA()
        self.fhp_smoother = MedianSmoother(window_size=10)
        self.state_machine = ErgonomicStateMachine()
        
        # Calibration States
        self.is_calibrating = False
        self.calibration_start = 0
        self.baseline_torso = None
        self.baseline_shoulder_y = None
        
        # Landmarks mapping (Left vs Right)
        self.sides = {
            'left': {
                'ear': self.mp_pose.PoseLandmark.LEFT_EAR,
                'shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                'hip': self.mp_pose.PoseLandmark.LEFT_HIP
            },
            'right': {
                'ear': self.mp_pose.PoseLandmark.RIGHT_EAR,
                'shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                'hip': self.mp_pose.PoseLandmark.RIGHT_HIP
            }
        }

    def detect_best_side(self, landmarks):
        # Compare average visibility of left vs right landmarks
        l_vis = sum([landmarks[l.value].visibility for l in self.sides['left'].values()])
        r_vis = sum([landmarks[l.value].visibility for l in self.sides['right'].values()])
        return 'left' if l_vis > r_vis else 'right'

    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_start = time.monotonic()
        print("Started Calibration. Please sit upright...")

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("System Ready. Press 'c' to Calibrate, 'q' to Quit.")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: continue

            frame = cv2.flip(frame, 1) # Mirror
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                
                # 1. Detect active side and face direction
                active_side = self.detect_best_side(lm)
                side_idx = self.sides[active_side]
                
                nose_lm = lm[self.mp_pose.PoseLandmark.NOSE.value]
                ear_lm = lm[side_idx['ear'].value]
                
                # If nose.x > ear.x, facing right (+1), else facing left (-1)
                facing_dir = 1 if nose_lm.x > ear_lm.x else -1

                # 2. Extract and smooth keypoints
                def pt(idx):
                    l = lm[idx.value]
                    return self.kp_smoother.update(idx.value, l.x * w, l.y * h, l.visibility > 0.4)

                ear = pt(side_idx['ear'])
                sh = pt(side_idx['shoulder'])
                hip_raw = lm[side_idx['hip'].value]
                
                # 3. Handle Hip Occlusion & Calibration Proxy
                hip = None
                if hip_raw.visibility > 0.4:
                    hip = pt(side_idx['hip'])
                elif self.baseline_torso is not None and sh is not None:
                    # Proxy hip position using baseline (dropping down vertically)
                    hip = (sh[0], int(sh[1] + self.baseline_torso))

                if ear and sh and hip:
                    # Draw skeleton
                    cv2.line(frame, ear, sh, (255, 200, 0), 2)
                    cv2.line(frame, sh, hip, (255, 200, 0), 2)
                    cv2.circle(frame, ear, 5, (0, 255, 255), -1)
                    cv2.circle(frame, sh, 5, (0, 255, 255), -1)
                    cv2.circle(frame, hip, 5, (0, 255, 255), -1)

                    # --- CALIBRATION LOGIC ---
                    if self.is_calibrating:
                        elapsed = time.monotonic() - self.calibration_start
                        cv2.putText(frame, f"Calibrating... {5 - int(elapsed)}s", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                        
                        if elapsed >= 5.0:
                            self.baseline_torso = dist(sh, hip)
                            self.baseline_shoulder_y = sh[1]
                            self.is_calibrating = False
                            print(f"Calibration Done. Torso Baseline: {self.baseline_torso:.1f}px")

                    # --- POSTURE ANALYSIS LOGIC ---
                    elif self.baseline_torso is not None:
                        # 1. Normalize Torso Length (Current vs Baseline logic)
                        torso_len = self.baseline_torso
                        
                        # 2. Forward Head Posture (FHP)
                        # Horizontal distance between ear and shoulder, aware of face direction
                        raw_fhp = (ear[0] - sh[0]) * facing_dir 
                        norm_fhp = raw_fhp / torso_len
                        smooth_fhp = self.fhp_smoother.update(norm_fhp)

                        # 3. Slump Proxy (Spine angle approximation)
                        slump_angle = angle_vertical(sh, hip)
                        
                        # Detect standing up (If shoulder moves up significantly compared to baseline)
                        is_standing = (self.baseline_shoulder_y - sh[1]) > (torso_len * 0.4)

                        now = time.monotonic()
                        delta_t = now - self.last_tick
                        self.last_tick = now
                        
                        if is_standing:
                            state = self.state_machine.update(is_bad)
                            self.stats.update(state, delta_t)
                            cv2.putText(frame, "STATUS: STANDING/AWAY", (30, 80), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                            self.state_machine.update(False) # Pause timer
                        else:
                            # Evaluate Posture
                            # FHP > 0.15 torso length is considered warning threshold
                            # Slump angle > 15 degrees is considered slouched
                            is_bad = smooth_fhp > 0.15 or slump_angle > 15.0
                            state = self.state_machine.update(is_bad)

                            # Colors based on state
                            colors = {"GOOD": (0,255,0), "WARNING": (0,255,255), 
                                      "BAD": (0,140,255), "CRITICAL": (0,0,255)}
                            color = colors[state]

                            cv2.putText(frame, f"FHP Ratio: {smooth_fhp:.2f} (Threshold 0.15)",
                                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.putText(frame, f"Slump Angle: {slump_angle:.1f} deg",
                                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.putText(frame, f"POSTURE: {state}",
                                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                            
                            if state != "GOOD":
                                cv2.putText(frame, "Please sit back and align your neck.",
                                            (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        cv2.putText(frame, "PRESS 'c' TO CALIBRATE POSTURE", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Landmarks not fully visible. Please adjust camera.", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Ergonomic Side-View Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.start_calibration()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PostureMonitorApp()
    app.run()