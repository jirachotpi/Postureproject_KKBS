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
from pymongo import MongoClient

class PostureStatistics:
    def __init__(self, db_name="ErgoSideDB", collection_name="posture_stats"):
        self.client = MongoClient("mongodb://localhost:27017/") # เชื่อมต่อ Local MongoDB
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # ตรวจสอบว่ามีข้อมูลของวันนี้หรือยัง ถ้าไม่มีให้สร้างขึ้นใหม่
        if not self.collection.find_one({"_id": self.current_date}):
            self._init_day_structure()

    def _init_day_structure(self):
        doc = {
            "_id": self.current_date, # ใช้ วันที่ เป็น ID เลย
            "total_sitting_seconds": 0,
            "states": {
                "GOOD": 0,
                "WARNING": 0,
                "BAD": 0,
                "CRITICAL": 0
            },
            "last_updated": str(datetime.now())
        }
        self.collection.insert_one(doc)

    def update(self, state, duration_sec):
        # ใช้ $inc ของ MongoDB เพื่อเพิ่มค่าโดยตรง (ประสิทธิภาพดีกว่าเขียนไฟล์ใหม่)
        self.collection.update_one(
            {"_id": self.current_date},
            {
                "$inc": {
                    "total_sitting_seconds": duration_sec,
                    f"states.{state}": duration_sec
                },
                "$set": {"last_updated": str(datetime.now())}
            }
        )

    @property
    def data(self):
        # คืนค่าข้อมูลของวันนี้สำหรับ endpoint /statistics
        return self.collection.find_one({"_id": self.current_date}) or {}


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
            model_complexity=1,
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
        
        # Current Metrics (เพื่อให้ API มาดึงค่าไปใช้ได้)
        self.current_state = "GOOD"
        self.current_fhp_ratio = 0.0
        self.current_slump_angle = 0.0
        self.is_user_standing = False

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
        l_vis = sum([landmarks[l.value].visibility for l in self.sides['left'].values()])
        r_vis = sum([landmarks[l.value].visibility for l in self.sides['right'].values()])
        return 'left' if l_vis > r_vis else 'right'

    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_start = time.monotonic()

    def process_frame(self, frame):
        """
        ฟังก์ชันหลัก: รับ 1 เฟรม (BGR), ประมวลผล, วาด Landmark, และคืนค่าเฟรมที่ประมวลผลแล้ว
        """
        # 1. Pre-process
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if not res.pose_landmarks:
            self.current_state = "NO_PERSON"
            self.current_fhp_ratio = 0.0
            self.current_slump_angle = 0.0
            self.is_user_standing = False
            return frame # คืนภาพเปล่าถ้าไม่เจอคน

        lm = res.pose_landmarks.landmark
        active_side = self.detect_best_side(lm)
        side_idx = self.sides[active_side]
        
        nose_lm = lm[self.mp_pose.PoseLandmark.NOSE.value]
        ear_lm = lm[side_idx['ear'].value]
        facing_dir = 1 if nose_lm.x > ear_lm.x else -1

        # 2. Extract Keypoints
        def pt(idx):
            l = lm[idx.value]
            return self.kp_smoother.update(idx.value, l.x * w, l.y * h, l.visibility > 0.4)

        ear = pt(side_idx['ear'])
        sh = pt(side_idx['shoulder'])
        hip_raw = lm[side_idx['hip'].value]
        
        # 3. Handle Hip Occlusion
        hip = None
        if hip_raw.visibility > 0.4:
            hip = pt(side_idx['hip'])
        elif self.baseline_torso is not None and sh is not None:
            hip = (sh[0], int(sh[1] + self.baseline_torso))

        if not (ear and sh and hip):
            self.current_state = "TRACKING"
            self.current_fhp_ratio = 0.0
            self.current_slump_angle = 0.0
            self.is_user_standing = False
            return frame

        # 4. Drawing & Logic
        cv2.line(frame, ear, sh, (255, 200, 0), 2)
        cv2.line(frame, sh, hip, (255, 200, 0), 2)

        # --- CALIBRATION ---
        if self.is_calibrating:
            elapsed = time.monotonic() - self.calibration_start
            cv2.putText(frame, f"CALIBRATING... {5 - int(elapsed)}s", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            if elapsed >= 5.0:
                self.baseline_torso = np.hypot(sh[0]-hip[0], sh[1]-hip[1])
                self.baseline_shoulder_y = sh[1]
                self.is_calibrating = False
        
        # --- ANALYSIS ---
        elif self.baseline_torso is not None:
            # คำนวณ Metric
            raw_fhp = (ear[0] - sh[0]) * facing_dir 
            self.current_fhp_ratio = float(self.fhp_smoother.update(raw_fhp / self.baseline_torso))
            
            dx, dy = sh[0] - hip[0], sh[1] - hip[1]
            self.current_slump_angle = float(abs(90 - abs(np.degrees(np.arctan2(dy, dx)))))
            
            self.is_user_standing = bool((self.baseline_shoulder_y - sh[1]) > (self.baseline_torso * 0.4))

            # Update Timing & Stats
            now = time.monotonic()
            delta_t = now - self.last_tick
            self.last_tick = now

            if self.is_user_standing:
                self.current_state = "AWAY"
                cv2.putText(frame, "STATUS: STANDING/AWAY", (30, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
            else:
                is_bad = self.current_fhp_ratio > 0.15 or self.current_slump_angle > 15.0
                self.current_state = self.state_machine.update(is_bad)
                self.stats.update(self.current_state, delta_t)

                # UI Feedback
                color = {"GOOD": (0,255,0), "WARNING": (0,255,255), 
                         "BAD": (0,140,255), "CRITICAL": (0,0,255)}.get(self.current_state, (255,255,255))
                
                cv2.putText(frame, f"POSTURE: {self.current_state}", (30, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        return frame

if __name__ == "__main__":
    app = PostureMonitorApp()
    app.run()