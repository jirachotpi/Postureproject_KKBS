import cv2
import mediapipe as mp
import numpy as np
import time

# ------------------------------
# Geometry Utilities
# ------------------------------

def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx*dx + dy*dy) ** 0.5


def angle_vertical(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    ang = np.degrees(np.arctan2(dy, dx))
    return abs(90 - abs(ang))


# ------------------------------
# EMA Smoothers
# ------------------------------

class KeypointEMA:
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.state = {}

    def update(self, idx, x, y, visible=True):

        if not visible:
            return self.state.get(idx)

        if idx not in self.state:
            self.state[idx] = (int(x), int(y))
        else:
            px, py = self.state[idx]
            x = self.alpha * x + (1 - self.alpha) * px
            y = self.alpha * y + (1 - self.alpha) * py
            self.state[idx] = (int(x), int(y))

        return self.state[idx]


class ScalarEMA:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.v = None

    def update(self, x):
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v


# ------------------------------
# Posture State Machine
# ------------------------------

class PostureState:
    def __init__(self, bad_seconds=1.5, good_seconds=0.5):
        self.bad_seconds = bad_seconds
        self.good_seconds = good_seconds

        self.state = "GOOD"
        self.last_bad_time = None
        self.last_good_time = None

    def update(self, is_bad):
        now = time.monotonic()

        if is_bad:
            self.last_good_time = None
            if self.last_bad_time is None:
                self.last_bad_time = now

            if now - self.last_bad_time >= self.bad_seconds:
                self.state = "BAD"
        else:
            self.last_bad_time = None
            if self.last_good_time is None:
                self.last_good_time = now

            if now - self.last_good_time >= self.good_seconds:
                self.state = "GOOD"

        return self.state


# ------------------------------
# Sitting Timer
# ------------------------------

class SittingMonitor:
    def __init__(self):
        self.start_time = None
        self.duration = 0

    def update(self, seated):
        now = time.time()

        if seated:
            if self.start_time is None:
                self.start_time = now
            self.duration = now - self.start_time
        else:
            self.start_time = None
            self.duration = 0

        return self.duration


# ------------------------------
# MediaPipe Setup
# ------------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True
)

angle_smoother = ScalarEMA()
kp_smoother = KeypointEMA()
state_machine = PostureState()
sitting_monitor = SittingMonitor()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Running... press q to quit")


# ------------------------------
# Main Loop
# ------------------------------

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        mp.solutions.drawing_utils.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        def pt(idx):
            l = lm[idx]
            return kp_smoother.update(
                idx,
                l.x * w,
                l.y * h,
                l.visibility > 0.3
            )

        l_sh = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        r_sh = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        l_ear = pt(mp_pose.PoseLandmark.LEFT_EAR.value)
        l_hip = pt(mp_pose.PoseLandmark.LEFT_HIP.value)
        l_knee = pt(mp_pose.PoseLandmark.LEFT_KNEE.value)

        neck_valid = None not in (l_sh, l_ear)
        full_body_valid = None not in (l_sh, l_ear, l_hip, l_knee)

        if neck_valid:

            raw_angle = angle_vertical(l_ear, l_sh)
            smooth_angle = angle_smoother.update(raw_angle)

            bad = smooth_angle > 25
            state = state_machine.update(bad)

            cv2.line(frame, l_sh, l_ear, (255,255,0), 2)

            color = (0,255,0) if state == "GOOD" else (0,0,255)

            cv2.putText(frame, f"Neck Angle: {smooth_angle:.1f}",
                        (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f"Posture: {state}",
                        (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            if full_body_valid:
                torso = dist(l_sh, l_hip)
                knee_ratio = dist(l_hip, l_knee) / (torso + 1e-6)
                seated = knee_ratio < 1.2
                duration = sitting_monitor.update(seated)
                
                cv2.putText(frame, f"Sitting: {int(duration)}s",
                        (30,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,255), 2)
            else:
                cv2.putText(frame, "Sitting: (Legs not visible)",
                        (30,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)

    cv2.imshow("Posture Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
