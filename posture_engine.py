import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from datetime import datetime
import json
import os
from pymongo import MongoClient


# =====================================================================
# Statistics & Logging Engine
# =====================================================================
class PostureStatistics:
    def __init__(self, db_name="ErgoSideDB", collection_name="posture_stats"):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        if not self.collection.find_one({"_id": self.current_date}):
            self._init_day_structure()

    def _init_day_structure(self):
        self.collection.insert_one({
            "_id": self.current_date,
            "total_sitting_seconds": 0,
            "states": {"GOOD": 0, "WARNING": 0, "BAD": 0, "CRITICAL": 0},
            "last_updated": str(datetime.now()),
        })

    def update(self, state, duration_sec):
        self.collection.update_one(
            {"_id": self.current_date},
            {
                "$inc": {"total_sitting_seconds": duration_sec, f"states.{state}": duration_sec},
                "$set": {"last_updated": str(datetime.now())},
            },
        )

    @property
    def data(self):
        return self.collection.find_one({"_id": self.current_date}) or {}


# =====================================================================
# Configuration Manager
# =====================================================================
class PostureConfig:
    def __init__(self, filename="config.json"):
        self.filename = filename
        self.defaults = {
            "slump_threshold": 15.0,
            "fhp_threshold": 0.15,
            "camera_source": "0"
        }
        self.data = self.defaults.copy()
        self.data = self.load()

    def load(self):
        if not os.path.exists(self.filename):
            self.save(self.defaults)
            return self.defaults.copy()
        try:
            with open(self.filename, 'r') as f:
                return {**self.defaults, **json.load(f)}
        except:
            return self.defaults.copy()

    def save(self, data):
        self.data.update(data)
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=4)


# =====================================================================
# Geometry Utilities
# =====================================================================
def dist(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])


def angle_3pts(a, b, c):
    """Angle at vertex b in degrees."""
    v1 = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    v2 = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


# =====================================================================
# Smoothers
# =====================================================================
class AdaptiveKeypointEMA:
    def __init__(self, base_alpha=0.2, fast_alpha=0.7, diff_thresh=30.0):
        self.base_alpha = base_alpha
        self.fast_alpha = fast_alpha
        self.diff_thresh = diff_thresh
        self.state = {}

    def update(self, idx, x, y, visible=True):
        if not visible:
            prev = self.state.get(idx)
            if prev is None:
                return None
            return (int(prev[0]), int(prev[1]))

        if idx not in self.state:
            self.state[idx] = (float(x), float(y))
        else:
            px, py = self.state[idx]
            alpha = self.fast_alpha if dist((x, y), (px, py)) > self.diff_thresh else self.base_alpha
            self.state[idx] = (
                alpha * float(x) + (1 - alpha) * float(px),
                alpha * float(y) + (1 - alpha) * float(py),
            )

        return (int(self.state[idx][0]), int(self.state[idx][1]))


class MedianSmoother:
    def __init__(self, window_size=7):
        self.q = deque(maxlen=window_size)

    def update(self, val):
        self.q.append(val)
        return float(np.median(self.q))


# =====================================================================
# Adaptive Thresholds — learned during calibration
# =====================================================================
class AdaptiveThresholds:
    """
    Stores per-metric warning/bad/critical thresholds.
    Default values follow ergonomic industry standards.
    After calibration, thresholds are personalised from the user's baseline.
    """

    def __init__(self):
        self.fhp:          dict = {"warn": 0.10, "bad": 0.18, "crit": 0.28}
        self.slump:        dict = {"warn": 10.0, "bad": 18.0, "crit": 28.0}
        self.arm_raise:    dict = {"warn": 35.0, "bad": 55.0, "crit": 75.0}
        self.shoulder_sym: dict = {"warn": 0.05, "bad": 0.10, "crit": 0.16}
        self._bufs: dict = {k: [] for k in ("fhp", "slump", "arm_raise", "shoulder_sym")}
        self.is_calibrated: bool = False

    def collect(self, fhp, slump, arm_raise, shoulder_sym):
        self._bufs["fhp"].append(abs(fhp))
        self._bufs["slump"].append(slump)
        self._bufs["arm_raise"].append(arm_raise)
        self._bufs["shoulder_sym"].append(shoulder_sym)

    def finalize(self):
        def build(buf, kw, kb, kc, fw, fb, fc):
            if len(buf) < 5:
                return None
            mu, sigma = float(np.mean(buf)), float(np.std(buf))
            return {
                "warn": max(fw, mu + kw * sigma),
                "bad":  max(fb, mu + kb * sigma),
                "crit": max(fc, mu + kc * sigma),
            }

        r = {
            "fhp":          build(self._bufs["fhp"],       1.5, 3.0, 5.0, 0.07, 0.13, 0.22),
            "slump":        build(self._bufs["slump"],      1.5, 3.0, 5.0,  8.0, 14.0, 24.0),
            "arm_raise":    build(self._bufs["arm_raise"],  1.5, 3.0, 5.0, 30.0, 50.0, 70.0),
            "shoulder_sym": build(self._bufs["shoulder_sym"], 2.0, 4.0, 6.0, 0.04, 0.08, 0.13),
        }
        if r["fhp"]:          self.fhp          = r["fhp"]
        if r["slump"]:        self.slump        = r["slump"]
        if r["arm_raise"]:    self.arm_raise    = r["arm_raise"]
        if r["shoulder_sym"]: self.shoulder_sym = r["shoulder_sym"]

        for b in self._bufs.values():
            b.clear()
        self.is_calibrated = True


# =====================================================================
# Per-Metric State Machine
# =====================================================================
class MetricStateMachine:
    """One instance per metric (head / spine / arm / shoulder).
    Faster time-constants than the overall machine for granular feedback."""
    WARN_SEC = 8.0
    BAD_SEC  = 20.0
    CRIT_SEC = 45.0

    def __init__(self, name: str):
        self.name = name
        self.state: str = "GOOD"
        self._bad_start: float | None = None

    def update(self, is_bad: bool) -> str:
        now = time.monotonic()
        if is_bad:
            if self._bad_start is None:
                self._bad_start = now
            elapsed = now - self._bad_start
            if   elapsed >= self.CRIT_SEC: self.state = "CRITICAL"
            elif elapsed >= self.BAD_SEC:  self.state = "BAD"
            elif elapsed >= self.WARN_SEC: self.state = "WARNING"
            else:                          self.state = "GOOD"
        else:
            self._bad_start = None
            self.state = "GOOD"
        return self.state


# =====================================================================
# Weighted Risk Score (0–100)
# =====================================================================
class RiskScoreCalculator:
    """
    Weights: head 30 | spine 35 | arm 20 | shoulder 15
    Each sub-score uses piecewise linear mapping:
       0 → warn  →  0–50
      warn → crit → 50–100
    """
    W = {"head": 30, "spine": 35, "arm": 20, "shoulder": 15}

    @staticmethod
    def _norm(v, tw, tc):
        v = max(v, 0.0)
        if v <= tw:
            return 50.0 * v / (tw + 1e-9)
        return 50.0 + 50.0 * min((v - tw) / (tc - tw + 1e-9), 1.0)

    def compute(self, fhp, slump, arm_raise, shoulder_sym, th: AdaptiveThresholds):
        s = {
            "head":     self._norm(abs(fhp),      th.fhp["warn"],          th.fhp["crit"]),
            "spine":    self._norm(slump,          th.slump["warn"],        th.slump["crit"]),
            "arm":      self._norm(arm_raise,      th.arm_raise["warn"],    th.arm_raise["crit"]),
            "shoulder": self._norm(shoulder_sym,   th.shoulder_sym["warn"], th.shoulder_sym["crit"]),
        }
        total = sum(s[k] * self.W[k] for k in s) / 100.0
        return min(round(total, 1), 100.0), {k: round(v, 1) for k, v in s.items()}


# =====================================================================
# Overall Ergonomic State Machine (risk-score driven)
# =====================================================================
class ErgonomicStateMachine:
    WARN_SEC = 15.0
    BAD_SEC  = 30.0
    CRIT_SEC = 60.0
    RISK_THRESHOLD = 28.0   # score > 28 / 100 is considered "problematic"

    def __init__(self):
        self.state: str = "GOOD"
        self._bad_start: float | None = None

    def update(self, risk_score: float) -> str:
        now = time.monotonic()
        is_bad = risk_score > self.RISK_THRESHOLD
        if is_bad:
            if self._bad_start is None:
                self._bad_start = now
            elapsed = now - self._bad_start
            if   elapsed >= self.CRIT_SEC: self.state = "CRITICAL"
            elif elapsed >= self.BAD_SEC:  self.state = "BAD"
            elif elapsed >= self.WARN_SEC: self.state = "WARNING"
            else:                          self.state = "GOOD"
        else:
            self._bad_start = None
            self.state = "GOOD"
        return self.state


# =====================================================================
# Drawing Helpers
# =====================================================================
STATE_COLORS = {
    "GOOD":     (80,  220, 100),
    "WARNING":  (0,   200, 255),
    "BAD":      (0,   140, 255),
    "CRITICAL": (60,   60, 255),
    "AWAY":     (180, 180, 180),
    "TRACKING": (140, 140, 140),
    "NO_PERSON":(100, 100, 100),
}


def _lerp(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return (int(c1[0]+(c2[0]-c1[0])*t), int(c1[1]+(c2[1]-c1[1])*t), int(c1[2]+(c2[2]-c1[2])*t))


def _risk_color(score):
    if score < 25:   return _lerp((80, 220, 100), (0, 200, 255),  score / 25.0)
    if score < 50:   return _lerp((0, 200, 255),  (0, 140, 255),  (score-25)/25.0)
    return               _lerp((0, 140, 255),  (60,  60, 255),  min((score-50)/50.0, 1.0))


def _glow_line(frame, p1, p2, color, thickness=2):
    if p1 is None or p2 is None:
        return
    cv2.line(frame, p1, p2, _lerp(color, (0,0,0), 0.55), thickness+8, cv2.LINE_AA)
    cv2.line(frame, p1, p2, _lerp(color, (255,255,255), 0.20), thickness+3, cv2.LINE_AA)
    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def _glow_circle(frame, center, radius, color):
    if center is None:
        return
    cv2.circle(frame, center, radius+5, _lerp(color,(0,0,0),0.6), 2, cv2.LINE_AA)
    cv2.circle(frame, center, radius+2, _lerp(color,(255,255,255),0.3), 2, cv2.LINE_AA)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, center, max(1, radius//3), (255,255,255), -1, cv2.LINE_AA)


def _bezier_bone(frame, p1, p2, color, thickness=2, n=24):
    """Quadratic Bézier curve between two joints."""
    if p1 is None or p2 is None:
        return
    x1, y1 = p1; x2, y2 = p2
    mx, my = (x1+x2)/2, (y1+y2)/2
    dx, dy = x2-x1, y2-y1
    L = np.hypot(dx, dy) + 1e-9
    # control point: perpendicular offset = 8 % of bone length
    cx, cy = mx - dy/L*L*0.08, my + dx/L*L*0.08
    pts = [
        (int((1-t)**2*x1 + 2*(1-t)*t*cx + t**2*x2),
         int((1-t)**2*y1 + 2*(1-t)*t*cy + t**2*y2))
        for t in (i/n for i in range(n+1))
    ]
    for i in range(len(pts)-1):
        _glow_line(frame, pts[i], pts[i+1], color, thickness)


def draw_skeleton(frame, kpts: dict, metric_states: dict):
    """
    Colour each bone segment by its owning metric state.
    ear→shoulder  = HEAD state
    shoulder→hip  = SPINE state
    shoulder/elbow/wrist = ARM state
    left_shoulder↔right_shoulder = SHOULDER SYMMETRY state
    """
    ch = STATE_COLORS.get(metric_states.get("head",  "GOOD"), (200,200,200))
    cs = STATE_COLORS.get(metric_states.get("spine", "GOOD"), (200,200,200))
    ca = STATE_COLORS.get(metric_states.get("arm",   "GOOD"), (200,200,200))
    cy = STATE_COLORS.get(metric_states.get("shoulder","GOOD"), (200,200,200))

    bones = [
        ("ear",      "shoulder", ch),
        ("shoulder", "hip",      cs),
        ("shoulder", "elbow",    ca),
        ("elbow",    "wrist",    ca),
    ]
    for a, b, col in bones:
        _bezier_bone(frame, kpts.get(a), kpts.get(b), col)

    # Cross-shoulder symmetry line
    _glow_line(frame, kpts.get("left_shoulder"), kpts.get("right_shoulder"), cy, 1)

    # Joints
    for name, radius, col in [
        ("ear", 6, ch), ("shoulder", 9, cs), ("hip", 7, cs),
        ("elbow", 7, ca), ("wrist", 5, ca),
        ("left_shoulder", 5, cy), ("right_shoulder", 5, cy),
    ]:
        _glow_circle(frame, kpts.get(name), radius, col)


def draw_hud(frame, state, risk_score, metric_scores, metric_states,
             raw_metrics, is_calibrated):
    """Semi-transparent left panel: risk gauge + 4 per-metric bars."""
    pw, ph = 260, 220
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8+pw, 8+ph), (10, 10, 18), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    rc = _risk_color(risk_score)
    cv2.rectangle(frame, (8, 8), (8+pw, 8+ph), rc, 1, cv2.LINE_AA)
    cv2.line(frame, (12, 12), (12, ph+6), rc, 3, cv2.LINE_AA)

    f = cv2.FONT_HERSHEY_SIMPLEX

    # Risk score
    cv2.putText(frame, "RISK SCORE", (24, 34), f, 0.40, (160,160,160), 1, cv2.LINE_AA)
    s_str = f"{risk_score:.0f}"
    cv2.putText(frame, s_str, (24, 70), f, 1.40, rc, 3, cv2.LINE_AA)
    cv2.putText(frame, "/ 100", (24+len(s_str)*23, 62), f, 0.50, (130,130,130), 1, cv2.LINE_AA)

    # Overall state
    sc = STATE_COLORS.get(state, (200,200,200))
    cv2.putText(frame, state, (24, 92), f, 0.62, sc, 2, cv2.LINE_AA)

    if is_calibrated:
        cv2.putText(frame, "[CALIBRATED]", (140, 92), f, 0.32, (80,200,80), 1, cv2.LINE_AA)

    cv2.line(frame, (20, 100), (8+pw-10, 100), (50,50,65), 1, cv2.LINE_AA)

    # Per-metric bars
    bar_w = pw - 58
    labels = [
        ("HEAD",     "head",     metric_scores.get("head",0),     metric_states.get("head","GOOD")),
        ("SPINE",    "spine",    metric_scores.get("spine",0),    metric_states.get("spine","GOOD")),
        ("ARM",      "arm",      metric_scores.get("arm",0),      metric_states.get("arm","GOOD")),
        ("SH. SYM.", "shoulder", metric_scores.get("shoulder",0), metric_states.get("shoulder","GOOD")),
    ]
    y = 116
    for lbl, _, mscore, mstate in labels:
        mc = STATE_COLORS.get(mstate, (180,180,180))
        cv2.putText(frame, lbl, (22, y), f, 0.33, (200,200,200), 1, cv2.LINE_AA)
        state_initial = mstate[0]
        cv2.putText(frame, state_initial, (22+bar_w+8, y), f, 0.35, mc, 1, cv2.LINE_AA)
        y += 5
        cv2.rectangle(frame, (22, y), (22+bar_w, y+7), (38,38,50), -1)
        fw = int(bar_w * min(mscore / 100.0, 1.0))
        if fw > 0:
            cv2.rectangle(frame, (22, y), (22+fw, y+7), mc, -1)
        y += 17

    if not is_calibrated:
        cv2.putText(frame, "[ CALIBRATE to personalise ]", (18, y+4), f, 0.32, (0,200,255), 1, cv2.LINE_AA)


# =====================================================================
# Core Application
# =====================================================================
class PostureMonitorApp:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.stats   = PostureStatistics()
        self.last_tick = time.monotonic()
        self.last_db_write = time.monotonic()

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True,
        )

        self.kp_smoother  = AdaptiveKeypointEMA()
        self.fhp_smoother = MedianSmoother(window_size=10)
        self.thresholds   = AdaptiveThresholds()
        self.risk_calc    = RiskScoreCalculator()
        self.state_machine = ErgonomicStateMachine()

        # Per-metric state machines
        self.sm_head     = MetricStateMachine("head")
        self.sm_spine    = MetricStateMachine("spine")
        self.sm_arm      = MetricStateMachine("arm")
        self.sm_shoulder = MetricStateMachine("shoulder")

        # Config & Persistence
        self.config = PostureConfig()
        self.apply_config(self.config.data)

        # Calibration
        self.is_calibrating:      bool = False
        self.calibration_start:   float = 0.0
        self.baseline_torso:      float | None = None
        self.baseline_shoulder_y: float | None = None

        # Exported metrics
        self.current_state         = "GOOD"
        self.current_risk_score    = 0.0
        self.current_metric_scores = {"head": 0.0, "spine": 0.0, "arm": 0.0, "shoulder": 0.0}
        self.current_metric_states = {"head": "GOOD", "spine": "GOOD", "arm": "GOOD", "shoulder": "GOOD"}
        self.current_fhp_ratio     = 0.0
        self.current_slump_angle   = 0.0
        self.current_arm_raise     = 0.0
        self.current_arm_elbow     = 0.0
        self.current_shoulder_sym  = 0.0
        self.is_user_standing      = False

        # Landmark index maps per side
        PL = self.mp_pose.PoseLandmark
        self.sides = {
            "left":  {"ear": PL.LEFT_EAR,  "shoulder": PL.LEFT_SHOULDER,
                      "elbow": PL.LEFT_ELBOW,  "wrist": PL.LEFT_WRIST,  "hip": PL.LEFT_HIP},
            "right": {"ear": PL.RIGHT_EAR, "shoulder": PL.RIGHT_SHOULDER,
                      "elbow": PL.RIGHT_ELBOW, "wrist": PL.RIGHT_WRIST, "hip": PL.RIGHT_HIP},
        }
        self._PL = PL

    # ------------------------------------------------------------------
    def detect_best_side(self, lm):
        l_vis = sum(lm[e.value].visibility for e in self.sides["left"].values())
        r_vis = sum(lm[e.value].visibility for e in self.sides["right"].values())
        return "left" if l_vis > r_vis else "right"

        self.is_calibrating    = True
        self.calibration_start = time.monotonic()
        self.thresholds        = AdaptiveThresholds()   # reset
        # Re-apply manual config overrides if they exist
        self.apply_config(self.config.data)

    def apply_config(self, data: dict):
        """Manually override thresholds from config."""
        if "slump_threshold" in data:
            self.thresholds.slump["warn"] = float(data["slump_threshold"])
        if "fhp_threshold" in data:
            self.thresholds.fhp["warn"] = float(data["fhp_threshold"])
        self.config.save(data)

    # ------------------------------------------------------------------
    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if not res.pose_landmarks:
            self.current_state = "NO_PERSON"
            self.current_risk_score = 0.0
            return frame

        lm = res.pose_landmarks.landmark

        def pt(enum):
            l = lm[enum.value]
            return self.kp_smoother.update(enum.value, l.x*w, l.y*h, l.visibility > 0.4)

        # ── Active side ──────────────────────────────────────────────
        side = self.detect_best_side(lm)
        si   = self.sides[side]

        nose_lm = lm[self._PL.NOSE.value]
        ear_lm  = lm[si["ear"].value]
        facing_dir = 1 if nose_lm.x > ear_lm.x else -1

        # ── Extract keypoints ────────────────────────────────────────
        ear   = pt(si["ear"])
        sh    = pt(si["shoulder"])
        elbow = pt(si["elbow"])
        wrist = pt(si["wrist"])

        # Both shoulders for symmetry (always)
        lsh = pt(self._PL.LEFT_SHOULDER)
        rsh = pt(self._PL.RIGHT_SHOULDER)

        # Hip with occlusion fallback
        hip_raw = lm[si["hip"].value]
        hip = None
        if hip_raw.visibility > 0.4:
            hip = pt(si["hip"])
        elif self.baseline_torso is not None and sh is not None:
            hip = (sh[0], int(sh[1] + self.baseline_torso))

        if not (ear and sh and hip):
            self.current_state = "TRACKING"
            return frame

        # ── Skeleton (colored by last metric states) ─────────────────
        kpts = {"ear": ear, "shoulder": sh, "hip": hip,
                "elbow": elbow, "wrist": wrist,
                "left_shoulder": lsh, "right_shoulder": rsh}
        draw_skeleton(frame, kpts, self.current_metric_states)

        # ── CALIBRATION ────────────────────────────────────────────
        if self.is_calibrating:
            elapsed   = time.monotonic() - self.calibration_start
            countdown = max(0, 5 - int(elapsed))

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 64), (0, 110, 190), -1)
            cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)
            cv2.putText(frame, f"CALIBRATING  {countdown}s  — sit naturally",
                        (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255,255,255), 2, cv2.LINE_AA)

            # Collect live samples for personalisation
            if self.baseline_torso and self.baseline_torso > 0:
                raw_fhp = (ear[0] - sh[0]) * facing_dir / self.baseline_torso
                dx, dy  = sh[0]-hip[0], sh[1]-hip[1]
                raw_slump = abs(90 - abs(np.degrees(np.arctan2(dy, dx))))
                raw_arm   = angle_3pts(hip, sh, elbow) if elbow else 0.0
                raw_sym   = abs(lsh[1] - rsh[1]) / self.baseline_torso if (lsh and rsh) else 0.0
                self.thresholds.collect(raw_fhp, raw_slump, raw_arm, raw_sym)

            if elapsed >= 5.0:
                self.baseline_torso      = np.hypot(sh[0]-hip[0], sh[1]-hip[1])
                self.baseline_shoulder_y = sh[1]
                self.thresholds.finalize()
                self.is_calibrating = False

        # ── ANALYSIS ───────────────────────────────────────────────
        elif self.baseline_torso is not None:

            TL: float = self.baseline_torso

            # Ensure shoulder_y baseline is set (should always be, but guard anyway)
            if self.baseline_shoulder_y is None:
                self.baseline_shoulder_y = float(sh[1])
            base_sh_y: float = self.baseline_shoulder_y

            # -- Raw metrics --
            raw_fhp = (ear[0] - sh[0]) * facing_dir
            self.current_fhp_ratio = float(self.fhp_smoother.update(raw_fhp / TL))

            dx, dy = sh[0]-hip[0], sh[1]-hip[1]
            self.current_slump_angle = float(abs(90 - abs(np.degrees(np.arctan2(dy, dx)))))

            self.current_arm_raise = float(angle_3pts(hip, sh, elbow)) if elbow else 0.0
            self.current_arm_elbow = float(angle_3pts(sh, elbow, wrist)) if (elbow and wrist) else 0.0

            # Shoulder symmetry: |left_y - right_y| / torso  (0 = perfectly level)
            if lsh and rsh:
                self.current_shoulder_sym = float(abs(lsh[1] - rsh[1]) / TL)
            else:
                self.current_shoulder_sym = 0.0

            self.is_user_standing = bool((base_sh_y - sh[1]) > TL * 0.4)

            # -- Timing --
            now     = time.monotonic()
            delta_t = now - self.last_tick
            self.last_tick = now

            if self.is_user_standing:
                self.current_state = "AWAY"
            else:
                th = self.thresholds

                # Per-metric bad flags
                head_bad  = abs(self.current_fhp_ratio)     > th.fhp["warn"]
                spine_bad = self.current_slump_angle         > th.slump["warn"]
                arm_bad   = self.current_arm_raise           > th.arm_raise["warn"]
                sym_bad   = self.current_shoulder_sym        > th.shoulder_sym["warn"]

                # Per-metric states (independent, faster feedback)
                self.current_metric_states = {
                    "head":     self.sm_head.update(head_bad),
                    "spine":    self.sm_spine.update(spine_bad),
                    "arm":      self.sm_arm.update(arm_bad),
                    "shoulder": self.sm_shoulder.update(sym_bad),
                }

                # Weighted risk score
                self.current_risk_score, self.current_metric_scores = self.risk_calc.compute(
                    self.current_fhp_ratio,
                    self.current_slump_angle,
                    self.current_arm_raise,
                    self.current_shoulder_sym,
                    th,
                )

                # Overall state from risk score + time buffer
                self.current_state = self.state_machine.update(self.current_risk_score)

                now_db = time.monotonic()
                if now_db - self.last_db_write >= 1.0:
                    self.stats.update(self.current_state, 1)
                    self.last_db_write = now_db

            # HUD
            draw_hud(
                frame,
                self.current_state,
                self.current_risk_score,
                self.current_metric_scores,
                self.current_metric_states,
                raw_metrics={
                    "fhp_ratio":    self.current_fhp_ratio,
                    "slump_angle":  self.current_slump_angle,
                    "arm_raise":    self.current_arm_raise,
                    "shoulder_sym": self.current_shoulder_sym,
                },
                is_calibrated=self.thresholds.is_calibrated,
            )

        else:
            # Baseline not yet set (before first calibration)
            self.baseline_torso      = np.hypot(sh[0]-hip[0], sh[1]-hip[1])
            self.baseline_shoulder_y = sh[1]
            draw_hud(frame, "GOOD", 0, {}, {}, {}, is_calibrated=False)

        return frame


if __name__ == "__main__":
    print("Run via: python main.py")
