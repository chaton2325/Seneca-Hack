# app.py — Flask + Socket.IO + MediaPipe (quiet, robust, well-documented)
# Goals:
# - Preserve endpoints/events/behaviors (safe drop-in)
# - Reduce noisy logs in production
# - Clear structure & English comments
# - Light input validation and error guards
# - Keep current threading-based Socket.IO (no eventlet/gevent required)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # Hide INFO/WARN logs from TF/TFLite
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"      # Force CPU (common on Windows + webcams)

import base64
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Any

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit

# ======================================================================
# 0) LOGGING (keep stdout clean in production)
# ======================================================================

logging.basicConfig(level=logging.ERROR)
for name in ("werkzeug", "engineio", "socketio"):
    logging.getLogger(name).setLevel(logging.ERROR)

log = logging.getLogger("fitmaster")
log.setLevel(logging.INFO)  # app-level logs (set to ERROR to be fully silent)


# ======================================================================
# 1) FLASK & SOCKET.IO SETUP
# ======================================================================

app = Flask(__name__)
app.config["SECRET_KEY"] = "sport_assistant_secret_key"

# Using threading for compatibility (no eventlet/gevent dependency)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    async_mode="threading",
)

# MediaPipe singletons
mp_pose = mp.solutions.pose


# ======================================================================
# 2) UTILS
# ======================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return max(lo, min(hi, x))


def norm01(x: float, a: float, b: float) -> float:
    """Normalize x linearly into [0, 1] given range [a, b]."""
    return 0.0 if b == a else (x - a) / (b - a)


# ======================================================================
# 3) DATA OBJECTS
# ======================================================================

@dataclass
class ExerciseCounter:
    """State machine + rolling metrics for a given exercise."""
    name: str
    count: int = 0
    stage: str = "up"  # "up" or "down" depending on exercise phase
    angle_threshold_down: float = 90
    angle_threshold_up: float = 160
    confidence_threshold: float = 0.8

    # Debounce/consistency for stage changes
    stable_frames_required: int = 3
    stable_frames_count: int = 0

    # Rep timing & quality metrics
    last_rep_time: float = 0.0
    rep_durations: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    current_rep_min_angle: float = 180.0
    current_rep_max_angle: float = 0.0
    last_rom_score: float = 0.0
    symmetry_diffs: Deque[float] = field(default_factory=lambda: deque(maxlen=8))


class AngleFilter:
    """
    Small robustness helper:
    - Aggregates last angles with visibility-weighted acceptance
    - Emits a median (or mid-mean) to reduce jitter/outliers
    - Provides a 'stable' heuristic based on the last 3 angles
    """
    def __init__(self, window_size: int = 7) -> None:
        self.window_size = window_size
        self.angles: Deque[float] = deque(maxlen=window_size)
        self.confs: Deque[float] = deque(maxlen=window_size)

    def add(self, angle: float, conf: float) -> None:
        self.angles.append(angle)
        self.confs.append(conf)

    def get(self) -> Optional[float]:
        if len(self.angles) < 2:
            return None
        valid = [a for a, c in zip(self.angles, self.confs) if c > 0.5]
        if len(valid) < 2:
            return None
        valid.sort()
        mid = len(valid) // 2
        return valid[mid] if len(valid) % 2 else 0.5 * (valid[mid - 1] + valid[mid])

    def stable(self, threshold: float = 10.0) -> bool:
        if len(self.angles) < self.window_size:
            return False
        last3 = list(self.angles)[-3:]
        return (len(last3) == 3) and ((max(last3) - min(last3)) < threshold)


# ======================================================================
# 4) CORE ASSISTANT (exercise logic)
# ======================================================================

class SportAssistant:
    """Owns the MediaPipe Pose model and exercise analyzers."""

    def __init__(self) -> None:
        # Use lightweight model_complexity=0 for fluidity; set to 1 for more precision
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=0,
            smooth_landmarks=True,
        )

        self.exercises: Dict[str, ExerciseCounter] = {
            "squats":      ExerciseCounter("Squats",      angle_threshold_down=90, angle_threshold_up=160),
            "bicep_curls": ExerciseCounter("Bicep Curls", angle_threshold_down=30, angle_threshold_up=160),
            "pushups":     ExerciseCounter("Push-ups",    angle_threshold_down=80, angle_threshold_up=160),
        }
        self.current_exercise: str = "squats"

        # Per-exercise angle smoothing
        self.filters: Dict[str, AngleFilter] = {k: AngleFilter(7) for k in self.exercises}

        # Last spoken/printed feedback (used by frontend for TTS throttling)
        self.last_feedback: str = ""
        self.last_fb_ts: float = 0.0

    # ---------------- Geometry helpers ----------------

    def calculate_angle(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """Compute internal angle ABC in degrees (0..180)."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        ang = abs(rad * 180.0 / np.pi)
        return float(360 - ang) if ang > 180.0 else float(ang)

    def get_conf(self, lm: List[Any], idxs: List[int]) -> float:
        """Average visibility over a list of landmarks indices."""
        return float(sum(lm[i].visibility for i in idxs) / max(1, len(idxs)))

    def vector_angle_deg(self, v: Tuple[float, float], ref: Tuple[float, float] = (0, -1)) -> float:
        """Angle (deg) between vector v and a reference (default up)."""
        vx, vy = v
        rx, ry = ref
        dot = vx * rx + vy * ry
        nv = math.hypot(vx, vy) * math.hypot(rx, ry)
        if nv == 0:
            return 0.0
        return math.degrees(math.acos(clamp(dot / nv, -1.0, 1.0)))

    def torso_forward_angle(self, lm: List[Any], side: str = "LEFT") -> float:
        """Torso lean estimate via hip→shoulder vector angle (0° ≈ vertical)."""
        hip = lm[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value]
        sh  = lm[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value]
        return self.vector_angle_deg((sh.x - hip.x, sh.y - hip.y))  # 0° ~ vertical

    def knee_over_toe_offset(self, lm: List[Any], side: str = "LEFT") -> float:
        """Relative knee-over-toe offset normalized by tibia length."""
        knee  = lm[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value]
        toe   = lm[getattr(mp_pose.PoseLandmark, f"{side}_FOOT_INDEX").value]
        ankle = lm[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value]
        tib   = math.hypot(ankle.x - knee.x, ankle.y - knee.y) + 1e-6
        return (knee.x - toe.x) / tib

    # ---------------- Analyzers (per exercise) ----------------

    def analyze_squat(self, lm: List[Any]) -> Dict[str, Any]:
        L = mp_pose.PoseLandmark
        hipL, kneeL, ankleL = L.LEFT_HIP.value, L.LEFT_KNEE.value, L.LEFT_ANKLE.value
        hipR, kneeR, ankleR = L.RIGHT_HIP.value, L.RIGHT_KNEE.value, L.RIGHT_ANKLE.value

        conf = self.get_conf(lm, [hipL, kneeL, ankleL, hipR, kneeR, ankleR])
        if conf < 0.6:
            return self._pack("Position non détectée - placez-vous face caméra", safety="danger")

        aL = self.calculate_angle([lm[hipL].x, lm[hipL].y], [lm[kneeL].x, lm[kneeL].y], [lm[ankleL].x, lm[ankleL].y])
        aR = self.calculate_angle([lm[hipR].x, lm[hipR].y], [lm[kneeR].x, lm[kneeR].y], [lm[ankleR].x, lm[ankleR].y])
        ang = 0.5 * (aL + aR)

        filt = self.filters["squats"]
        ex = self.exercises["squats"]
        filt.add(ang, conf)
        angle = filt.get()
        if angle is None:
            return self._pack("Calibration en cours...", safety="warn")

        # Track per-rep angle span to estimate ROM
        ex.current_rep_min_angle = min(ex.current_rep_min_angle, angle)
        ex.current_rep_max_angle = max(ex.current_rep_max_angle, angle)

        feedback: Optional[str] = None

        # Stage transitions (with stability guard)
        if angle > ex.angle_threshold_up and filt.stable():
            if ex.stage != "up":
                if ex.last_rep_time > 0:
                    dur = time.time() - ex.last_rep_time
                    if 0.2 < dur < 15:
                        ex.rep_durations.append(dur)
                ex.last_rep_time = time.time()
                ex.stage = "up"
                ex.stable_frames_count = 0
            else:
                ex.stable_frames_count += 1

        elif angle < ex.angle_threshold_down and filt.stable():
            if ex.stage == "up" and ex.stable_frames_count >= ex.stable_frames_required:
                ex.stage = "down"
                ex.count += 1
                ex.stable_frames_count = 0
                feedback = f"Squat #{ex.count} - Excellent! 🎉"

                # ROM scoring (lower min angle → deeper squat)
                min_a = ex.current_rep_min_angle if ex.current_rep_min_angle < 170 else 170
                ex.last_rom_score = clamp(100 * norm01(170 - min_a, 0, 110), 0, 100)

                # Symmetry diff (left vs right knee angles)
                ex.symmetry_diffs.append(abs(aL - aR))

                # Reset for next rep
                ex.current_rep_min_angle, ex.current_rep_max_angle = 180.0, 0.0

        # Live guidance if not at a transition moment
        if not feedback:
            if angle < 70:
                feedback = "Parfait! Maintiens bas 💪"
            elif angle > 170:
                feedback = "Debout - prêt à descendre 🏋️"
            elif 70 <= angle <= 90:
                feedback = "Descends un peu plus 📉"
            elif 160 <= angle <= 170:
                feedback = "Remonte complètement 📈"
            else:
                feedback = f"En mouvement... {int(angle)}° 🔄"

        # Safety heuristics (torso lean & knee-over-toe)
        torso = 0.5 * (self.torso_forward_angle(lm, "LEFT") + self.torso_forward_angle(lm, "RIGHT"))
        knee_toe = max(self.knee_over_toe_offset(lm, "LEFT"), self.knee_over_toe_offset(lm, "RIGHT"))
        safety = "ok"
        if torso > 30 or knee_toe > 0.25:
            safety = "danger"
        elif torso > 20 or knee_toe > 0.15:
            safety = "warn"

        # Aggregate metrics
        tempo = (sum(ex.rep_durations) / len(ex.rep_durations)) if ex.rep_durations else None
        rom   = ex.last_rom_score if ex.last_rom_score else clamp(100 * norm01(170 - angle, 0, 110), 0, 100)
        sym   = (sum(ex.symmetry_diffs) / len(ex.symmetry_diffs)) if ex.symmetry_diffs else None

        # Composite score: ROM + tempo; safety penalties
        score = rom
        if tempo is not None:
            tempo_score = 100 - 100 * abs((tempo - 3.0) / 3.0)
            score = 0.6 * rom + 0.4 * clamp(tempo_score, 0, 100)
        if safety == "warn":
            score *= 0.85
        if safety == "danger":
            score *= 0.6

        return self._pack(feedback, tempo, rom, sym, safety, int(clamp(score, 0, 100)))

    def analyze_bicep_curl(self, lm: List[Any]) -> Dict[str, Any]:
        L = mp_pose.PoseLandmark
        shL, elL, wrL = L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value
        shR, elR, wrR = L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_WRIST.value

        conf = self.get_conf(lm, [shL, elL, wrL])
        if conf < 0.6:
            return self._pack("Montre ton bras gauche", safety="warn")

        aL = self.calculate_angle([lm[shL].x, lm[shL].y], [lm[elL].x, lm[elL].y], [lm[wrL].x, lm[wrL].y])
        filt = self.filters["bicep_curls"]
        ex = self.exercises["bicep_curls"]
        filt.add(aL, conf)
        angle = filt.get()
        if angle is None:
            return self._pack("Calibration en cours...", safety="warn")

        ex.current_rep_min_angle = min(ex.current_rep_min_angle, angle)
        ex.current_rep_max_angle = max(ex.current_rep_max_angle, angle)
        feedback: Optional[str] = None

        if angle > ex.angle_threshold_up and filt.stable():
            if ex.stage != "down":
                if ex.last_rep_time > 0:
                    dur = time.time() - ex.last_rep_time
                    if 0.2 < dur < 15:
                        ex.rep_durations.append(dur)
                ex.last_rep_time = time.time()
                ex.stage = "down"
                ex.stable_frames_count = 0
            else:
                ex.stable_frames_count += 1

        elif angle < ex.angle_threshold_down and filt.stable():
            if ex.stage == "down" and ex.stable_frames_count >= ex.stable_frames_required:
                ex.stage = "up"
                ex.count += 1
                ex.stable_frames_count = 0
                feedback = f"Bicep Curl #{ex.count} - Parfait! 💪"

                min_a = ex.current_rep_min_angle if ex.current_rep_min_angle < 160 else 160
                ex.last_rom_score = clamp(100 * norm01(160 - min_a, 0, 130), 0, 100)

                # Compare with right arm when visible
                confR = self.get_conf(lm, [shR, elR, wrR])
                if confR > 0.5:
                    aR = self.calculate_angle([lm[shR].x, lm[shR].y], [lm[elR].x, lm[elR].y], [lm[wrR].x, lm[wrR].y])
                    ex.symmetry_diffs.append(abs(aL - aR))

                ex.current_rep_min_angle, ex.current_rep_max_angle = 180.0, 0.0

        if not feedback:
            if angle < 50:
                feedback = "Flexion max - tiens 🔥"
            elif angle > 140:
                feedback = "Extension complète 💪"
            elif 50 <= angle <= 80:
                feedback = "Remonte lentement 📈"
            elif 120 <= angle <= 140:
                feedback = "Descends en contrôle 📉"
            else:
                feedback = f"En mouvement... {int(angle)}° 🔄"

        tempo = (sum(ex.rep_durations) / len(ex.rep_durations)) if ex.rep_durations else None
        rom   = ex.last_rom_score if ex.last_rom_score else clamp(100 * norm01(160 - angle, 0, 130), 0, 100)
        sym   = (sum(ex.symmetry_diffs) / len(ex.symmetry_diffs)) if ex.symmetry_diffs else None
        score = rom
        if tempo is not None:
            tempo_score = 100 - 100 * abs((tempo - 2.5) / 2.5)
            score = 0.6 * rom + 0.4 * clamp(tempo_score, 0, 100)
        return self._pack(feedback, tempo, rom, sym, "ok", int(clamp(score, 0, 100)))

    def analyze_pushup(self, lm: List[Any]) -> Dict[str, Any]:
        L = mp_pose.PoseLandmark
        shL, elL, wrL = L.LEFT_SHOULDER.value, L.LEFT_ELBOW.value, L.LEFT_WRIST.value
        shR, elR, wrR = L.RIGHT_SHOULDER.value, L.RIGHT_ELBOW.value, L.RIGHT_WRIST.value

        conf = self.get_conf(lm, [shL, elL, wrL, shR, elR, wrR])
        if conf < 0.6:
            return self._pack("Place-toi en planche", safety="danger")

        aL = self.calculate_angle([lm[shL].x, lm[shL].y], [lm[elL].x, lm[elL].y], [lm[wrL].x, lm[wrL].y])
        aR = self.calculate_angle([lm[shR].x, lm[shR].y], [lm[elR].x, lm[elR].y], [lm[wrR].x, lm[wrR].y])
        ang = 0.5 * (aL + aR)

        filt = self.filters["pushups"]
        ex = self.exercises["pushups"]
        filt.add(ang, conf)
        angle = filt.get()
        if angle is None:
            return self._pack("Calibration en cours...", safety="warn")

        ex.current_rep_min_angle = min(ex.current_rep_min_angle, angle)
        ex.current_rep_max_angle = max(ex.current_rep_max_angle, angle)
        feedback: Optional[str] = None

        if angle > ex.angle_threshold_up and filt.stable():
            if ex.stage != "up":
                if ex.last_rep_time > 0:
                    dur = time.time() - ex.last_rep_time
                    if 0.2 < dur < 15:
                        ex.rep_durations.append(dur)
                ex.last_rep_time = time.time()
                ex.stage = "up"
                ex.stable_frames_count = 0
            else:
                ex.stable_frames_count += 1

        elif angle < ex.angle_threshold_down and filt.stable():
            if ex.stage == "up" and ex.stable_frames_count >= ex.stable_frames_required:
                ex.stage = "down"
                ex.count += 1
                ex.stable_frames_count = 0
                feedback = f"Pompe #{ex.count} - Excellent! 🔥"

                min_a = ex.current_rep_min_angle if ex.current_rep_min_angle < 150 else 150
                ex.last_rom_score = clamp(100 * norm01(150 - min_a, 0, 120), 0, 100)
                ex.symmetry_diffs.append(abs(aL - aR))

                ex.current_rep_min_angle, ex.current_rep_max_angle = 180.0, 0.0

        if not feedback:
            if angle < 90:
                feedback = "Bas parfait - remonte 💪"
            elif angle > 150:
                feedback = "Haut - prêt à descendre 🤸"
            elif 90 <= angle <= 110:
                feedback = "Pousse fort 📈"
            elif 130 <= angle <= 150:
                feedback = "Descends en contrôle 📉"
            else:
                feedback = f"En mouvement... {int(angle)}° 🔄"

        # Safety via torso lean (proxy for sagging hips/arched back)
        torso = 0.5 * (self.torso_forward_angle(lm, "LEFT") + self.torso_forward_angle(lm, "RIGHT"))
        safety = "ok"
        if torso > 25:
            safety = "warn"
        if torso > 35:
            safety = "danger"

        tempo = (sum(ex.rep_durations) / len(ex.rep_durations)) if ex.rep_durations else None
        rom   = ex.last_rom_score if ex.last_rom_score else clamp(100 * norm01(150 - angle, 0, 120), 0, 100)
        sym   = (sum(ex.symmetry_diffs) / len(ex.symmetry_diffs)) if ex.symmetry_diffs else None

        score = rom
        if tempo is not None:
            tempo_score = 100 - 100 * abs((tempo - 2.5) / 2.5)
            score = 0.6 * rom + 0.4 * clamp(tempo_score, 0, 100)
        if safety == "warn":
            score *= 0.85
        if safety == "danger":
            score *= 0.6

        return self._pack(feedback, tempo, rom, sym, safety, int(clamp(score, 0, 100)))

    # ---------------- Dispatch & pack ----------------

    def analyze(self, lm: List[Any]) -> Dict[str, Any]:
        ce = self.current_exercise
        if ce == "squats":
            return self.analyze_squat(lm)
        elif ce == "bicep_curls":
            return self.analyze_bicep_curl(lm)
        elif ce == "pushups":
            return self.analyze_pushup(lm)
        return self._pack("Exercice non reconnu", safety="warn")

    def _pack(
        self,
        feedback: str,
        tempo: Optional[float] = None,
        rom: Optional[float] = None,
        symmetry: Optional[float] = None,
        safety: str = "ok",
        score: int = 0,
    ) -> Dict[str, Any]:
        """Uniform payload for the frontend."""
        return {
            "feedback": feedback,
            "tempo": tempo,
            "rom": rom,
            "symmetry": symmetry,
            "safety": safety,
            "score": score,
        }


sport_assistant = SportAssistant()


# ======================================================================
# 5) IMAGE PROCESSOR (decode + resize + pose)
# ======================================================================

class ImageProcessor:
    """
    Decodes incoming base64 frames, downsizes to an inference resolution,
    runs MediaPipe Pose, and returns a compact payload for the client.
    """
    def __init__(self) -> None:
        # Small inference size keeps latency low; client sends higher-res
        self.infer_w, self.infer_h = 256, 144

    def process_frame(self, data_url: str) -> Optional[Dict[str, Any]]:
        """Decode base64 image → BGR → resize → analyze pose → payload."""
        try:
            # Strip data URL header if present
            if data_url.startswith("data:image"):
                data_url = data_url.split(",", 1)[1]

            img_buf = base64.b64decode(data_url)
            img = cv2.imdecode(np.frombuffer(img_buf, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None

            img_small = cv2.resize(img, (self.infer_w, self.infer_h), interpolation=cv2.INTER_AREA)
            return self.analyze_pose(img_small)
        except Exception as e:
            log.error("Frame processing error: %s", e)
            return None

    def analyze_pose(self, bgr_small: np.ndarray) -> Dict[str, Any]:
        """Run MediaPipe pose and build the response dict."""
        rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # small perf gain

        results = sport_assistant.pose.process(rgb)

        # Default payload (warn until a pose is found)
        payload: Dict[str, Any] = {
            "landmarks": None,
            "feedback": "Positionnez-vous devant la caméra",
            "exercise": sport_assistant.current_exercise,
            "count": sport_assistant.exercises[sport_assistant.current_exercise].count,
            "stage": sport_assistant.exercises[sport_assistant.current_exercise].stage,
            "tempo": None,
            "rom": None,
            "symmetry": None,
            "safety": "warn",
            "score": 0,
        }

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # Send only the essentials to keep payload light
            payload["landmarks"] = [
                {"x": p.x, "y": p.y, "z": p.z, "visibility": p.visibility} for p in lm
            ]
            payload.update(sport_assistant.analyze(lm))

        return payload


image_processor = ImageProcessor()


# ======================================================================
# 6) ROUTES & SOCKET.IO HANDLERS
# ======================================================================

@app.route("/")
def index():
    """Render the main web UI (templates/index.html)."""
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """Serve favicon to avoid 404 noise in logs."""
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


@socketio.on("connect")
def handle_connect():
    """Client connected (handshake ok)."""
    emit("connected", {"data": "Connexion établie"})


@socketio.on("disconnect")
def handle_disconnect():
    """Client disconnected (cleanup if needed)."""
    pass


@socketio.on("process_frame")
def handle_frame(data: Dict[str, Any]):
    """
    Receive base64 frame from client, run pose + analysis, and emit result.
    The client expects 'pose_result' messages.
    """
    try:
        img = (data or {}).get("image")
        if not img:
            return
        result = image_processor.process_frame(img)
        if result:
            emit("pose_result", result)
    except Exception as e:
        log.error("Socket frame error: %s", e)


@app.route("/change_exercise", methods=["POST"])
def change_exercise():
    """
    Change current exercise. Body: {"exercise": "squats" | "bicep_curls" | "pushups"}.
    Frontend depends on the 'success' flag.
    """
    ex = (request.get_json(silent=True) or {}).get("exercise", "squats")
    if ex in sport_assistant.exercises:
        sport_assistant.current_exercise = ex
        return jsonify({"success": True, "exercise": ex})
    return jsonify({"success": False, "error": "Exercice non trouvé"}), 400


@app.route("/reset_counter", methods=["POST"])
def reset_counter():
    """Reset counters/rolling metrics for the current exercise."""
    ex = sport_assistant.exercises[sport_assistant.current_exercise]
    ex.count = 0
    ex.stage = "up"
    ex.rep_durations.clear()
    ex.symmetry_diffs.clear()
    ex.current_rep_min_angle, ex.current_rep_max_angle = 180.0, 0.0
    ex.last_rom_score = 0.0
    ex.last_rep_time = 0.0
    return jsonify({"success": True, "count": 0})


@app.route("/get_stats")
def get_stats():
    """Return a snapshot of current exercise states (for debugging/UI)."""
    stats = {
        name: {"count": ex.count, "stage": ex.stage, "name": ex.name}
        for name, ex in sport_assistant.exercises.items()
    }
    return jsonify({"current_exercise": sport_assistant.current_exercise, "exercises": stats})


# ======================================================================
# 7) MAIN
# ======================================================================

if __name__ == "__main__":
    # For local dev: Werkzeug + Socket.IO (threading). Avoid debug logs by default.
    socketio.run(app, debug=False, host="0.0.0.0", port=8093, allow_unsafe_werkzeug=True)
