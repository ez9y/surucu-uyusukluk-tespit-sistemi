# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import threading
import queue
import sys
import os
from collections import deque
from datetime import datetime

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

import customtkinter as ctk
from PIL import Image

try:
    import winsound
    def play_beep(high=False):
        freq = 1200 if high else 900
        threading.Thread(target=lambda: winsound.Beep(freq, 400), daemon=True).start()
except ImportError:
    def play_beep(high=False):
        print("\a", end="", flush=True)

def _res(path):
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, path)

# ── AYARLAR ──────────────────────────────────────────
IMG_SHAPE        = 128
EYE_THRESHOLD    = 0.5
YAWN_THRESHOLD   = 0.85
PERCLOS_SECONDS  = 30
PERCLOS_ALARM    = 0.50
YAWN_COUNT_ALARM = 4
YAWN_RESET_SEC   = 60
RISK_LOW         = 23
RISK_MEDIUM      = 47
RISK_HIGH        = 70
BEEP_INTERVAL    = 3.0

EYE_MODEL_PATH   = _res("eye_model_compat.h5")
YAWN_MODEL_PATH  = _res("yawn_model_compat.h5")
FACE_TASK_PATH   = _res("face_landmarker.task")
AGIZ_UST         = 0.62

LEFT_EYE  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

# ── YARDIMCI ─────────────────────────────────────────
def list_cameras(max_test=6):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def get_camera_names():
    """DirectShow sırasıyla kamera isimlerini döndürür (OpenCV index ile eşleşir)."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        g = FilterGraph()
        return g.get_input_devices()
    except Exception:
        return []

def preprocess_eye(roi):
    img = cv2.resize(roi, (IMG_SHAPE, IMG_SHAPE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

def preprocess_yawn(roi):
    img = cv2.resize(roi, (IMG_SHAPE, IMG_SHAPE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(preprocess_input(img.astype(np.float32)), axis=0)

def perclos_score(p):
    if p >= 0.80: return 3
    if p >= 0.50: return 2
    if p >= 0.20: return 1
    return 0

def compute_risk(perclos, yawn_count):
    ps = (perclos_score(perclos) / 3) * 70
    ys = min(yawn_count / YAWN_COUNT_ALARM, 1.0) * 30
    return int(ps + ys)

def risk_color_hex(score):
    if score >= RISK_HIGH:   return "#FF3333"
    if score >= RISK_MEDIUM: return "#FF8800"
    if score >= RISK_LOW:    return "#FFCC00"
    return "#33CC66"

def risk_label(score):
    if score >= RISK_HIGH:   return "HIGH RISK"
    if score >= RISK_MEDIUM: return "MEDIUM RISK"
    if score >= RISK_LOW:    return "LOW RISK"
    return "NORMAL"


# ── PERCLOS TRACKER ───────────────────────────────────
class PerclosTracker:
    def __init__(self, window_sec):
        self.window_sec = window_sec
        self.history = deque()

    def update(self, eye_closed):
        now = time.time()
        self.history.append((now, 1 if eye_closed else 0))
        while self.history and (now - self.history[0][0]) > self.window_sec:
            self.history.popleft()

    def get(self):
        if not self.history: return 0.0
        return sum(v for _, v in self.history) / len(self.history)

    def elapsed(self):
        if not self.history: return 0
        return min(time.time() - self.history[0][0], self.window_sec)

    def clear(self):
        self.history.clear()


# ── DETECTION ENGINE ──────────────────────────────────
class DetectionEngine:
    def __init__(self, camera_id, eye_model, yawn_model,
                 frame_queue, stats_queue, log_queue):
        self.camera_id   = camera_id
        self.eye_model   = eye_model
        self.yawn_model  = yawn_model
        self.frame_queue = frame_queue
        self.stats_queue = stats_queue
        self.log_queue   = log_queue
        self.running     = False
        self._reset_flag = threading.Event()

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def request_reset(self):
        self._reset_flag.set()

    def _put(self, q, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(item)
            except Exception:
                pass

    def _log(self, msg):
        try: self.log_queue.put_nowait(msg)
        except queue.Full: pass

    def _run(self):
        face_mesh = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_base.BaseOptions(model_asset_path=FACE_TASK_PATH),
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._log(f"[ERROR] Camera {self.camera_id} acilamadi!")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        perclos_tracker  = PerclosTracker(PERCLOS_SECONDS)
        yawn_count       = 0
        last_yawn_state  = False
        yawn_consec      = 0
        last_beep_time   = 0
        yawn_timer_start = time.time()
        session_start    = time.time()
        last_log_time    = time.time()
        fps_time         = time.time()
        fps_counter      = 0
        current_fps      = 0.0

        while self.running:
            if self._reset_flag.is_set():
                perclos_tracker.clear()
                yawn_count       = 0
                last_yawn_state  = False
                yawn_timer_start = time.time()
                session_start    = time.time()
                self._reset_flag.clear()
                ts = datetime.now().strftime("%H:%M:%S")
                self._log(f"[{ts}] Statistics reset")

            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            fps_counter += 1
            now_fps = time.time()
            if now_fps - fps_time >= 1.0:
                current_fps = fps_counter / (now_fps - fps_time)
                fps_counter = 0
                fps_time    = now_fps

            if time.time() - yawn_timer_start > YAWN_RESET_SEC:
                yawn_count       = 0
                yawn_timer_start = time.time()

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = face_mesh.detect(mp_img)

            eye_closed    = False
            yawn_detect   = False
            face_detected = len(results.face_landmarks) > 0

            if face_detected:
                lms = results.face_landmarks[0]
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                xs  = [p[0] for p in pts]; ys = [p[1] for p in pts]
                fx  = max(0, min(xs)); fy = max(0, min(ys))
                fw  = min(w, max(xs)) - fx
                fh  = min(h, max(ys)) - fy
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (100, 220, 100), 2)

                # Göz tespiti (landmark tabanlı) — her iki göz tek batch'te
                eye_boxes  = []
                eye_crops  = []
                for eye_idx in [LEFT_EYE, RIGHT_EYE]:
                    ep  = [pts[i] for i in eye_idx]
                    ex1 = max(0, min(p[0] for p in ep) - 8)
                    ey1 = max(0, min(p[1] for p in ep) - 6)
                    ex2 = min(w, max(p[0] for p in ep) + 8)
                    ey2 = min(h, max(p[1] for p in ep) + 6)
                    crop = frame[ey1:ey2, ex1:ex2]
                    if crop.size == 0: continue
                    eye_boxes.append((ex1, ey1, ex2, ey2))
                    eye_crops.append(preprocess_eye(crop)[0])

                eye_preds = []
                if eye_crops:
                    batch = np.stack(eye_crops, axis=0)
                    preds = self.eye_model(batch, training=False).numpy()[:, 0]
                    for (ex1, ey1, ex2, ey2), pred in zip(eye_boxes, preds):
                        eye_preds.append(float(pred))
                        renk = (0, 255, 0) if pred > EYE_THRESHOLD else (0, 0, 255)
                        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), renk, 2)

                if eye_preds:
                    eye_closed = (sum(eye_preds) / len(eye_preds)) < EYE_THRESHOLD

                # Ağız tespiti (geometrik bant)
                ay   = int(fh * AGIZ_UST)
                agiz = frame[fy+ay:fy+fh, fx:fx+fw]
                if agiz.shape[0] > 10 and agiz.shape[1] > 10:
                    yp = float(self.yawn_model(preprocess_yawn(agiz), training=False).numpy()[0][0])
                    yawn_detect = yp > YAWN_THRESHOLD
                    cv2.rectangle(frame, (fx, fy+ay), (fx+fw, fy+fh), (255, 150, 50), 1)
                    if yawn_detect:
                        cv2.putText(frame, "YAWN",
                                    (fx+2, fy+fh+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 255), 2)

                eye_txt = "CLOSED" if eye_closed else "OPEN"
                eye_col = (0, 80, 255) if eye_closed else (0, 255, 100)
                cv2.putText(frame, eye_txt, (fx+2, fy-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, eye_col, 1)

                if yawn_detect:
                    yawn_consec += 1
                else:
                    yawn_consec = 0

                sustained = yawn_consec >= 8
                if sustained and not last_yawn_state:
                    yawn_count += 1
                    ts = datetime.now().strftime("%H:%M:%S")
                    self._log(f"[{ts}] Yawn detected ({yawn_count}. kez)")
                last_yawn_state = sustained

            perclos_tracker.update(eye_closed)
            perclos = perclos_tracker.get()
            elapsed = perclos_tracker.elapsed()
            risk    = compute_risk(perclos, yawn_count)

            perclos_alarm = perclos >= PERCLOS_ALARM and elapsed >= 5
            yawn_alarm    = yawn_count >= YAWN_COUNT_ALARM
            alarm_active  = perclos_alarm or yawn_alarm

            now = time.time()
            if now - last_log_time >= 30.0:
                ts = datetime.now().strftime("%H:%M:%S")
                m, s = divmod(int(elapsed), 60)
                self._log(
                    f"[{ts}] STATUS – PERCLOS: %{int(perclos*100)} | "
                    f"Risk: {risk} | Yawn: {yawn_count} | Time: {m:02d}:{s:02d}")
                last_log_time = now

            if alarm_active and (now - last_beep_time) > BEEP_INTERVAL:
                play_beep(high=perclos_alarm)
                last_beep_time = now
                reason = []
                if perclos_alarm: reason.append(f"PERCLOS %{int(perclos*100)}")
                if yawn_alarm:    reason.append(f"Yawn {yawn_count}x")
                ts = datetime.now().strftime("%H:%M:%S")
                self._log(f"[{ts}] !! ALARM: {' | '.join(reason)}")

            if alarm_active:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

            stats = {
                "risk":          risk,
                "risk_label":    risk_label(risk),
                "risk_color":    risk_color_hex(risk),
                "perclos":       int(perclos * 100),
                "perclos_alarm": perclos_alarm,
                "yawn_count":    yawn_count,
                "yawn_alarm":    yawn_alarm,
                "eye_closed":    eye_closed,
                "face_active":   face_detected,
                "face_found":    face_detected,
                "elapsed":       elapsed,
                "alarm_active":  alarm_active,
                "session_sec":   int(time.time() - session_start),
                "fps":           current_fps,
            }

            self._put(self.frame_queue, frame)
            self._put(self.stats_queue, stats)

        cap.release()


# ── ANA UYGULAMA ──────────────────────────────────────
class DrowsinessApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Driver Drowsiness Detection")
        self.geometry("1150x700")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        icon_path = _res("icon.ico")
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)

        self.eye_model   = None
        self.yawn_model  = None
        self.engine      = None
        self.running     = False
        self._camera_ids = []

        self.frame_queue = queue.Queue(maxsize=2)
        self.stats_queue = queue.Queue(maxsize=2)
        self.log_queue   = queue.Queue()
        self.log_entries = []

        self._build_ui()
        self._load_models_async()
        self._detect_cameras_async()
        self._poll()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sol panel
        left = ctk.CTkFrame(self, corner_radius=12)
        left.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="nsew")
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(left, text="Driver Drowsiness Detection",
                     font=("Helvetica", 16, "bold")).grid(row=0, column=0, pady=(10, 0))

        self.video_label = ctk.CTkLabel(
            left, text="Waiting for camera to start...",
            font=("Helvetica", 13), text_color="gray",
            width=720, height=500)
        self.video_label.grid(row=1, column=0, padx=8, pady=6, sticky="nsew")

        ctrl = ctk.CTkFrame(left, fg_color="transparent")
        ctrl.grid(row=2, column=0, padx=8, pady=(0, 4), sticky="ew")

        ctk.CTkLabel(ctrl, text="Camera:", font=("Helvetica", 12)).pack(side="left", padx=(4, 2))
        self.camera_combo = ctk.CTkComboBox(ctrl, width=130, values=["Scanning..."],
                                             state="disabled")
        self.camera_combo.pack(side="left", padx=(0, 10))

        self.start_btn = ctk.CTkButton(
            ctrl, text="  Start", width=110, height=36,
            font=("Helvetica", 13, "bold"),
            fg_color="#1976D2", hover_color="#0D47A1",
            command=self._toggle, state="disabled")
        self.start_btn.pack(side="left", padx=4)

        self.reset_btn = ctk.CTkButton(
            ctrl, text="  Reset", width=100, height=36,
            font=("Helvetica", 13),
            fg_color="#424242", hover_color="#212121",
            command=self._reset)
        self.reset_btn.pack(side="left", padx=4)

        self.export_btn = ctk.CTkButton(
            ctrl, text="  Save Log", width=120, height=36,
            font=("Helvetica", 13),
            fg_color="#37474F", hover_color="#263238",
            command=self._export_log)
        self.export_btn.pack(side="right", padx=4)

        self.status_label = ctk.CTkLabel(
            left, text="Loading models, please wait...",
            font=("Helvetica", 11), text_color="#78909C")
        self.status_label.grid(row=3, column=0, pady=(0, 8))

        # Sağ panel
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(6, weight=1)

        self.risk_frame = ctk.CTkFrame(right, corner_radius=10, fg_color="#1E1E2E", height=100)
        self.risk_frame.grid(row=0, column=0, padx=10, pady=(12, 4), sticky="ew")
        self.risk_frame.grid_propagate(False)
        self.risk_frame.grid_columnconfigure(0, weight=1)

        self.risk_score_lbl = ctk.CTkLabel(
            self.risk_frame, text="RISK: 0 / 100",
            font=("Helvetica", 24, "bold"), text_color="white")
        self.risk_score_lbl.grid(row=0, column=0, pady=(18, 2))

        self.risk_status_lbl = ctk.CTkLabel(
            self.risk_frame, text="NORMAL",
            font=("Helvetica", 15, "bold"), text_color="#33CC66")
        self.risk_status_lbl.grid(row=1, column=0, pady=(0, 12))

        self.risk_bar = ctk.CTkProgressBar(right, height=16, corner_radius=8,
                                            progress_color="#33CC66")
        self.risk_bar.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.risk_bar.set(0)

        stats_frame = ctk.CTkFrame(right, corner_radius=10)
        stats_frame.grid(row=2, column=0, padx=10, pady=(0, 8), sticky="ew")
        stats_frame.grid_columnconfigure(1, weight=1)

        self._stat_lbls = {}
        rows = [
            ("perclos", "PERCLOS",  "%0"),
            ("yawn",    "Yawn",   "0 / 4"),
            ("eye",     "Eye",      "OPEN"),
            ("face",    "Face",      "–"),
            ("elapsed", "Time",     "00:00"),
            ("fps",     "FPS",      "–"),
        ]
        for i, (key, title, default) in enumerate(rows):
            ctk.CTkLabel(stats_frame, text=title,
                         font=("Helvetica", 11), text_color="#78909C").grid(
                row=i, column=0, padx=(12, 4), pady=5, sticky="w")
            lbl = ctk.CTkLabel(stats_frame, text=default,
                               font=("Helvetica", 12, "bold"))
            lbl.grid(row=i, column=1, padx=(4, 12), pady=5, sticky="e")
            self._stat_lbls[key] = lbl

        self.calib_bar = ctk.CTkProgressBar(right, height=10, corner_radius=5,
                                             progress_color="#0288D1")
        self.calib_bar.grid(row=3, column=0, padx=10, pady=(0, 2), sticky="ew")
        self.calib_bar.set(0)
        self.calib_lbl = ctk.CTkLabel(right, text=f"Calibration: 0 / {PERCLOS_SECONDS} s",
                                       font=("Helvetica", 10), text_color="#78909C")
        self.calib_lbl.grid(row=4, column=0, padx=10, pady=(0, 6))

        ctk.CTkLabel(right, text="Event Log",
                     font=("Helvetica", 12, "bold")).grid(
            row=5, column=0, padx=10, pady=(4, 2), sticky="w")

        self.log_box = ctk.CTkTextbox(right, font=("Courier", 10),
                                       state="disabled", wrap="word")
        self.log_box.grid(row=6, column=0, padx=10, pady=(0, 12), sticky="nsew")

    def _load_models_async(self):
        def _load():
            try:
                eye  = load_model(EYE_MODEL_PATH,  compile=False)
                yawn = load_model(YAWN_MODEL_PATH, compile=False)
                self.eye_model  = eye
                self.yawn_model = yawn
                self.after(0, lambda: self.status_label.configure(
                    text="Ready – Press Start"))
                self.after(0, lambda: self.start_btn.configure(state="normal"))
            except Exception as e:
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: self.status_label.configure(
                    text=f"Model load error: {msg}", text_color="#EF5350"))
        threading.Thread(target=_load, daemon=True).start()

    def _detect_cameras_async(self):
        def _scan():
            cams = list_cameras()
            dev_names = get_camera_names()
            if cams:
                names = []
                for i, cam_id in enumerate(cams):
                    label = dev_names[i] if i < len(dev_names) else f"Camera {cam_id}"
                    names.append(f"[{cam_id}] {label}")
                self._camera_ids = cams
                self.after(0, lambda n=names: self.camera_combo.configure(
                    values=n, state="normal"))
                self.after(0, lambda n=names: self.camera_combo.set(n[0]))
            else:
                self.after(0, lambda: self.camera_combo.configure(
                    values=["No camera found"]))
        threading.Thread(target=_scan, daemon=True).start()

    def _toggle(self):
        if not self.running:
            self._start()
        else:
            self._stop()

    def _start(self):
        if not (self.eye_model and self.yawn_model):
            return

        cam_name = self.camera_combo.get()
        try:
            idx    = self.camera_combo.cget("values").index(cam_name)
            cam_id = self._camera_ids[idx]
        except (ValueError, IndexError):
            cam_id = 0

        self.engine = DetectionEngine(
            cam_id, self.eye_model, self.yawn_model,
            self.frame_queue, self.stats_queue, self.log_queue)
        self.engine.start()
        self.running = True

        self.start_btn.configure(text="  Stop",
                                  fg_color="#C62828", hover_color="#B71C1C")
        self.camera_combo.configure(state="disabled")
        self.status_label.configure(text="System running...", text_color="#81C784")

        ts = datetime.now().strftime("%H:%M:%S")
        self._add_log(f"[{ts}] System started – Camera {cam_id}")

    def _stop(self):
        if self.engine:
            self.engine.stop()
            self.engine = None
        self.running = False
        self.start_btn.configure(text="  Start",
                                  fg_color="#1976D2", hover_color="#0D47A1")
        self.camera_combo.configure(state="normal")
        self.status_label.configure(text="Stopped", text_color="#78909C")
        self.video_label.configure(image=None,
                                    text="Camera stopped.\nPress Start to resume.")
        self.risk_bar.set(0)
        ts = datetime.now().strftime("%H:%M:%S")
        self._add_log(f"[{ts}] System stopped")

    def _reset(self):
        if self.engine:
            self.engine.request_reset()
        else:
            ts = datetime.now().strftime("%H:%M:%S")
            self._add_log(f"[{ts}] System not running, nothing to reset")

    def _export_log(self):
        if not self.log_entries:
            self.status_label.configure(text="Log is empty, nothing to save")
            return
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drowsiness_log_{ts}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("Driver Drowsiness Detection - Event Log\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 44 + "\n\n")
            f.write("\n".join(self.log_entries))
        self.status_label.configure(text=f"Log saved: {filename}",
                                     text_color="#81C784")

    def _add_log(self, text):
        self.log_entries.append(text)
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _poll(self):
        while not self.log_queue.empty():
            try:
                self._add_log(self.log_queue.get_nowait())
            except queue.Empty:
                break

        if not self.stats_queue.empty():
            try:
                self._update_stats(self.stats_queue.get_nowait())
            except queue.Empty:
                pass

        if not self.frame_queue.empty():
            try:
                self._update_frame(self.frame_queue.get_nowait())
            except queue.Empty:
                pass

        self.after(30, self._poll)

    def _update_frame(self, frame):
        h, w         = frame.shape[:2]
        max_w, max_h = 716, 496
        scale        = min(max_w / w, max_h / h)
        nw, nh       = int(w * scale), int(h * scale)
        frame        = cv2.resize(frame, (nw, nh))
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img          = ctk.CTkImage(Image.fromarray(rgb), size=(nw, nh))
        self.video_label.configure(image=img, text="")
        self.video_label._ctk_image = img

    def _update_stats(self, s):
        risk  = s["risk"]
        color = s["risk_color"]
        elapsed = s["elapsed"]

        self.risk_score_lbl.configure(text=f"RISK: {risk} / 100")
        self.risk_status_lbl.configure(text=s["risk_label"], text_color=color)
        self.risk_bar.set(risk / 100)
        self.risk_bar.configure(progress_color=color)
        self.risk_frame.configure(fg_color=color if s["alarm_active"] else "#1E1E2E")
        self.risk_score_lbl.configure(text_color="#1A1A1A" if s["alarm_active"] else "white")
        self.risk_status_lbl.configure(text_color="#1A1A1A" if s["alarm_active"] else color)

        calib_ratio = min(elapsed / PERCLOS_SECONDS, 1.0)
        self.calib_bar.set(calib_ratio)
        if elapsed < PERCLOS_SECONDS:
            self.calib_lbl.configure(
                text=f"Calibration: {int(elapsed)} / {PERCLOS_SECONDS} s")
        else:
            self.calib_lbl.configure(text="Calibration complete")

        self._stat_lbls["perclos"].configure(
            text=f"%{s['perclos']}  (threshold: %{int(PERCLOS_ALARM*100)})",
            text_color="#EF5350" if s["perclos_alarm"] else "white")

        self._stat_lbls["yawn"].configure(
            text=f"{s['yawn_count']} / {YAWN_COUNT_ALARM}",
            text_color="#EF5350" if s["yawn_alarm"] else "white")

        self._stat_lbls["eye"].configure(
            text="CLOSED" if s["eye_closed"] else "OPEN",
            text_color="#EF5350" if s["eye_closed"] else "#66BB6A")

        face_map = {
            (True,  True):  ("ACTIVE",   "#66BB6A"),
            (False, True):  ("LAST POS", "#FFA726"),
            (False, False): ("NONE",     "#78909C"),
            (True,  False): ("NONE",     "#78909C"),
        }
        face_txt, face_col = face_map.get(
            (s["face_active"], s["face_found"]), ("–", "gray"))
        self._stat_lbls["face"].configure(text=face_txt, text_color=face_col)

        m, sec = divmod(s["session_sec"], 60)
        self._stat_lbls["elapsed"].configure(text=f"{m:02d}:{sec:02d}")
        self._stat_lbls["fps"].configure(text=f"{s['fps']:.1f}")

    def on_close(self):
        self._stop()
        self.destroy()


if __name__ == "__main__":
    app = DrowsinessApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
