# -*- coding: utf-8 -*-
"""
Surucu Uyusukluk Tespiti - Real-Time Sistemi
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
import time
import winsound
from collections import deque

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────
IMG_SHAPE        = 128
EYE_THRESHOLD    = 0.5
YAWN_THRESHOLD   = 0.85

PERCLOS_SECONDS  = 12
PERCLOS_ALARM    = 0.50   # standart: skor 2 esigi

YAWN_COUNT_ALARM = 4
YAWN_RESET_SEC   = 60

RISK_LOW    = 23   # skor 1: P >= %20
RISK_MEDIUM = 47   # skor 2: P >= %50
RISK_HIGH   = 70   # skor 3: P >= %80
BEEP_INTERVAL = 3.0

EYE_MODEL_PATH  = "eye_model_compat.h5"
YAWN_MODEL_PATH = "yawn_model_compat.h5"

# Agiz bolgesi (yuzun alt yuzdesi)
AGIZ_UST = 0.62

# MediaPipe Face Mesh - sol/sag goz landmark indeksleri
LEFT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# ─────────────────────────────────────────────
# SES
# ─────────────────────────────────────────────
def play_beep(high=False):
    winsound.Beep(1200 if high else 900, 400)

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def load_models():
    print("Modeller yukleniyor...")
    eye_model  = load_model(EYE_MODEL_PATH, compile=False)
    yawn_model = load_model(YAWN_MODEL_PATH, compile=False)
    print("Modeller yuklendi.")
    return eye_model, yawn_model

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_eye(roi):
    img = cv2.resize(roi, (IMG_SHAPE, IMG_SHAPE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_yawn(roi):
    img = cv2.resize(roi, (IMG_SHAPE, IMG_SHAPE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

# ─────────────────────────────────────────────
# RISK
# ─────────────────────────────────────────────
def perclos_score(perclos):
    if perclos >= 0.80:   return 3
    elif perclos >= 0.50: return 2
    elif perclos >= 0.20: return 1
    else:                 return 0

def compute_risk(perclos, yawn_count):
    ps = (perclos_score(perclos) / 3) * 70   # skor 0-3 → 0-70
    ys = min(yawn_count / YAWN_COUNT_ALARM, 1.0) * 30
    return int(ps + ys)

def risk_color(s):
    if s >= RISK_HIGH:       return (0, 0, 255)
    elif s >= RISK_MEDIUM:   return (0, 140, 255)
    elif s >= RISK_LOW:      return (0, 220, 255)
    else:                    return (0, 200, 0)

def risk_label(s):
    if s >= RISK_HIGH:       return "YUKSEK RISK"
    elif s >= RISK_MEDIUM:   return "ORTA RISK"
    elif s >= RISK_LOW:      return "DUSUK RISK"
    else:                    return "NORMAL"

# ─────────────────────────────────────────────
# PERCLOS TRACKER
# ─────────────────────────────────────────────
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
        if not self.history:
            return 0.0
        return sum(v for _, v in self.history) / len(self.history)

    def elapsed(self):
        if not self.history:
            return 0
        return min(time.time() - self.history[0][0], self.window_sec)

# ─────────────────────────────────────────────
# ANA DONGU
# ─────────────────────────────────────────────
def run_detection():
    eye_model, yawn_model = load_models()

    face_mesh = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path='face_landmarker.task'),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Kamera acilamadi!")

    perclos_tracker  = PerclosTracker(PERCLOS_SECONDS)
    yawn_count       = 0
    last_yawn_state  = False
    yawn_consec      = 0
    last_beep_time   = 0
    yawn_timer_start = time.time()
    face_detected    = False

    print("Baslatildi. Cikis: q | Sifirla: r")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        if time.time() - yawn_timer_start > YAWN_RESET_SEC:
            yawn_count = 0
            yawn_timer_start = time.time()

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = face_mesh.detect(mp_img)

        eye_closed    = False
        yawn_detect   = False
        face_detected = len(results.face_landmarks) > 0

        if face_detected:
            lms = results.face_landmarks[0]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

            # Yuz siniri
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            fx = max(0, min(xs)); fy = max(0, min(ys))
            fw = min(w, max(xs)) - fx; fh = min(h, max(ys)) - fy
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 200, 0), 2)

            # ── GÖZ TESPİTİ (MediaPipe landmark) ─────
            eye_preds = []
            for eye_idx, tag in [(LEFT_EYE, "Sol"), (RIGHT_EYE, "Sag")]:
                eye_pts = [pts[i] for i in eye_idx]
                ex1 = max(0, min(p[0] for p in eye_pts) - 8)
                ey1 = max(0, min(p[1] for p in eye_pts) - 6)
                ex2 = min(w, max(p[0] for p in eye_pts) + 8)
                ey2 = min(h, max(p[1] for p in eye_pts) + 6)
                eye_crop = frame[ey1:ey2, ex1:ex2]
                if eye_crop.size == 0:
                    continue
                pred  = eye_model.predict(preprocess_eye(eye_crop), verbose=0)[0][0]
                eye_preds.append(pred)
                durum = "ACIK" if pred > EYE_THRESHOLD else "KAPALI"
                renk  = (0, 255, 0) if pred > EYE_THRESHOLD else (0, 0, 255)
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), renk, 2)
                cv2.putText(frame, f"{tag}:{durum}({pred:.2f})",
                            (ex1, ey1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, renk, 1)

            if eye_preds:
                eye_closed = (sum(eye_preds) / len(eye_preds)) < EYE_THRESHOLD

            # ── AGIZ BÖLGESİ ─────────────────────
            agiz_ust_px = int(fh * AGIZ_UST)
            agiz_roi    = frame[fy + agiz_ust_px:fy + fh, fx:fx + fw]

            if agiz_roi.shape[0] > 10 and agiz_roi.shape[1] > 10:
                yawn_pred   = yawn_model.predict(preprocess_yawn(agiz_roi), verbose=0)[0][0]
                yawn_detect = yawn_pred > YAWN_THRESHOLD
                yawn_txt    = f"YAWN({yawn_pred:.2f})" if yawn_detect else f"NoYawn({yawn_pred:.2f})"
                yawn_renk   = (0, 0, 255) if yawn_detect else (0, 255, 0)
                cv2.putText(frame, yawn_txt,
                            (fx, fy + fh + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, yawn_renk, 2)
                cv2.rectangle(frame,
                              (fx,      fy + agiz_ust_px),
                              (fx + fw, fy + fh),
                              (255, 100, 0), 2)

            if yawn_detect:
                yawn_consec += 1
            else:
                yawn_consec = 0

            sustained = yawn_consec >= 8
            if sustained and not last_yawn_state:
                yawn_count += 1
            last_yawn_state = sustained

        # ── PERCLOS ─────────────────────────────
        perclos_tracker.update(eye_closed)
        perclos = perclos_tracker.get()
        elapsed = perclos_tracker.elapsed()

        # ── RISK ─────────────────────────────────
        risk    = compute_risk(perclos, yawn_count)
        r_color = risk_color(risk)
        r_label = risk_label(risk)

        # ── UYARI ────────────────────────────────
        perclos_alarm = perclos >= PERCLOS_ALARM and elapsed >= 5
        yawn_alarm    = yawn_count >= YAWN_COUNT_ALARM
        alarm_active  = perclos_alarm or yawn_alarm

        now = time.time()
        if alarm_active and (now - last_beep_time) > BEEP_INTERVAL:
            play_beep(high=perclos_alarm)
            last_beep_time = now

        # ── EKRAN ────────────────────────────────
        perclos_pct   = int(perclos * 100)
        perclos_color = (0, 0, 255) if perclos_alarm else (0, 255, 100)

        cv2.putText(frame, f"PERCLOS: %{perclos_pct}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, perclos_color, 2)

        cv2.putText(frame, f"Esnemek: {yawn_count}/{YAWN_COUNT_ALARM}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255) if yawn_alarm else (0, 220, 100), 2)
        cv2.putText(frame, f"Goz: {'KAPALI' if eye_closed else 'ACIK'}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255) if eye_closed else (0, 255, 0), 2)

        # Risk kutusu
        cv2.rectangle(frame, (w-210, 10), (w-10, 105), r_color, -1)
        cv2.rectangle(frame, (w-210, 10), (w-10, 105), (255, 255, 255), 2)
        cv2.putText(frame, f"RISK: {risk}/100",
                    (w-200, 48), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, r_label,
                    (w-200, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Risk cubugu
        bar_w = int((w - 20) * risk / 100)
        cv2.rectangle(frame, (10, h-30), (w-10, h-10), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h-30), (10+bar_w, h-10), r_color, -1)
        cv2.putText(frame, f"RISK SKORU  ({PERCLOS_SECONDS}sn pencere)",
                    (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        # Alarm overlay
        if alarm_active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            reason = []
            if perclos_alarm: reason.append(f"PERCLOS %{perclos_pct}")
            if yawn_alarm:    reason.append(f"Esnemek {yawn_count}x")
            cv2.putText(frame, "! UYKU UYARISI: " + " | ".join(reason),
                        (w//2 - 250, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Surucu Uyusukluk Tespiti", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            yawn_count = 0
            perclos_tracker.history.clear()
            yawn_timer_start = time.time()
            print("Sifirlandi.")

    cap.release()
    cv2.destroyAllWindows()
    print("Sistem kapatildi.")


if __name__ == "__main__":
    run_detection()
