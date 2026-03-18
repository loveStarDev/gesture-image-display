# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import mediapipe as mp
import numpy as np
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =============================================
# 손동작-이미지 매핑
# =============================================
GESTURE_IMAGE_MAP = {
    "claw":      "images/1",   # 1. 호랑이 어흥 (손바닥+손가락 구부림)
    "peace":     "images/2",   # 2. V 포즈
    "thumb_up":  "images/3",   # 3. 엄지 치켜들기
    "gun":       "images/4",   # 4. 검지 아래+엄지 옆+나머지 말기 (ㅜ자)
}

GESTURE_LABELS = {
    "claw":     "[1] Claw / Tiger",
    "peace":    "[2] Peace / V",
    "thumb_up": "[3] Thumbs Up",
    "gun":      "[4] Gun / T-shape",
}

DISPLAY_DURATION = 90
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
IMAGE_WINDOW = "Image"
CAM_WINDOW   = "Gesture Cam"


def dist3(a, b):
    """3D 거리 (x, y, z 모두 사용)"""
    return ((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2) ** 0.5


def hand_scale(lm):
    """손 크기 기준 (손목~중지MCP 3D 거리)"""
    return dist3(lm[0], lm[9]) + 1e-6


def is_finger_extended(lm, tip, pip, mcp):
    """
    손가락 펼침 여부: 3D 거리로 tip-mcp > pip-mcp 이면 펼침
    손바닥/손등/방향 무관
    """
    return dist3(lm[tip], lm[mcp]) > dist3(lm[pip], lm[mcp]) * 1.2


def is_finger_bent_dist(lm, tip, mcp, scale):
    """손가락 구부림: tip-mcp 3D 거리가 손 크기 대비 작으면 구부림"""
    return dist3(lm[tip], lm[mcp]) < scale * 0.65


def get_finger_states(landmarks, is_right):
    """[thumb, index, middle, ring, pinky] 펼침 여부 - 방향 무관"""
    lm = landmarks
    fingers = []

    # 엄지: 좌우 방향으로 판별
    if is_right:
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)

    # 검지~새끼: 3D 거리 기반
    for tip, pip, mcp in zip([8, 12, 16, 20], [6, 10, 14, 18], [5, 9, 13, 17]):
        fingers.append(is_finger_extended(lm, tip, pip, mcp))

    return fingers  # [thumb, index, middle, ring, pinky]


def is_palm_facing(landmarks, is_right):
    wrist_x = landmarks[0].x
    mid_mcp_x = landmarks[9].x
    return wrist_x > mid_mcp_x if is_right else wrist_x < mid_mcp_x


def detect_claw(lm, fingers, palm_facing):
    """
    호랑이 어흥: 손바닥이 보이고 + 검지~새끼 4손가락 모두 구부러짐
    tip-mcp 거리 기반으로 구부림 판별 (방향 무관)
    """
    if not palm_facing:
        return False
    scale = hand_scale(lm)
    bent_count = sum(
        1 for tip, mcp in zip([8, 12, 16, 20], [5, 9, 13, 17])
        if is_finger_bent_dist(lm, tip, mcp, scale)
    )
    return bent_count >= 4


def detect_gun(lm, fingers, is_right):
    """
    ㅜ자 / 총 모양: 엄지+검지 펼침, 중지/약지/새끼 접힘
    손바닥/손등/방향 무관하게 손가락 상태만으로 판별
    """
    thumb, index, middle, ring, pinky = fingers
    return thumb and index and not middle and not ring and not pinky


def analyze_hand(landmarks, handedness):
    lm = landmarks
    label = handedness[0].category_name
    is_right = label == "Right"
    fingers = get_finger_states(lm, is_right)
    palm = is_palm_facing(lm, is_right)
    return {
        "lm":          lm,
        "fingers":     fingers,
        "is_right":    is_right,
        "palm_facing": palm,
    }


def detect_gesture(hands_data):
    if not hands_data:
        return None

    for h in hands_data:
        lm = h["lm"]
        fingers = h["fingers"]
        is_right = h["is_right"]
        palm = h["palm_facing"]
        thumb, index, middle, ring, pinky = fingers

        # 1. 호랑이 어흥: 손바닥 보임 + 4손가락 구부림
        if detect_claw(lm, fingers, palm):
            return "claw"

        # 2. V / 브이: 검지+중지 펼침, 나머지 접힘 (손바닥/손등 무관)
        if index and middle and not ring and not pinky and not thumb:
            return "peace"

        # 3. 엄지 치켜들기: 엄지만 펼침 (손바닥/손등 무관)
        if thumb and not index and not middle and not ring and not pinky:
            return "thumb_up"

        # 4. ㅜ자 / 총: 엄지+검지 펼침, 나머지 접힘, 검지가 아래를 향함
        if detect_gun(lm, fingers, is_right):
            return "gun"

    return None


def draw_landmarks_on_frame(frame, landmarks_list):
    h, w = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]
    for lm_list in landmarks_list:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def load_images():
    images = {}
    exts = [".jpeg", ".jpg", ".png", ".bmp", ".webp"]
    for gesture, base_path in GESTURE_IMAGE_MAP.items():
        base_full = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_path)
        for ext in exts:
            full_path = base_full + ext
            if os.path.exists(full_path):
                img = cv2.imread(full_path)
                if img is not None:
                    images[gesture] = img
                    print(f"[OK] {gesture} -> {base_path}{ext}")
                    break
        if gesture not in images:
            print(f"[INFO] No image for: {base_path}")
    return images


def draw_status_bar(frame, gesture, counter):
    h, w = frame.shape[:2]
    bar_h = 44

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    if gesture:
        text = GESTURE_LABELS.get(gesture, gesture)
        cv2.putText(frame, text, (14, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 230, 50), 2)
        ratio = counter / DISPLAY_DURATION
        bar_w = int((w - 28) * ratio)
        cv2.rectangle(frame, (14, h - 6), (14 + bar_w, h - 2), (50, 200, 50), -1)
    else:
        cv2.putText(frame, "Waiting for gesture...", (14, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 1)

    cv2.line(frame, (0, h - bar_h), (w, h - bar_h), (60, 60, 60), 1)
    cv2.putText(frame, "Q:Quit", (w - 72, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)


def show_image_window(img):
    screen_w, screen_h = 800, 600
    ih, iw = img.shape[:2]
    scale = min(screen_w / iw, screen_h / ih, 1.0)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh))
    cv2.imshow(IMAGE_WINDOW, resized)


def main():
    print("=== Gesture Image Display ===")
    for k, v in GESTURE_LABELS.items():
        print(f"  {v}")
    print("  Q - Quit\n")

    images = load_images()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    current_gesture = None
    display_counter = 0
    image_showing = False

    print("Camera opened. Starting detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        hands_data = []
        landmarks_for_draw = []
        if result.hand_landmarks:
            for lm_list, handedness in zip(result.hand_landmarks, result.handedness):
                hands_data.append(analyze_hand(lm_list, handedness))
                landmarks_for_draw.append(lm_list)

        draw_landmarks_on_frame(frame, landmarks_for_draw)

        detected = detect_gesture(hands_data)

        if detected:
            if detected != current_gesture:
                print(f"Detected: {detected}")
            current_gesture = detected
            display_counter = DISPLAY_DURATION
        else:
            current_gesture = None
            display_counter = 0

        # 이미지 별도 창 표시
        if current_gesture and current_gesture in images:
            show_image_window(images[current_gesture])
            image_showing = True
        elif image_showing:
            cv2.destroyWindow(IMAGE_WINDOW)
            image_showing = False

        draw_status_bar(frame, current_gesture, display_counter)
        cv2.imshow(CAM_WINDOW, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
