import time
import cv2
import numpy as np
import torch
from ai_emotion_panel import EmotionAIPanel
from ultralytics import YOLO
from emotion_ferplus import FerPlusOnnx, EMOTION_LABELS

# ----------------------------
# Settings
# ----------------------------
FACE_MODEL_PATH = "yolov8n-face-lindevs.pt"
EMO_MODEL_PATH = "emotion-ferplus-8.onnx"

LLM_MODE = "groq"  # "auto" = gemini -> groq -> local | "gemini" | "groq" | "local"

CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480      # camera window resolution
TARGET_FPS = 30

FACE_CONF = 0.50
CROP_MARGIN = 0.15

# Video performance knobs
DETECT_EVERY_N_FRAMES = 1          # face detection every frame
EMOTION_EVERY_N_FRAMES = 6         # emotion detection every 6 frames

# Video smoothing (reduces jitter)
SMOOTHING_ALPHA = 0.35             # higher = more responsive, lower = smoother

# ----------------------------
# Helper Functions
# ----------------------------

# Clamp a value v into the inclusive range [lo, hi] (prevents box coords going out of frame bounds).
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Draw a readable text label with a black background box at image position (x, y).
def put_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - baseline - 8), (x + tw + 10, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 5, y - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

# From multiple detected face boxes, select the index of the box with the largest area (assumes one main face).
def pick_largest_box(xyxy: np.ndarray):
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(np.argmax(areas))

# Expand a bounding box by a margin percentage and clamp it to the image boundaries.
def expand_box(box, w, h, margin=0.15):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    mx = bw * margin
    my = bh * margin
    x1 = clamp(int(x1 - mx), 0, w - 1)
    y1 = clamp(int(y1 - my), 0, h - 1)
    x2 = clamp(int(x2 + mx), 0, w - 1)
    y2 = clamp(int(y2 + my), 0, h - 1)
    return x1, y1, x2, y2

# ----------------------------
# Main
# ----------------------------

# Capture frames from the webcam, run face detection + emotion recognition, and display results (plus AI response panel).
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Torch device: {device}")

    # Load models
    face_model = YOLO(FACE_MODEL_PATH)
    emo_model = FerPlusOnnx(EMO_MODEL_PATH, use_gpu=True)

    # 30fps but very slow to open the camera
    # cap = cv2.VideoCapture(CAM_INDEX)
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_MSMF)

    # Very fast to open the camera but 15fps
    # cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    panel = EmotionAIPanel(config_path="api_keys.json", width=520, height=480, llm_mode=LLM_MODE)
    panel.setup()

    # Move windows so they sit next to each other
    cv2.namedWindow("Face + Emotion (YOLOv8-Face + FER+)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face + Emotion (YOLOv8-Face + FER+)", FRAME_W, FRAME_H)
    cv2.moveWindow("Face + Emotion (YOLOv8-Face + FER+)", 50, 50)
    cv2.moveWindow(panel.window_name, 50 + FRAME_W + 20, 50)

    last_face_box = None  # (x1,y1,x2,y2) for reuse when skipping detection
    last_label = "neutral"
    last_conf = 0.0
    smoothed_probs = None

    frame_count = 0
    fps = 0.0
    t_fps = time.time()
    frames_for_fps = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        frames_for_fps += 1
        frame_count += 1

        # ----------------------------
        # Face detection (optionally skip some frames)
        # ----------------------------
        if (frame_count % DETECT_EVERY_N_FRAMES) == 0 or last_face_box is None:
            res = face_model.predict(
                frame,
                conf=FACE_CONF,
                device=device,
                verbose=False
            )[0]

            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                bi = pick_largest_box(boxes)  # one-face: pick largest
                x1, y1, x2, y2 = expand_box(boxes[bi], w, h, margin=CROP_MARGIN)
                last_face_box = (x1, y1, x2, y2)
            else:
                last_face_box = None

        # ----------------------------
        # Emotion (run every N frames if face exists)
        # ----------------------------
        if last_face_box is not None:
            x1, y1, x2, y2 = last_face_box
            face_crop = frame[y1:y2, x1:x2]

            if (frame_count % EMOTION_EVERY_N_FRAMES) == 0 and face_crop.size > 0:
                label, conf, probs = emo_model.predict(face_crop)

                if smoothed_probs is None:
                    smoothed_probs = probs
                else:
                    smoothed_probs = (1.0 - SMOOTHING_ALPHA) * smoothed_probs + SMOOTHING_ALPHA * probs

                li = int(np.argmax(smoothed_probs))
                last_label = EMOTION_LABELS[li]

                # Feed current dominant emotion label to the AI panel's 5-second collector
                if last_face_box is not None:
                    panel.update_emotion_sample(last_label)

                last_conf = float(smoothed_probs[li])

            # Draw overlay
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            put_label(frame, f"{last_label} {last_conf:.2f}", x1, y1)

        # ----------------------------
        # FPS meter
        # ----------------------------
        dt = time.time() - t_fps
        if dt >= 0.5:
            fps = frames_for_fps / dt
            frames_for_fps = 0
            t_fps = time.time()

        put_label(frame, f"FPS: {fps:.1f} | det:{DETECT_EVERY_N_FRAMES} emo:{EMOTION_EVERY_N_FRAMES}", 10, 30)
        cv2.imshow("Face + Emotion (YOLOv8-Face + FER+)", frame)

        # Draw + show the AI response panel window
        panel_img = panel.draw_panel()
        cv2.imshow(panel.window_name, panel_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()