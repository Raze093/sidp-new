import numpy as np
import onnxruntime as ort

EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

class FerPlusOnnx:
    """
    FER+ ONNX expects input (1,1,64,64) grayscale.
    Output (1,8) emotion scores corresponding to EMOTION_LABELS.
    """
    def __init__(self, model_path: str, use_gpu: bool = True):
        providers = ["CPUExecutionProvider"]
        if use_gpu:
            # Requires onnxruntime-gpu installed
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess_face_bgr(self, face_bgr: np.ndarray) -> np.ndarray:
        # Convert to grayscale (fast manual conversion)
        gray = (
            0.114 * face_bgr[:, :, 0] +
            0.587 * face_bgr[:, :, 1] +
            0.299 * face_bgr[:, :, 2]
        ).astype(np.uint8)

        import cv2
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        x = gray[np.newaxis, np.newaxis, :, :].astype(np.float32)
        return x

    def predict(self, face_bgr: np.ndarray):
        x = self.preprocess_face_bgr(face_bgr)
        scores = self.sess.run(None, {self.input_name: x})[0]  # (1,8)
        scores = np.squeeze(scores).astype(np.float32)         # (8,)
        probs = softmax(scores)
        idx = int(np.argmax(probs))
        return EMOTION_LABELS[idx], float(probs[idx]), probs
