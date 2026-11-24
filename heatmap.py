# heatmap.py
import cv2, numpy as np

def make_overlay(img, cam, alpha=0.5):
    """img: processed image 0–1 (H×W). cam: model attention grid (h×w). Returns BGR uint8."""
    cam = cam.astype(np.float32)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=12)
    cam = np.clip(cam / (cam.max() + 1e-6), 0, 1)

    gray = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)

    a = cam[..., None] * alpha                 # weak attention stays transparent
    out = base * (1 - a) + heat * a
    tissue = (gray > 10)[..., None]            # keep the black background black
    return np.where(tissue, out, base).astype(np.uint8)