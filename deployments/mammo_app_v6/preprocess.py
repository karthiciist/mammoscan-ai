import cv2, numpy as np

def load(path):
    if path.lower().endswith(".dcm"):
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        d = pydicom.dcmread(path)
        img = apply_voi_lut(d.pixel_array, d).astype(np.float32)
        if d.PhotometricInterpretation == "MONOCHROME1":
            img = img.max() - img
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if img.ndim == 3:
            img = img.mean(-1)
    return (img - img.min()) / (img.max() - img.min() + 1e-6)

def preprocess(path, out_hw=(1024, 512)):
    img = load(path)
    mask = (img > 0.05).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:                                            # crop to breast
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]
    if img[:, :img.shape[1]//2].sum() < img[:, img.shape[1]//2:].sum():
        img = np.fliplr(img)                            # breast faces left
    img = cv2.resize(np.ascontiguousarray(img), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)
    return (img * 65535).astype(np.uint16)