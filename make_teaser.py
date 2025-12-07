#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_teaser.py
Generate teaser overlays for 5 CAMs on two inputs:
1) a user-supplied cat image, and
2) a generated QR image with text "cat".

Outputs: teaser/*.png (10 images total)
- One overlay per CAM per input, labeled with the CAM name on the image.
- Filenames: <input>_<cam>.png

Designed to run within ~2 GB VRAM:
- Single-image forward pass
- Grad-CAM hooks closed via context manager
- Explicit CUDA cache clears between methods
"""

import os, argparse, random
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import qrcode
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, LayerCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ----------- Repro & device -----------
SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- CAM set (5 methods -> 10 images total) -----------
CAM_CLASSES = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
    "layercam": LayerCAM,
    "eigengradcam": EigenGradCAM,
}

# ----------- Preprocessing -----------
pre_tf_224 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225))
])

def load_resnet50_imagenet():
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    m = torchvision.models.resnet50(weights=w)
    m.eval().to(device)
    return m

def find_last_conv(model: nn.Module) -> nn.Module:
    last = None
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    if last is None:
        raise RuntimeError("No Conv2d found for CAM target layer.")
    return last

# ----------- QR generation (text='cat') -----------
def make_qr_matrix(text="cat", error='M'):
    qr = qrcode.QRCode(
        version=None,  # auto-fit
        error_correction=getattr(qrcode.constants, f'ERROR_CORRECT_{error}'),
        box_size=1, border=0
    )
    qr.add_data(text)
    qr.make(fit=True)
    return np.array(qr.get_matrix(), dtype=np.uint8)  # 1=black module

def render_qr_from_matrix(mat, scale=12, bg=255):
    # white background 255, black modules 0
    img = np.kron(1 - mat, np.ones((scale, scale), dtype=np.uint8)) * 255
    if bg != 255:
        img[img == 255] = bg
    return img  # grayscale uint8

# ----------- Helpers -----------
def norm01(a):
    a = a.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        a = (a - mn) / (mx - mn)
    else:
        a = np.zeros_like(a, dtype=np.float32)
    return a

def overlay_heatmap_on_gray(gray_u8, heat01, alpha=0.55):
    """Return BGR overlay at original resolution."""
    H, W = gray_u8.shape[:2]
    heat = cv2.resize(heat01.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    heat_u8 = (heat * 255.0).clip(0,255).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    if gray_u8.ndim == 2:
        base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    else:
        base = gray_u8.copy()
        if base.shape[2] == 4:
            base = base[:, :, :3]
    out = cv2.addWeighted(base, 1.0, cm, alpha, 0.0)
    return out

def label_on_image(bgr, text, x=8, y=28):
    """Draw a filled label box + white text."""
    out = bgr.copy()
    # background rectangle
    cv2.rectangle(out, (x-6, y-22), (x + 8 + 9*len(text), y+8), (0,0,0), thickness=-1)
    # text
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return out

def ensure_gray_u8(img):
    """Accepts BGR/RGB/GRAY, returns GRAY uint8."""
    if img.ndim == 2:
        g = img
    elif img.ndim == 3:
        if img.shape[2] == 3:
            # assume BGR or RGB? We'll detect by heuristic: most user images read by cv2 are BGR.
            # Since we don't know, convert using cv2.cvtColor assuming BGR->GRAY first.
            # If user provided RGB via PIL, still OK for CAM.
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            g = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image array shape.")
    if g.dtype != np.uint8:
        g = g.clip(0,255).astype(np.uint8)
    return g

def run_cam_and_save(model, target_layer, gray_u8, cam_name, out_path_png):
    """Compute CAM heatmap for the predicted class and save overlay."""
    # to RGB and preprocess
    rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
    x = pre_tf_224(rgb).unsqueeze(0).to(device)
    x.requires_grad_(True)

    # forward for predicted class
    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(1).item())

    CAMCls = CAM_CLASSES[cam_name]
    with CAMCls(model=model, target_layers=[target_layer]) as cam:
        # enable grad only during CAM call
        torch.cuda.empty_cache()
        targets = [ClassifierOutputTarget(pred)]
        heat = cam(input_tensor=x, targets=targets)[0]  # (224,224) float in [0,1]
    heat01 = norm01(heat)
    overlay = overlay_heatmap_on_gray(gray_u8, heat01, alpha=0.55)
    overlay = label_on_image(overlay, cam_name.upper())
    cv2.imwrite(str(out_path_png), overlay)
    # free cache between methods to stay within 2GB
    del x, heat
    torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True,
                    help="Path to a cat image (jpg/png).")
    ap.add_argument("--out", default="teaser",
                    help="Output folder (default: teaser)")
    ap.add_argument("--qr_scale", type=int, default=18,
                    help="Module scale for QR rendering (higher => higher resolution). Default 18.")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Output folder: {out_dir.resolve()}")

    # Load model
    model = load_resnet50_imagenet()
    target_layer = find_last_conv(model)
    for p in target_layer.parameters():
        p.requires_grad_(True)  # only target conv needs grads

    # --- Load cat image ---
    cat_path = Path(args.cat)
    if not cat_path.exists():
        raise FileNotFoundError(f"Cat image not found: {cat_path}")
    cat_bgr = cv2.imread(str(cat_path), cv2.IMREAD_COLOR)
    if cat_bgr is None:
        raise RuntimeError(f"Failed to read image: {cat_path}")
    cat_gray = ensure_gray_u8(cat_bgr)

    # --- Generate QR image for text "cat" ---
    mat = make_qr_matrix(text="cat", error='M')
    qr_gray = render_qr_from_matrix(mat, scale=args.qr_scale, bg=255)  # high-res grayscale

    # --- CAMs ---
    inputs = [("cat", cat_gray), ("qr_cat", qr_gray)]
    for inp_name, gray in inputs:
        for cam_name in CAM_CLASSES.keys():
            out_png = out_dir / f"{inp_name}_{cam_name}.png"
            print(f"[RUN] {inp_name} - {cam_name} -> {out_png.name}")
            run_cam_and_save(model, target_layer, gray, cam_name, out_png)

    print("[DONE] Wrote 10 images to:", out_dir.resolve())

if __name__ == "__main__":
    main()
