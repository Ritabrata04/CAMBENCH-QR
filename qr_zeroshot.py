#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qr_zero_shot.py
Zero-shot (frozen backbone + linear head) benchmark of multiple CAM methods
on synthetic QR vs non-QR classification with structure-aware metrics.

Outputs (CSV/PNGs) are saved under: /home/ritabrata/qr_stuff/outputs_zero_shot

Author: you :)
"""

import os, time, random, copy, math, string
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import qrcode

# CAM methods
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, XGradCAM,
    LayerCAM, EigenCAM, EigenGradCAM,
    ScoreCAM, AblationCAM
)
from pytorch_grad_cam.ablation_layer import AblationLayerVit  # not used, but keeps import consistent
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------------------- Paths & device ----------------------
BASEDIR = Path("/home/ritabrata/qr_stuff")
OUTDIR  = BASEDIR / "outputs_zero_shot"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"[INFO] Using device: {device}")

# ---------------------- Utilities ----------------------
def to_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def norm01(a):
    a = a.astype(np.float32)
    m, M = a.min(), a.max()
    return (a - m) / (M - m + 1e-6)

# ---------------------- QR helpers (auto-fit) ----------------------
def version_from_mat(mat: np.ndarray) -> int:
    N = mat.shape[0]
    return int((N - 21)//4 + 1)

def make_qr_matrix_autofit(data: str, error: str = 'M') -> np.ndarray:
    qr = qrcode.QRCode(
        version=None,
        error_correction=getattr(qrcode.constants, f'ERROR_CORRECT_{error}'),
        box_size=1, border=0
    )
    qr.add_data(data); qr.make(fit=True)
    return np.array(qr.get_matrix(), dtype=np.uint8)  # 1 = black

def random_qr_text(L: int) -> str:
    alphabet = string.ascii_uppercase + string.digits + " $%*+-./:"
    return "".join(random.choice(alphabet) for _ in range(L))

def masks_from_matrix(mat):
    N = mat.shape[0]
    M_f = np.zeros_like(mat, dtype=np.uint8)
    for (r,c) in [(0,0),(0,N-7),(N-7,0)]:
        M_f[r:r+7, c:c+7] = 1
    M_t = np.zeros_like(mat, dtype=np.uint8)
    M_t[6,:] = 1; M_t[:,6] = 1
    for (r,c) in [(0,0),(0,N-7),(N-7,0)]:
        M_t[r:r+7, c:c+7] = 0
    M_box = np.ones_like(mat, dtype=np.uint8)
    return M_f, M_t, M_box

def render_from_modules(mat, scale=10, bg=255):
    img = np.kron(1 - mat, np.ones((scale, scale), dtype=np.uint8)) * 255
    if bg != 255:
        img[img==255] = bg
    return img

def warp_affine_mask(mask_bin, M, W, H):
    warped = cv2.warpAffine(mask_bin.astype(np.uint8)*255, M, (W,H),
                            flags=cv2.INTER_NEAREST, borderValue=0)
    return (warped>127).astype(np.uint8)

def warp_perspective_mask(mask_bin, Hmat, W, H):
    warped = cv2.warpPerspective(mask_bin.astype(np.uint8)*255, Hmat, (W,H),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
    return (warped>127).astype(np.uint8)

def random_perspective(src_size, max_perturb=0.10):
    W=H=src_size
    src = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    dst = src.copy()
    for i in range(4):
        dx = np.random.uniform(-max_perturb, max_perturb)*W
        dy = np.random.uniform(-max_perturb, max_perturb)*H
        dst[i] = (src[i][0]+dx, src[i][1]+dy)
    Hmat = cv2.getPerspectiveTransform(src, dst)
    return Hmat

def apply_distortions(img, masks, rotate_deg=0, blur_ksize=0,
                      contrast=1.0, brightness_shift=0,
                      occlude=False, motion=False, perspective=False):
    H, W = img.shape[:2]
    out = img.astype(np.float32)

    # photometric
    out = np.clip(out*contrast + brightness_shift, 0, 255)

    # rotation
    if rotate_deg != 0:
        M = cv2.getRotationMatrix2D((W/2, H/2), rotate_deg, 1.0)
        out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=255)
        masks = [warp_affine_mask(m, M, W, H) for m in masks]

    # perspective
    if perspective:
        Hmat = random_perspective(W, max_perturb=0.10)
        out = cv2.warpPerspective(out, Hmat, (W, H),
                                  flags=cv2.INTER_NEAREST, borderValue=255)
        masks = [warp_perspective_mask(m, Hmat, W, H) for m in masks]

    # blur
    if blur_ksize > 0:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        out = cv2.GaussianBlur(out, (k,k), 0)

    # motion blur
    if motion:
        k = random.choice([5,7,9])
        kernel = np.zeros((k,k), dtype=np.float32); kernel[k//2,:] = 1.0/k
        out = cv2.filter2D(out, -1, kernel)

    # occlusion
    if occlude:
        x0 = np.random.randint(0, W//2); y0 = np.random.randint(0, H//2)
        s  = np.random.randint(H//10, H//4)
        out[y0:y0+s, x0:x0+s] = 255

    return out.astype(np.uint8), masks

def expand_mask_to_scale(mask_bin: np.ndarray, scale: int) -> np.ndarray:
    """Repeat each module into a scale×scale pixel block."""
    return np.kron(mask_bin.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8))


def generate_qr_rich(max_version=6, scale_choices=(8,10,12)):
    # keep version small; retry with shorter text/ECC if needed
    for _ in range(12):
        ecc  = random.choice(list("LMQH"))
        L    = random.randint(6, 16)
        text = random_qr_text(L)
        mat  = make_qr_matrix_autofit(text, error=ecc)
        v    = version_from_mat(mat)
        if v <= max_version:
            break
        # shrink data
        for L2 in [10,8,6,4]:
            text = random_qr_text(L2)
            mat  = make_qr_matrix_autofit(text, error=ecc)
            v    = version_from_mat(mat)
            if v <= max_version: break
        if v <= max_version: break
        # relax ECC
        mat  = make_qr_matrix_autofit(random_qr_text(4), error='L')
        v    = version_from_mat(mat)
        if v <= max_version: break

    # module-space masks (N×N)
    Mf_mod, Mt_mod, Mb_mod = masks_from_matrix(mat)

    # image rendering
    scale = random.choice(scale_choices)
    bg    = random.randint(230,255)
    base  = render_from_modules(mat, scale=scale, bg=bg)    # (H,W) = (N*scale, N*scale)

    # >>> expand masks to pixel space (H,W) BEFORE distortions
    Mf = expand_mask_to_scale(Mf_mod, scale)
    Mt = expand_mask_to_scale(Mt_mod, scale)
    Mb = expand_mask_to_scale(Mb_mod, scale)

    # distortions (same M / Hmat applied to image AND masks)
    rot  = random.choice([0,5,-5,10,-10])
    blur = random.choice([0,1,2,3])
    ctr  = random.choice([0.85,1.0,1.15])
    br   = random.choice([-15,-5,0,5,15])
    occ  = random.random() < 0.25
    mot  = random.random() < 0.20
    persp= random.random() < 0.30

    img, (Mf, Mt, Mb) = apply_distortions(
        base, [Mf, Mt, Mb],
        rotate_deg=rot, blur_ksize=blur,
        contrast=ctr, brightness_shift=br,
        occlude=occ, motion=mot, perspective=persp
    )

    # ensure binary {0,1} for masks
    Mf = (Mf > 0).astype(np.uint8)
    Mt = (Mt > 0).astype(np.uint8)
    Mb = (Mb > 0).astype(np.uint8)
    return img, (Mf, Mt, Mb)  # all in (H,W) pixel space now


# ---------------------- Negatives ----------------------
def checkerboard(size=280, cell=20):
    img = np.ones((size,size), np.uint8)*255
    for i in range(0,size,cell):
        for j in range(0,size,cell):
            if ((i//cell)+(j//cell))%2==0: img[i:i+cell,j:j+cell]=0
    return img

def random_shapes(size=280, n=20):
    img = np.ones((size,size), np.uint8)*255
    for _ in range(n):
        x1,y1 = np.random.randint(0,size-10,2)
        x2 = np.random.randint(x1+5,min(size,x1+80))
        y2 = np.random.randint(y1+5,min(size,y1+80))
        cv2.rectangle(img,(x1,y1),(x2,y2), random.choice([0,80,160,200]), -1)
    return img

def make_negative():
    return checkerboard() if random.random()<0.5 else random_shapes()

# ---------------------- Dataset & model ----------------------
pre_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225))
])

class SimpleDS(Dataset):
    def __init__(self, imgs, labels):
        self.imgs, self.labels = imgs, labels
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = self.imgs[i]
        if img.ndim==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return pre_tf(img), torch.tensor(self.labels[i], dtype=torch.long)

def build_split(n_qr_train=600, n_neg_train=600, n_qr_test=300, n_neg_test=300):
    # Train
    tr_imgs, tr_lbls = [], []
    for _ in tqdm(range(n_qr_train), desc="Train QR", ncols=100):
        img,_ = generate_qr_rich(max_version=6); tr_imgs.append(img); tr_lbls.append(1)
    for _ in tqdm(range(n_neg_train), desc="Train Non-QR", ncols=100):
        img = make_negative(); tr_imgs.append(img); tr_lbls.append(0)
    # Test (save masks for metrics)
    te_imgs, te_lbls, te_masks = [], [], []
    for _ in tqdm(range(n_qr_test), desc="Test QR", ncols=100):
        img, masks = generate_qr_rich(max_version=6); te_imgs.append(img); te_lbls.append(1); te_masks.append(masks)
    for _ in tqdm(range(n_neg_test), desc="Test Non-QR", ncols=100):
        img = make_negative(); te_imgs.append(img); te_lbls.append(0); te_masks.append((np.zeros_like(img), np.zeros_like(img), np.zeros_like(img)))

    # shuffle
    idx = list(range(len(tr_imgs))); random.shuffle(idx)
    tr_imgs = [tr_imgs[i] for i in idx]; tr_lbls = [tr_lbls[i] for i in idx]
    idx = list(range(len(te_imgs))); random.shuffle(idx)
    te_imgs = [te_imgs[i] for i in idx]; te_lbls = [te_lbls[i] for i in idx]; te_masks = [te_masks[i] for i in idx]
    return (tr_imgs, tr_lbls), (te_imgs, te_lbls, te_masks)

def build_model_head_only(num_classes=2):
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    m = torchvision.models.resnet50(weights=w)
    for p in m.parameters(): p.requires_grad_(False)
    feat = m.fc.in_features
    m.fc = nn.Linear(feat, num_classes)
    return m.to(device)

def train_head(model, train_dl, epochs=2, lr=1e-3):
    opt = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train(); total=correct=0; loss_sum=0.0
        pbar = tqdm(train_dl, desc=f"[train] epoch {ep}/{epochs}", ncols=100)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb); loss = ce(logits, yb)
            loss.backward(); opt.step()
            total += yb.size(0); correct += (logits.argmax(1)==yb).sum().item()
            loss_sum += loss.item()*yb.size(0)
            pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.3f}")
    model.eval()
    return model

def test_accuracy(model, imgs, labels, batch=256):
    ds = SimpleDS(imgs, labels)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))
    correct=total=0
    with torch.no_grad():
        for xb, yb in tqdm(dl, desc="[eval] accuracy", ncols=100):
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(1)
            correct += (pred==yb).sum().item(); total += yb.size(0)
    return correct/total

# ---------------------- CAM machinery ----------------------
def find_last_conv_layer(model: nn.Module) -> nn.Module:
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for CAM.")
    return last_conv

def get_cam(method_name: str):
    name = method_name.lower()
    if name == "gradcam":       return GradCAM
    if name == "gradcam++":     return GradCAMPlusPlus
    if name == "xgradcam":      return XGradCAM
    if name == "layercam":      return LayerCAM
    if name == "eigencam":      return EigenCAM
    if name == "eigengradcam":  return EigenGradCAM
    if name == "scorecam":      return ScoreCAM
    if name == "ablationcam":   return AblationCAM
    raise ValueError(f"Unknown CAM method: {method_name}")

def cam_on_image(model, img_gray_u8, target_layer, CAMCls):
    """Return CAM heatmap (float[0,1]) resized to image size."""
    rgb = cv2.cvtColor(img_gray_u8, cv2.COLOR_GRAY2RGB)
    x   = pre_tf(rgb).unsqueeze(0).to(device)
    x.requires_grad_(True)
    logits = model(x); pred = int(logits.argmax(1).item())
    # Avoid unsupported kwargs; some versions of ScoreCAM don't accept batch_size
    with CAMCls(model=model, target_layers=[target_layer]) as cam:
        model.zero_grad(set_to_none=True)
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]  # (224,224) in [0,1]
    H,W = img_gray_u8.shape[:2]
    return cv2.resize(heat.astype(np.float32), (W,H), interpolation=cv2.INTER_CUBIC)

# ---------------------- Metrics ----------------------
def _resize_mask_if_needed(mask: np.ndarray, hw: Tuple[int,int]) -> np.ndarray:
    """Resize binary mask to match (H,W) using nearest-neighbor if needed."""
    H, W = hw
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)

def FMR_TMR_BL(H01, Mf, Mt, Mb):
    # ensure all shapes match
    H, W = H01.shape[:2]
    Mf = _resize_mask_if_needed(Mf, (H, W))
    Mt = _resize_mask_if_needed(Mt, (H, W))
    Mb = _resize_mask_if_needed(Mb, (H, W))

    H01 = H01.astype(np.float32)
    eps = 1e-6
    S   = float(H01.sum()) + eps
    FMR = float((H01 * Mf).sum() / S)
    TMR = float((H01 * Mt).sum() / S)
    BL  = float((H01 * (1 - Mb)).sum() / S)
    return FMR, TMR, BL

def SLC_AUCs(H01, Mf, Mt, Mb, nT=31):
    H, W = H01.shape[:2]
    Mf = _resize_mask_if_needed(Mf, (H, W))
    Mt = _resize_mask_if_needed(Mt, (H, W))
    Mb = _resize_mask_if_needed(Mb, (H, W))

    H01 = H01.astype(np.float32)
    vals = H01.ravel()
    taus = np.quantile(vals, np.linspace(0.0, 0.99, nT))
    eps  = 1e-6
    insideF = []; insideT = []; outside = []
    for t in taus:
        mask = (H01 >= t).astype(np.float32)
        mass = mask.sum() + eps
        insideF.append(float((mask * Mf).sum()/mass))
        insideT.append(float((mask * Mt).sum()/mass))
        outside.append(float((mask * (1 - Mb)).sum()/mass))
    return float(np.mean(insideF)), float(np.mean(insideT)), float(np.mean(outside))

def distance_to_structure(H01, Mf, Mt):
    H, W = H01.shape[:2]
    Mf = _resize_mask_if_needed(Mf, (H, W))
    Mt = _resize_mask_if_needed(Mt, (H, W))
    M  = np.clip(Mf + Mt, 0, 1).astype(np.uint8)
    inv = (1 - M).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    norm = math.sqrt(H*H + W*W) + 1e-6
    H01  = H01.astype(np.float32)
    return float((H01 * dist).sum() / (H01.sum() + 1e-6)) / norm

# ---------------------- Visualization ----------------------
def overlay_heatmap_on_gray(gray, H01, alpha=0.6):
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cm = cv2.applyColorMap((H01*255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(gray3, 1.0, cm, alpha, 0)
    return out

# ---------------------- Main pipeline ----------------------
def main():
    # 1) Data
    print("[STEP] Building train/test splits…")
    (tr_imgs, tr_lbls), (te_imgs, te_lbls, te_masks) = build_split(
        n_qr_train=600, n_neg_train=600, n_qr_test=300, n_neg_test=300
    )

    # 2) Train head-only (zero-shot regime)
    print("[STEP] Training linear head on frozen ResNet-50…")
    model = build_model_head_only(2)
    train_dl = DataLoader(SimpleDS(tr_imgs, tr_lbls),
                          batch_size=256, shuffle=True,
                          num_workers=4, pin_memory=(device.type=="cuda"))
    model = train_head(model, train_dl, epochs=2, lr=1e-3)

    # 3) Accuracy on test
    acc = test_accuracy(model, te_imgs, te_lbls, batch=256)
    print(f"\n[RESULT] Zero-shot test accuracy: {acc:.4f}\n")

    # 4) CAM methods benchmark on positive class subset (structure metrics only make sense on QR)
    pos_indices = [i for i, y in enumerate(te_lbls) if y == 1]
    # To keep runtime reasonable (ScoreCAM/AblationCAM are slow), evaluate on up to N_eval_pos
    N_eval_pos = min(180, len(pos_indices))
    eval_ids = pos_indices[:N_eval_pos]
    print(f"[INFO] Evaluating CAMs on {N_eval_pos} QR images for structure metrics.")

    target_layer = find_last_conv_layer(model)
    # ensure last block grads are allowed for CAM (while keeping model.eval())
    for p in target_layer.parameters(): p.requires_grad_(True)
    model.eval()

    methods = [
        "gradcam", "gradcam++", "xgradcam", "layercam",
        "eigencam", "eigengradcam",
        "scorecam", "ablationcam"
    ]

    rows_all = []
    vis_dir = OUTDIR / "qual_zero_shot"
    vis_dir.mkdir(exist_ok=True)

    for mname in methods:
        CAMCls = get_cam(mname)
        print(f"\n[METHOD] {mname}")
        t0 = time.perf_counter()
        FMRs=[]; TMRs=[]; BLs=[]; AUCf=[]; AUCt=[]; AUCbg=[]; DtSs=[]
        # pick 6 examples for visualization
        vis_pick = set(random.sample(eval_ids, k=min(6, len(eval_ids))))
        with tqdm(total=len(eval_ids), ncols=100, desc=f"{mname} eval") as pbar:
            for j in eval_ids:
                img = te_imgs[j]; (Mf, Mt, Mb) = te_masks[j]
                Hmap = cam_on_image(model, img, target_layer, CAMCls)
                H01  = norm01(Hmap)

                # metrics
                FMR, TMR, BL = FMR_TMR_BL(H01, Mf, Mt, Mb)
                aucF, aucT, aucBG = SLC_AUCs(H01, Mf, Mt, Mb, nT=31)
                dts = distance_to_structure(H01, Mf, Mt)

                FMRs.append(FMR); TMRs.append(TMR); BLs.append(BL)
                AUCf.append(aucF); AUCt.append(aucT); AUCbg.append(aucBG); DtSs.append(dts)

                # optional qualitative save
                if j in vis_pick:
                    over = overlay_heatmap_on_gray(img, H01, alpha=0.55)
                    fn = vis_dir / f"{mname}_ex{j}.png"
                    cv2.imwrite(str(fn), over)

                pbar.update(1)

        elapsed = time.perf_counter() - t0
        ms_per_img = (elapsed / len(eval_ids)) * 1000.0

        res = dict(
            method=mname,
            FMR=np.mean(FMRs), TMR=np.mean(TMRs), BL=np.mean(BLs),
            AUC_MISF=np.mean(AUCf), AUC_MIST=np.mean(AUCt), AUC_BG=np.mean(AUCbg),
            DtS=np.mean(DtSs),
            ms_per_img=ms_per_img
        )
        # Composite structure score (you can tune weights)
        res["StructureScore"] = res["AUC_MISF"] + res["AUC_MIST"] - 3.0*res["AUC_BG"] - res["DtS"]

        rows_all.append(res)

        # Save per-method CSV too
        pd.DataFrame({
            "FMR":FMRs, "TMR":TMRs, "BL":BLs,
            "AUC_MISF":AUCf, "AUC_MIST":AUCt, "AUC_BG":AUCbg,
            "DtS":DtSs
        }).to_csv(OUTDIR / f"details_zero_shot_{mname}.csv", index=False)

        print(f"[{mname}] mean FMR={res['FMR']:.4f}  TMR={res['TMR']:.4f}  BL={res['BL']:.4f}  "
              f"AUC_F={res['AUC_MISF']:.3f}  AUC_T={res['AUC_MIST']:.3f}  AUC_BG={res['AUC_BG']:.3f}  "
              f"DtS={res['DtS']:.4f}  ms/img={res['ms_per_img']:.1f}  StructScore={res['StructureScore']:.3f}")

    df = pd.DataFrame(rows_all).sort_values(by=["StructureScore","AUC_MISF","AUC_MIST"], ascending=[False,False,False])
    df.insert(1, "accuracy_test", acc)
    df.to_csv(OUTDIR / "zero_shot_methods_summary.csv", index=False)

    print("\n==================== SUMMARY (Zero-shot) ====================")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("Saved:", OUTDIR / "zero_shot_methods_summary.csv")

    # quick bar plots
    plt.figure(figsize=(10,4))
    plt.bar(df["method"], df["StructureScore"])
    plt.title("Zero-shot: StructureScore by method"); plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(OUTDIR / "zero_shot_structurescore.png", dpi=160); plt.close()

    plt.figure(figsize=(10,4))
    plt.bar(df["method"], df["ms_per_img"])
    plt.title("Zero-shot: Latency (ms/img)"); plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(OUTDIR / "zero_shot_latency.png", dpi=160); plt.close()

    print("[DONE] Outputs saved to:", OUTDIR.resolve())
        # --- NEW: plot structural AUCs (mean ± std) and print caption ---
    # Load per-method per-image AUCs (written above) to get error bars
    auc_stats = []
    for mname in df["method"].tolist():
        det_path = OUTDIR / f"details_zero_shot_{mname}.csv"
        if det_path.exists():
            det = pd.read_csv(det_path)
            auc_stats.append({
                "method": mname,
                "AUC_MISF_mean": det["AUC_MISF"].mean(),
                "AUC_MIST_mean": det["AUC_MIST"].mean(),
                "AUC_BG_mean":   det["AUC_BG"].mean(),
                "AUC_MISF_std":  det["AUC_MISF"].std(ddof=1),
                "AUC_MIST_std":  det["AUC_MIST"].std(ddof=1),
                "AUC_BG_std":    det["AUC_BG"].std(ddof=1),
                "n": len(det)
            })
        else:
            # Fallback: use the summary row, no error bars
            row = df[df.method == mname].iloc[0]
            auc_stats.append({
                "method": mname,
                "AUC_MISF_mean": float(row["AUC_MISF"]),
                "AUC_MIST_mean": float(row["AUC_MIST"]),
                "AUC_BG_mean":   float(row["AUC_BG"]),
                "AUC_MISF_std":  0.0,
                "AUC_MIST_std":  0.0,
                "AUC_BG_std":    0.0,
                "n": int(N_eval_pos) if 'N_eval_pos' in locals() else 0
            })

    aucdf = pd.DataFrame(auc_stats)
    # keep the same method order as in df
    aucdf = aucdf.set_index("method").loc[df["method"]].reset_index()

    # Grouped bar chart
    labels = aucdf["method"].tolist()
    x = np.arange(len(labels))
    w = 0.28

    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.bar(x - w, aucdf["AUC_MISF_mean"], yerr=aucdf["AUC_MISF_std"],
                width=w, capsize=3, label="AUC_MISF (Finder ↑)")
    b2 = ax.bar(x,       aucdf["AUC_MIST_mean"], yerr=aucdf["AUC_MIST_std"],
                width=w, capsize=3, label="AUC_MIST (Timing ↑)")
    b3 = ax.bar(x + w,   aucdf["AUC_BG_mean"],   yerr=aucdf["AUC_BG_std"],
                width=w, capsize=3, label="AUC_BG (Outside ↓)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Area under structural localization curve")
    ax.set_title("Zero-shot structural AUCs (higher is better except AUC_BG)")
    ax.legend(ncol=3, frameon=False)
    plt.tight_layout()
    fig.savefig(OUTDIR / "zero_shot_AUCs.png", dpi=300)
    plt.close(fig)

    # Console caption (useful as paper figure caption)
    # If all methods have the same n, show it; otherwise show a range.
    n_vals = sorted(aucdf["n"].unique())
    if len(n_vals) == 1:
        n_str = f"{n_vals[0]}"
    else:
        n_str = f"{n_vals[0]}–{n_vals[-1]}"

    caption = (
        "Zero-shot structural AUCs by CAM (mean±std over "
        f"{n_str} QR images per method). "
        "Blue=AUC_MISF (mass inside finder patterns, ↑ better), "
        "Orange=AUC_MIST (mass inside timing patterns, ↑ better), "
        "Green=AUC_BG (mass outside QR region, ↓ lower is better)."
    )
    print("[FIG] Saved", OUTDIR / "zero_shot_AUCs.png")
    print("[CAPTION]", caption)


if __name__ == "__main__":
    main()
