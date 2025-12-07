#!/usr/bin/env python3
"""
qr_ft_compare.py
Benchmark CAM methods (EigenGrad-CAM, LayerCAM, XGrad-CAM) under:
  - Zero-shot (frozen backbone + linear head)
  - FT-Struct (layer4+fc finetune, LR=1e-3, 2 epochs)
  - FT-LeakMin (layer4+fc finetune, LR=3e-4, 4 epochs)
Saves metrics, timing, and 3-4 qualitative figures.

Outputs:
  /home/ritabrata/qr_stuff/outputs_ft/*.csv
  /home/ritabrata/qr_stuff/figs/*.png
"""

import os, random, math, copy, string, time
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

import qrcode
from pytorch_grad_cam import XGradCAM, LayerCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ----------------- Repro & paths -----------------
SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"[INFO] Using device: {device}")

ROOT = Path("/home/ritabrata/qr_stuff")
OUT_DIR = ROOT / "outputs_ft"; OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "figs"; FIG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- QR helpers -----------------
def version_from_mat(mat: np.ndarray) -> int:
    N = mat.shape[0]
    return int((N - 21)//4 + 1)

def make_qr_matrix_autofit(data: str, error: str='M') -> np.ndarray:
    qr = qrcode.QRCode(version=None,
                       error_correction=getattr(qrcode.constants, f'ERROR_CORRECT_{error}'),
                       box_size=1, border=0)
    qr.add_data(data); qr.make(fit=True)
    return np.array(qr.get_matrix(), dtype=np.uint8)

def random_qr_text(length: int) -> str:
    alphabet = string.ascii_uppercase + string.digits + " $%*+-./:"
    return "".join(random.choice(alphabet) for _ in range(length))

def masks_from_matrix(mat):
    N = mat.shape[0]
    M_finder = np.zeros_like(mat, dtype=np.uint8)
    for (r,c) in [(0,0),(0,N-7),(N-7,0)]:
        M_finder[r:r+7, c:c+7] = 1
    M_timing = np.zeros_like(mat, dtype=np.uint8)
    M_timing[6,:] = 1; M_timing[:,6] = 1
    for (r,c) in [(0,0),(0,N-7),(N-7,0)]:
        M_timing[r:r+7, c:c+7] = 0
    M_qrbox = np.ones_like(mat, dtype=np.uint8)
    return M_finder, M_timing, M_qrbox

def render_from_modules(mat, scale=10, bg=255):
    img = np.kron(1-mat, np.ones((scale,scale), dtype=np.uint8)) * 255
    if bg != 255:
        mask_white = (img==255)
        img = img.copy()
        img[mask_white] = bg
    return img

def expand_mask_to_scale(mask_bin: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask_bin.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8))

def warp_affine_mask(mask_bin, M, W, H):
    warped = cv2.warpAffine(mask_bin.astype(np.uint8)*255, M, (W,H),
                            flags=cv2.INTER_NEAREST, borderValue=0)
    return (warped>127).astype(np.uint8)

def warp_perspective_mask(mask_bin, Hmat, W, H):
    warped = cv2.warpPerspective(mask_bin.astype(np.uint8)*255, Hmat, (W,H),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
    return (warped>127).astype(np.uint8)

def random_perspective(src_size, max_perturb=0.10):
    W = H = src_size
    src = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    dst = src.copy()
    for i in range(4):
        dx = np.random.uniform(-max_perturb, max_perturb)*W
        dy = np.random.uniform(-max_perturb, max_perturb)*H
        dst[i] = (src[i][0]+dx, src[i][1]+dy)
    Hmat = cv2.getPerspectiveTransform(src, dst)
    return Hmat

def apply_distortions(img, masks, meta,
                      rotate_deg=0, blur_ksize=0, contrast=1.0,
                      brightness_shift=0, occlude=False, motion=False,
                      perspective=False):
    H, W = img.shape[:2]
    out = img.astype(np.float32)
    out = np.clip(out*contrast + brightness_shift, 0, 255)

    if rotate_deg != 0:
        M = cv2.getRotationMatrix2D((W/2, H/2), rotate_deg, 1.0)
        out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=255)
        masks = [warp_affine_mask(m, M, W, H) for m in masks]
        meta["rotate_deg"] = rotate_deg

    if perspective:
        Hmat = random_perspective(W, max_perturb=0.12)
        out  = cv2.warpPerspective(out, Hmat, (W, H), flags=cv2.INTER_NEAREST, borderValue=255)
        masks = [warp_perspective_mask(m, Hmat, W, H) for m in masks]
        meta["perspective"] = True

    if blur_ksize > 0:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        out = cv2.GaussianBlur(out, (k,k), 0)
        meta["blur"] = blur_ksize

    if motion:
        k = random.choice([5,7,9])
        kernel = np.zeros((k,k), dtype=np.float32); kernel[k//2, :] = 1.0/k
        out = cv2.filter2D(out, -1, kernel)
        meta["motion"] = True

    if occlude:
        x0 = np.random.randint(0, W//2); y0 = np.random.randint(0, H//2)
        s  = np.random.randint(H//10, H//4)
        out[y0:y0+s, x0:x0+s] = 255
        meta["occlude"] = True

    return out.astype(np.uint8), masks, meta

def generate_qr_rich_with_meta(max_version=6, scale_choices=(8,10,12)):
    # build a reasonably small version by retries
    for _ in range(12):
        ecc  = random.choice(list("LMQH"))
        L    = random.randint(6, 16)
        text = random_qr_text(L)
        mat  = make_qr_matrix_autofit(text, error=ecc)
        v    = version_from_mat(mat)
        if v <= max_version: break
        for L2 in [10,8,6,4]:
            text = random_qr_text(L2)
            mat  = make_qr_matrix_autofit(text, error=ecc)
            v    = version_from_mat(mat)
            if v <= max_version: break
        if v <= max_version: break
        mat = make_qr_matrix_autofit(random_qr_text(4), error='L')
        v   = version_from_mat(mat)
        if v <= max_version: break

    Mf_mod, Mt_mod, Mb_mod = masks_from_matrix(mat)
    scale = random.choice(scale_choices)
    bg    = random.randint(230,255)
    base  = render_from_modules(mat, scale=scale, bg=bg)  # (H,W)=(N*scale, N*scale)

    # expand masks to pixel space BEFORE warps
    Mf = expand_mask_to_scale(Mf_mod, scale)
    Mt = expand_mask_to_scale(Mt_mod, scale)
    Mb = expand_mask_to_scale(Mb_mod, scale)

    # distortions
    meta = dict(version=version_from_mat(mat), ecc=None, scale=scale,
                rotate_deg=0, blur=0, motion=False, perspective=False, occlude=False)
    rot  = random.choice([0,5,-5,10,-10])
    blur = random.choice([0,1,2,3])
    ctr  = random.choice([0.85,1.0,1.15])
    br   = random.choice([-15,-5,0,5,15])
    occ  = random.random() < 0.25
    mot  = random.random() < 0.20
    persp= random.random() < 0.30
    img, (Mf, Mt, Mb), meta = apply_distortions(base, [Mf, Mt, Mb], meta,
                                                rotate_deg=rot, blur_ksize=blur,
                                                contrast=ctr, brightness_shift=br,
                                                occlude=occ, motion=mot, perspective=persp)
    Mf=(Mf>0).astype(np.uint8); Mt=(Mt>0).astype(np.uint8); Mb=(Mb>0).astype(np.uint8)
    return img, (Mf, Mt, Mb), meta

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

# ----------------- Dataset & model -----------------
pre_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225))
])

class QRDataset(Dataset):
    def __init__(self, images, labels):
        self.images, self.labels = images, labels
    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = self.images[i]
        if img.ndim==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return pre_tf(img), torch.tensor(self.labels[i], dtype=torch.long)

def build_splits(n_train_qr=600, n_train_neg=600, n_test_qr=300, n_test_neg=300):
    # TRAIN
    tr_qr, tr_masks, tr_meta = [], [], []
    for _ in tqdm(range(n_train_qr), desc="Train QR", ncols=100):
        img, masks, meta = generate_qr_rich_with_meta()
        tr_qr.append(img); tr_masks.append(masks); tr_meta.append(meta)
    tr_neg = []
    for _ in tqdm(range(n_train_neg), desc="Train Non-QR", ncols=100):
        tr_neg.append(make_negative())

    tr_imgs  = tr_qr + tr_neg
    tr_lbls  = [1]*len(tr_qr) + [0]*len(tr_neg)

    # TEST
    te_qr, te_masks, te_meta = [], [], []
    for _ in tqdm(range(n_test_qr), desc="Test QR", ncols=100):
        img, masks, meta = generate_qr_rich_with_meta()
        te_qr.append(img); te_masks.append(masks); te_meta.append(meta)
    te_neg = []
    for _ in tqdm(range(n_test_neg), desc="Test Non-QR", ncols=100):
        te_neg.append(make_negative())

    te_imgs = te_qr + te_neg
    te_lbls = [1]*len(te_qr) + [0]*len(te_neg)

    return (tr_imgs, tr_lbls), (te_imgs, te_lbls), (te_qr, te_masks, te_meta)

def build_resnet50(num_classes=2):
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    m = torchvision.models.resnet50(weights=w)
    for p in m.parameters(): p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(device)

def train_head(model, train_dl, epochs=2, lr=1e-3):
    opt = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train(); tot=0; corr=0; loss_sum=0.0
        pbar = tqdm(train_dl, desc=f"[train] epoch {ep}/{epochs}", ncols=100)
        for xb,yb in pbar:
            xb=xb.to(device, non_blocking=True); yb=yb.to(device, non_blocking=True)
            opt.zero_grad(); logits=model(xb); loss=ce(logits,yb); loss.backward(); opt.step()
            tot+= yb.size(0); corr += (logits.argmax(1)==yb).sum().item(); loss_sum += loss.item()*yb.size(0)
            pbar.set_postfix(acc=f"{corr/tot:.3f}", loss=f"{loss_sum/tot:.4f}")
    model.eval(); return model

def finetune_last_block(model, train_dl, lr=1e-3, epochs=2):
    # unfreeze layer4 + fc
    for p in model.layer4.parameters(): p.requires_grad_(True)
    for p in model.fc.parameters(): p.requires_grad_(True)
    params = list(model.layer4.parameters()) + list(model.fc.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train(); tot=0; corr=0; loss_sum=0.0
        pbar = tqdm(train_dl, desc=f"[finetune] epoch {ep}/{epochs} (lr={lr})", ncols=100)
        for xb,yb in pbar:
            xb=xb.to(device, non_blocking=True); yb=yb.to(device, non_blocking=True)
            opt.zero_grad(); logits=model(xb); loss=ce(logits,yb); loss.backward(); opt.step()
            tot+= yb.size(0); corr += (logits.argmax(1)==yb).sum().item(); loss_sum += loss.item()*yb.size(0)
            pbar.set_postfix(acc=f"{corr/tot:.3f}", loss=f"{loss_sum/tot:.4f}")
    model.eval(); return model

# ----------------- CAM & metrics -----------------
def find_last_conv_layer(model: nn.Module) -> nn.Module:
    last_conv=None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv=m
    if last_conv is None:
        raise RuntimeError("No Conv2d for CAM.")
    return last_conv

def get_cam(model, img_gray_u8, cam_cls, target_layer):
    rgb = cv2.cvtColor(img_gray_u8, cv2.COLOR_GRAY2RGB)
    x = pre_tf(rgb).unsqueeze(0).to(device)
    x.requires_grad_(True)
    logits = model(x)
    pred = int(logits.argmax(1))
    # enable grads in target layer
    for p in target_layer.parameters(): p.requires_grad_(True)
    with cam_cls(model=model, target_layers=[target_layer]) as cam:
        model.zero_grad(set_to_none=True)
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]  # (224,224) float [0,1]
    return heat

def resize_to(arr, hw):
    H,W = hw
    return cv2.resize(arr, (W,H), interpolation=cv2.INTER_CUBIC)

def _resize_mask_if_needed(mask: np.ndarray, hw: Tuple[int,int]) -> np.ndarray:
    H,W = hw
    if mask.shape != (H,W):
        mask = cv2.resize(mask.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST)
    return (mask>0).astype(np.uint8)

def FMR_TMR_BL(H01, Mf, Mt, Mb):
    H,W = H01.shape[:2]
    Mf=_resize_mask_if_needed(Mf,(H,W)); Mt=_resize_mask_if_needed(Mt,(H,W)); Mb=_resize_mask_if_needed(Mb,(H,W))
    H01 = H01.astype(np.float32); eps=1e-6; S=float(H01.sum())+eps
    FMR=float((H01* Mf).sum()/S); TMR=float((H01* Mt).sum()/S); BL=float((H01*(1-Mb)).sum()/S)
    return FMR,TMR,BL

def SLC_AUCs(H01, Mf, Mt, Mb, nT=31):
    H,W = H01.shape[:2]
    Mf=_resize_mask_if_needed(Mf,(H,W)); Mt=_resize_mask_if_needed(Mt,(H,W)); Mb=_resize_mask_if_needed(Mb,(H,W))
    vals = H01.astype(np.float32).ravel()
    taus = np.quantile(vals, np.linspace(0.0, 0.99, nT))
    eps=1e-6
    insideF=[]; insideT=[]; outside=[]
    for t in taus:
        mask = (H01>=t).astype(np.float32)
        mass = mask.sum()+eps
        insideF.append(float((mask* Mf).sum()/mass))
        insideT.append(float((mask* Mt).sum()/mass))
        outside.append(float((mask*(1-Mb)).sum()/mass))
    return float(np.mean(insideF)), float(np.mean(insideT)), float(np.mean(outside))

def distance_to_structure(H01, Mf, Mt):
    H,W = H01.shape[:2]
    Mf=_resize_mask_if_needed(Mf,(H,W)); Mt=_resize_mask_if_needed(Mt,(H,W))
    M = np.clip(Mf+Mt,0,1).astype(np.uint8)
    inv = (1-M).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    norm = math.sqrt(H*H+W*W)+1e-6
    H01 = H01.astype(np.float32)
    return float((H01*dist).sum()/(H01.sum()+1e-6))/norm

def structure_score(FMR, TMR, BL):
    return float(0.5*(FMR+TMR) - BL)

# ----------------- Qualitative overlays -----------------
def overlay_heatmap(img_gray, H01, Mf, Mt, title=None):
    H,W = img_gray.shape[:2]
    Hc = (255*H01/ max(H01.max(),1e-6)).astype(np.uint8)
    Hc = cv2.applyColorMap(Hc, cv2.COLORMAP_JET)
    base = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    out  = cv2.addWeighted(base, 0.6, Hc, 0.4, 0)

    # draw finder/timing outlines
    Mf=_resize_mask_if_needed(Mf,(H,W)); Mt=_resize_mask_if_needed(Mt,(H,W))
    cnts,_ = cv2.findContours((Mf*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0,255,0), 2)  # green finder
    cnts,_ = cv2.findContours((Mt*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255,255,0), 1)  # yellow timing

    if title:
        cv2.putText(out, title, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def save_triptych(example_name, img_gray, masks, heatmaps_dict_rows):
    """
    heatmaps_dict_rows: list of 2 rows (ZS, FT), each is dict:
         {"EigenGrad-CAM": H01, "LayerCAM": H01, "XGrad-CAM": H01}
    """
    Mf,Mt,Mb = masks
    panels=[]
    for row_title, hdict in heatmaps_dict_rows:
        row=[]
        for col_title in ["EigenGrad-CAM","LayerCAM","XGrad-CAM"]:
            H01 = hdict[col_title]
            panel = overlay_heatmap(img_gray, H01, Mf, Mt, title=f"{row_title} | {col_title}")
            row.append(panel)
        panels.append(np.hstack(row))
    canvas = np.vstack(panels)
    out_path = FIG_DIR / f"{example_name}.png"
    cv2.imwrite(str(out_path), canvas[:, :, ::-1])  # BGR->RGB handled above
    print(f"[FIG] Saved {out_path}")

# ----------------- Evaluation loops -----------------
def evaluate_methods(model, methods, qr_imgs, qr_masks, tag):
    target_layer = find_last_conv_layer(model)
    for p in target_layer.parameters(): 
        p.requires_grad_(True)

    rows = []

    # Warmup one call to stabilize timing
    _ = get_cam(model, qr_imgs[0], list(methods.values())[0], target_layer)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for name, CAMCls in methods.items():
        t0 = time.time()
        for i, img in enumerate(tqdm(qr_imgs, desc=f"{tag} {name}", ncols=100)):
            Mf, Mt, Mb = qr_masks[i]
            H = get_cam(model, img, CAMCls, target_layer)
            H = resize_to(H, img.shape[:2])

            FMR, TMR, BL = FMR_TMR_BL(H, Mf, Mt, Mb)
            aF, aT, aBG   = SLC_AUCs(H, Mf, Mt, Mb)
            DtS           = distance_to_structure(H, Mf, Mt)

            rows.append(dict(
                tag=tag, method=name, idx=i,
                FMR=FMR, TMR=TMR, BL=BL,
                AUC_MISF=aF, AUC_MIST=aT, AUC_BG=aBG,
                DtS=DtS
            ))
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        ms_per_img = 1000.0 * elapsed / len(qr_imgs)
        # attach timing to all rows for this method
        for r in rows:
            if r["method"] == name:
                r["ms_per_img"] = ms_per_img

    df = pd.DataFrame(rows)
    df["StructureScore"] = 0.5 * (df["FMR"] + df["TMR"]) - df["BL"]

    out_csv = OUT_DIR / f"metrics_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")

    # ---- numeric-only summary (prevents the pandas error) ----
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # keep only the metrics we care to display (ordered)
    want = [c for c in [
        "FMR","TMR","BL","AUC_MISF","AUC_MIST","AUC_BG",
        "DtS","StructureScore","ms_per_img"
    ] if c in num_cols]

    sm = (df.groupby("method")[want]
            .mean()
            .reset_index())

    # Pretty print sorted by StructureScore desc (higher is better)
    if "StructureScore" in sm.columns:
        sm_disp = sm.copy()
        sm_disp[want] = sm_disp[want].astype(float).round(6)
        sm_disp = sm_disp.sort_values("StructureScore", ascending=False)
    else:
        sm_disp = sm

    out_sum = OUT_DIR / f"summary_{tag}.csv"
    sm.to_csv(out_sum, index=False)
    print(f"[SUMMARY] {tag} (mean over {len(qr_imgs)} QR imgs)")
    print(sm_disp.to_string(index=False))

    return df


def evaluate_accuracy(model, test_imgs, test_lbls, batch_size=256):
    dl = DataLoader(QRDataset(test_imgs, test_lbls), batch_size=batch_size,
                    shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))
    model.eval(); tot=0; corr=0
    for xb,yb in tqdm(dl, desc="[eval] accuracy", ncols=100):
        xb=xb.to(device, non_blocking=True); yb=yb.to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(xb)
            pred   = logits.argmax(1)
        tot += yb.size(0); corr += (pred==yb).sum().item()
    acc = corr/tot
    print(f"[RESULT] Accuracy: {acc:.4f}")
    return acc

# ----------------- Example selection -----------------
def pick_examples(df_zs, qr_imgs, qr_masks, qr_meta):
    # Ex1: Clean success (no persp, no occl, blur<=1), high (FMR+TMR) for EigenGrad-CAM, low BL
    clean_idxs = [i for i,m in enumerate(qr_meta)
                  if (not m.get("perspective", False)) and (not m.get("occlude", False)) and (m.get("blur",0) <= 1)]
    zs_eigen = df_zs[df_zs["method"]=="EigenGrad-CAM"]
    zs_eigen = zs_eigen[zs_eigen["idx"].isin(clean_idxs)].copy()
    zs_eigen["score"] = zs_eigen["FMR"]+zs_eigen["TMR"] - zs_eigen["BL"]
    ex1_idx = int(zs_eigen.sort_values("score", ascending=False)["idx"].head(1).values[0]) if len(zs_eigen)>0 else 0

    # Ex2: Leaky case for XGrad-CAM (max BL)
    zs_x = df_zs[df_zs["method"]=="XGrad-CAM"].copy()
    ex2_idx = int(zs_x.sort_values("BL", ascending=False)["idx"].head(1).values[0])

    # Ex3: Perspective case
    persp_idxs = [i for i,m in enumerate(qr_meta) if m.get("perspective", False)]
    ex3_idx = persp_idxs[0] if len(persp_idxs)>0 else 1

    # Ex4: Occlusion or heavy blur
    hard_idxs = [i for i,m in enumerate(qr_meta) if (m.get("occlude", False) or m.get("blur",0)>=3)]
    ex4_idx = hard_idxs[0] if len(hard_idxs)>0 else 2

    return [ex1_idx, ex2_idx, ex3_idx, ex4_idx]

# ----------------- Main -----------------
def main():
    # 1) Build splits
    print("[STEP] Building train/test splits…")
    (tr_imgs, tr_lbls), (te_imgs, te_lbls), (te_qr, te_masks, te_meta) = build_splits()

    # 2) Train linear head (ZS backbone)
    print("[STEP] Training linear head on frozen ResNet-50…")
    base = build_resnet50()
    tr_dl = DataLoader(QRDataset(tr_imgs, tr_lbls), batch_size=128, shuffle=True,
                       num_workers=2, pin_memory=(device.type=="cuda"))
    base = train_head(base, tr_dl, epochs=2, lr=1e-3)

    # 3) Accuracy check
    _ = evaluate_accuracy(base, te_imgs, te_lbls)

    # 4) Methods dict
    methods = {
        "EigenGrad-CAM": EigenGradCAM,
        "LayerCAM":      LayerCAM,
        "XGrad-CAM":     XGradCAM,
    }

    # 5) ZS evaluation (QR-only structure metrics)
    print("\n[STEP] Evaluating CAMs (Zero-shot)…")
    df_zs = evaluate_methods(base, methods, te_qr, te_masks, tag="ZS")

    # 6) Finetune: FT-Struct (lr=1e-3, 2 ep)
    ft_struct = copy.deepcopy(base).to(device)
    print("\n[STEP] Finetune last block (FT-Struct)…")
    ft_struct = finetune_last_block(ft_struct, tr_dl, lr=1e-3, epochs=2)
    print("[STEP] Evaluating CAMs (FT-Struct)…")
    df_fs = evaluate_methods(ft_struct, methods, te_qr, te_masks, tag="FT_STRUCT")

    # 7) Finetune: FT-LeakMin (lr=3e-4, 4 ep)
    ft_leak = copy.deepcopy(base).to(device)
    print("\n[STEP] Finetune last block (FT-LeakMin)…")
    ft_leak = finetune_last_block(ft_leak, tr_dl, lr=3e-4, epochs=4)
    print("[STEP] Evaluating CAMs (FT-LeakMin)…")
    df_fl = evaluate_methods(ft_leak, methods, te_qr, te_masks, tag="FT_LEAKMIN")

    # 8) Merge summary
    all_df = pd.concat([df_zs, df_fs, df_fl], ignore_index=True)
    all_df.to_csv(OUT_DIR/"metrics_all.csv", index=False)
    print(f"[SAVE] {OUT_DIR/'metrics_all.csv'}")

    # 9) Pick qualitative examples
    ex_ids = pick_examples(df_zs, te_qr, te_masks, te_meta)
    print("[INFO] Qual examples idx:", ex_ids)

    # 10) Make triptychs: ZS row vs FT-Struct row
    for tag, model_ref in [("ZS", base), ("FT_STRUCT", ft_struct)]:
        pass  # heatmaps computed again in the loop below for clarity

    # Precompute CAMs for examples for both regimes
    def cam_set(model, img, tl):
        return {
            "EigenGrad-CAM": resize_to(get_cam(model, img, EigenGradCAM, tl), img.shape[:2]),
            "LayerCAM":      resize_to(get_cam(model, img, LayerCAM, tl),      img.shape[:2]),
            "XGrad-CAM":     resize_to(get_cam(model, img, XGradCAM, tl),      img.shape[:2]),
        }

    # Save 4 qualitative panels
    for k, idx in enumerate(ex_ids, 1):
        img = te_qr[idx]; masks = te_masks[idx]
        tl0 = find_last_conv_layer(base);      [p.requires_grad_(True) for p in tl0.parameters()]
        tl1 = find_last_conv_layer(ft_struct); [p.requires_grad_(True) for p in tl1.parameters()]

        cams_zs = cam_set(base, img, tl0)
        cams_fs = cam_set(ft_struct, img, tl1)
        save_triptych(f"qual_ex{k:02d}_idx{idx}", img, masks,
                      [("Zero-shot", cams_zs), ("FT-Struct", cams_fs)])

    print("\n[DONE] Metrics in outputs_ft/, figures in figs/")

if __name__ == "__main__":
    main()
