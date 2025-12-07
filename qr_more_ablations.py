# /home/ritabrata/qr_stuff/qr_more_ablations.py
# ============================================================
# Extra ablations for "Structural XAI for QR Classification"
# (Faithfulness + Robustness) — paper-ready outputs
#
# New in this version:
#  - FIX: all mask ops auto-align (no shape mismatches)
#  - BATCHED CAM generation (huge speedup, high GPU use)
#  - BATCHED faithfulness (vectorized steps per image)
#  - Precompute & reuse tensors (fewer CPU↔GPU hops)
#  - Accurate runtime/memory (GPU sync)
#
# Outputs:
#  - CSVs & PNGs under /home/ritabrata/qr_stuff/outputs_more and figs_more
#  - Loads prior metrics from outputs_ft/ if present (no re-computation)
#
# Regimes:
#  - ZS (head-only), FT_STRUCT (last block FT, 2ep @ 1e-3),
#    FT_LEAKMIN (last block FT, 4ep @ 3e-4)
# Methods (fast default): eigengradcam, layercam, xgradcam
# Toggle heavy methods by setting HEAVY_METHODS=True.
# ============================================================

import os, random, math, time, copy, string, json
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
from tqdm import tqdm
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim

# ---- Optional CAMs (import what exists; skip missing) ----
AVAILABLE_CAMS = {}
try:
    from pytorch_grad_cam import (GradCAM, GradCAMPlusPlus, XGradCAM, LayerCAM,
                                  EigenCAM, EigenGradCAM, ScoreCAM, AblationCAM)
    AVAILABLE_CAMS.update({
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "xgradcam": XGradCAM,
        "layercam": LayerCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "scorecam": ScoreCAM,
        "ablationcam": AblationCAM,
    })
except Exception as e:
    print("[WARN] pytorch_grad_cam subset only:", e)

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ----------------- Paths & toggles -----------------
BASE = Path("/home/ritabrata/qr_stuff")
OUT = BASE / "outputs_more"
FIG = BASE / "figs_more"
CKPT = BASE / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

# Repro & device
SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"[INFO] Using device: {device}")

# Methods to run (fast by default)
METHODS = ["eigengradcam","layercam","xgradcam"]
HEAVY_METHODS = False  # True → add gradcam, gradcam++, ablationcam, scorecam, eigencam

if HEAVY_METHODS:
    for m in ["gradcam","gradcam++","ablationcam","scorecam","eigencam"]:
        if m in AVAILABLE_CAMS:
            METHODS.append(m)

# Regimes to use
REGIMES = ["ZS","FT_STRUCT","FT_LEAKMIN"]

# Eval sizes
N_EVAL  = 200          # images per eval for most ablations
N_FAITH = 100          # fewer for insertion/deletion to keep runtime reasonable

# Big GPU utilization controls (adjust to ~16–20 GB VRAM)
BATCH_CAM   = 48   # batched CAM images at once (increase to use more VRAM)
BATCH_STEPS = 16   # batched faithfulness steps per forward pass

# ----------------- Dataset generation (rich) -----------------
import qrcode

def make_qr_matrix_autofit(data: str, error: str = 'M') -> np.ndarray:
    qr = qrcode.QRCode(version=None,
                       error_correction=getattr(qrcode.constants, f'ERROR_CORRECT_{error}'),
                       box_size=1, border=0)
    qr.add_data(data); qr.make(fit=True)
    return np.array(qr.get_matrix(), dtype=np.uint8)  # 1=black

def version_from_mat(mat: np.ndarray) -> int:
    N = mat.shape[0]
    return int((N-21)//4 + 1)

def random_qr_text(L=8) -> str:
    alphabet = string.ascii_uppercase + string.digits + " $%*+-./:"
    return "".join(random.choice(alphabet) for _ in range(L))

def masks_from_matrix(mat):
    N = mat.shape[0]
    M_finder = np.zeros_like(mat, dtype=np.uint8)
    for (r,c) in [(0,0),(0,N-7),(N-7,0)]: M_finder[r:r+7, c:c+7] = 1
    M_timing = np.zeros_like(mat, dtype=np.uint8); M_timing[6,:]=1; M_timing[:,6]=1
    for (r,c) in [(0,0),(0,N-7),(N-7,0)]: M_timing[r:r+7, c:c+7] = 0
    M_qrbox = np.ones_like(mat, dtype=np.uint8)
    return M_finder, M_timing, M_qrbox

def render_from_modules(mat, scale=10, bg=255):
    img = np.kron(1-mat, np.ones((scale,scale), dtype=np.uint8)) * 255
    if bg != 255:
        mask_white = (img==255)
        img = img.copy(); img[mask_white] = bg
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

def apply_distortions(img, masks, rotate_deg=0, blur_ksize=0, contrast=1.0,
                      brightness_shift=0, occlude=False, motion=False, perspective=False):
    H, W = img.shape[:2]
    out = img.astype(np.float32)
    out = np.clip(out*contrast + brightness_shift, 0, 255)

    # rotation
    if rotate_deg != 0:
        M = cv2.getRotationMatrix2D((W/2, H/2), rotate_deg, 1.0)
        out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=255)
        masks = [warp_affine_mask(m, M, W, H) for m in masks]
    # perspective
    if perspective:
        Hmat = random_perspective(W, max_perturb=0.10)
        out = cv2.warpPerspective(out, Hmat, (W,H), flags=cv2.INTER_NEAREST, borderValue=255)
        masks = [warp_perspective_mask(m, Hmat, W, H) for m in masks]
    # blur
    if blur_ksize>0:
        k = blur_ksize if blur_ksize%2==1 else blur_ksize+1
        out = cv2.GaussianBlur(out, (k,k), 0)
    # motion
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

def generate_qr_rich(max_version=6, scale_choices=(8,10,12)):
    # keep version ≤ 6 (auto-fit)
    for _ in range(12):
        ecc = random.choice(list("LMQH"))
        for L in [12,10,8,6,4]:
            text = random_qr_text(L)
            mat = make_qr_matrix_autofit(text, error=ecc)
            if version_from_mat(mat) <= max_version: break
        if version_from_mat(mat) <= max_version: break
    Mf, Mt, Mb = masks_from_matrix(mat)
    bg = random.randint(230,255)
    scale = random.choice(scale_choices)
    base = render_from_modules(mat, scale=scale, bg=bg)
    rot  = random.choice([0,5,-5,10,-10])
    blur = random.choice([0,1,2,3])
    ctr  = random.choice([0.85,1.0,1.15])
    br   = random.choice([-15,-5,0,5,15])
    occ  = random.random()<0.25
    mot  = random.random()<0.20
    persp= random.random()<0.30
    img, masks = apply_distortions(base, [Mf,Mt,Mb], rotate_deg=rot,
                                   blur_ksize=blur, contrast=ctr,
                                   brightness_shift=br, occlude=occ,
                                   motion=mot, perspective=persp)
    return img, masks

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

# -------------- Data & transforms --------------
def build_eval_set(n_qr=200, n_neg=200):
    imgs, masks = [], []
    for _ in tqdm(range(n_qr), desc="Eval QR", ncols=100):
        i,m = generate_qr_rich(max_version=6, scale_choices=(8,10,12))
        imgs.append(i); masks.append(m)
    negs=[]
    for _ in tqdm(range(n_neg), desc="Eval Non-QR", ncols=100):
        negs.append(make_negative())
    return imgs, masks, negs

def tf_compose(size=224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),
                             std=(0.229,0.224,0.225))
    ])

# -------------- Model & regimes --------------
def build_resnet50(num_classes=2):
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    m = torchvision.models.resnet50(weights=w)
    for p in m.parameters(): p.requires_grad_(False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(device)

def train_head(model, train_dl, epochs=2, lr=1e-3):
    opt = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    for ep in range(1, epochs+1):
        model.train(); total=correct=loss_sum=0.0
        pbar = tqdm(train_dl, desc=f"[train] ep {ep}/{epochs}", ncols=100)
        for xb,yb in pbar:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            opt.zero_grad(); logits=model(xb); loss=ce(logits,yb)
            loss.backward(); opt.step()
            total+=yb.size(0); correct+=(logits.argmax(1)==yb).sum().item()
            loss_sum+=loss.item()*yb.size(0)
            pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.3f}")
    model.eval(); return model

def finetune_last_block(model, train_dl, lr=1e-3, epochs=2):
    # unfreeze layer4 + head
    for p in model.layer4.parameters(): p.requires_grad_(True)
    for p in model.fc.parameters(): p.requires_grad_(True)
    params = list(model.layer4.parameters())+list(model.fc.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    for ep in range(1,epochs+1):
        model.train(); total=correct=loss_sum=0.0
        pbar = tqdm(train_dl, desc=f"[finetune] ep {ep}/{epochs} (lr={lr})", ncols=100)
        for xb,yb in pbar:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            opt.zero_grad(); logits=model(xb); loss=ce(logits,yb)
            loss.backward(); opt.step()
            total+=yb.size(0); correct+=(logits.argmax(1)==yb).sum().item()
            loss_sum+=loss.item()*yb.size(0)
            pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.3f}")
    model.eval(); return model

def make_loader(images, labels, bs=256, size=224):  # larger bs for speed
    pre_tf = tf_compose(size)
    class DS(Dataset):
        def __len__(self): return len(images)
        def __getitem__(self, i):
            img = images[i]
            if img.ndim==2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return pre_tf(img), torch.tensor(labels[i],dtype=torch.long)
    return DataLoader(DS(), batch_size=bs, shuffle=True,
                      num_workers=2, pin_memory=(device.type=="cuda"))

# -------------- CAM utils --------------
def find_last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d): last = m
    if last is None: raise RuntimeError("No Conv2d found for CAM")
    return last

def preprocess_numpy_list(imgs: List[np.ndarray], size=224) -> torch.Tensor:
    """Convert list of grayscale np.uint8 images → (B,3,H,W) normalized tensor."""
    pre_tf = tf_compose(size)
    batch = []
    for img in imgs:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim==2 else img
        batch.append(pre_tf(rgb))
    return torch.stack(batch, dim=0)

def get_cam_maps_batch(model, imgs_list, method="eigengradcam", size=224,
                       target_layer=None, scorecam_batch=None) -> List[np.ndarray]:
    """Compute CAMs for a list of images as one batched pass."""
    if method not in AVAILABLE_CAMS:
        raise ValueError(f"Method {method} not available.")
    CAMCls = AVAILABLE_CAMS[method]
    if target_layer is None:
        target_layer = find_last_conv(model)

    x = preprocess_numpy_list(imgs_list, size).to(device)
    x.requires_grad_(True)
    logits = model(x)
    preds = logits.argmax(1).tolist()
    targets = [ClassifierOutputTarget(int(p)) for p in preds]
    extra = {}
    if method == "scorecam" and (scorecam_batch is not None):
        extra["batch_size"] = int(scorecam_batch)

    with CAMCls(model=model, target_layers=[target_layer], **extra) as cam:
        model.zero_grad(set_to_none=True)
        grayscale_cam = cam(input_tensor=x, targets=targets)  # (B,H,W)
        cams = [grayscale_cam[i] for i in range(grayscale_cam.shape[0])]
    return cams

# -------------- Metrics --------------
def resize_to(arr, hw):
    H,W = hw
    return cv2.resize(arr, (W,H), interpolation=cv2.INTER_CUBIC)

def ensure_mask_size(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Resize a binary mask to (H,W) using nearest-neighbor, then binarize.
    Accepts {0,1} or [0..1] masks.
    """
    m = mask.astype(np.uint8)
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)

def FMR_TMR_BL(cam01: np.ndarray, Mf: np.ndarray, Mt: np.ndarray, Mb: np.ndarray):
    """
    Structural metrics with automatic mask alignment.
    cam01: float array (H,W) ideally in [0,1]
    Mf, Mt, Mb: binary masks at ANY resolution; will be resized to (H,W)
    Returns: FMR, TMR, BL
    """
    cam = cam01.astype(np.float32)
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)

    H, W = cam.shape[:2]
    Mf = ensure_mask_size(Mf, H, W).astype(np.float32)
    Mt = ensure_mask_size(Mt, H, W).astype(np.float32)
    Mb = ensure_mask_size(Mb, H, W).astype(np.float32)

    eps = 1e-6
    S = cam.sum() + eps
    FMR = float((cam * Mf).sum() / S)
    TMR = float((cam * Mt).sum() / S)
    BL  = float((cam * (1.0 - Mb)).sum() / S)
    return FMR, TMR, BL

def dist_to_structure(cam01, Mb):
    from scipy.ndimage import distance_transform_edt as edt
    inv = (Mb==0).astype(np.uint8)
    d = edt(inv)  # distance to nearest QR region
    mass = cam01.sum()+1e-6
    return float((cam01*d).sum()/mass)

def auc_curve(y: np.ndarray):
    if len(y) < 2:
        return float(y.mean()) if len(y) else 0.0
    x = np.linspace(0, 1, len(y))
    return float(np.trapezoid(y, x))

# -------------- Faithfulness: insertion/deletion (batched per steps) --------------
def blur_img(img, k=21):
    k = k if k%2==1 else k+1
    return cv2.GaussianBlur(img, (k,k), 0)

def logits_for_batch(model, imgs, size=224):
    """imgs: list of grayscale np.uint8 (H,W). Returns prob/logit arrays (B,)."""
    x = preprocess_numpy_list(imgs, size).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:,1].float().cpu().numpy()
        logs  = logits[:,1].float().cpu().numpy()
    return probs, logs

def insertion_deletion_curves_batched(model, img, cam, Mb, steps=20, size=224, structure_mask=None):
    """
    Faithfulness curves, batched across steps for a single image:
      - Plain insertion/deletion (rank all pixels by CAM) OR
      - Structure-aware (restrict ranking to a provided mask).
    """
    H, W = img.shape[:2]
    cam_r = resize_to(cam, (H, W))

    # Build ranking indices
    if structure_mask is not None:
        sm = structure_mask
        if sm.shape != (H, W):
            sm = ensure_mask_size(sm, H, W)
        sm = (sm > 0).astype(np.uint8)
        idxs = np.flatnonzero(sm.ravel())  # pixels inside structure
        if len(idxs) == 0:
            idxs = np.arange(H * W, dtype=np.int64)
    else:
        idxs = np.arange(H * W, dtype=np.int64)

    scores = cam_r.ravel()[idxs]
    order = np.argsort(-scores)           # descending by CAM
    idxs = idxs[order]                    # sorted global indices to reveal/remove

    ks = (np.linspace(0, 1, steps) * len(idxs)).astype(int)
    ks[-1] = len(idxs)

    img_blur = blur_img(img, 21)
    img_ins  = img_blur.copy()
    img_del  = img.copy()

    # for batched steps, we will build a stack of images at each k chunk
    reveal = np.zeros(H * W, dtype=bool)
    probs_ins_all, probs_del_all = [], []

    # chunk steps to keep activation memory high but manageable
    CH = max(1, BATCH_STEPS)
    for start in range(0, len(ks), CH):
        end = min(len(ks), start + CH)
        imgs_ins_batch, imgs_del_batch = [], []
        reveal_batch = reveal.copy()
        for j in range(start, end):
            k = ks[j]
            if k > 0:
                reveal_batch[idxs[:k]] = True
            ry, rx = np.divmod(np.flatnonzero(reveal_batch), W)
            ins = img_blur.copy(); ins[ry, rx] = img[ry, rx]
            dele = img.copy();     dele[ry, rx] = 255
            imgs_ins_batch.append(ins)
            imgs_del_batch.append(dele)
        # run both batches
        p_ins, _ = logits_for_batch(model, imgs_ins_batch, size=size)
        p_del, _ = logits_for_batch(model, imgs_del_batch, size=size)
        probs_ins_all.extend(p_ins.tolist())
        probs_del_all.extend(p_del.tolist())
        # update reveal to last in chunk for next chunk continuity
        reveal = reveal_batch.copy()

    AUC_ins = auc_curve(np.array(probs_ins_all))
    AUC_del = auc_curve(1 - np.array(probs_del_all))
    return np.linspace(0,1,len(ks)), probs_ins_all, probs_del_all, AUC_ins, AUC_del

# -------------- Causal occlusion per region --------------
def occlude_mask(img: np.ndarray, mask: np.ndarray, val: int = 255) -> np.ndarray:
    """
    Occlude 'img' where 'mask'==1 with constant 'val'.
    Works for HxW or HxWxC; 'mask' can be any resolution.
    """
    out = img.copy()
    H, W = out.shape[:2]
    m = ensure_mask_size(mask, H, W).astype(bool)
    out[m] = val
    return out

# ----------------- Main orchestration -----------------
def load_or_train_models(train_imgs, train_labels):
    dl = make_loader(train_imgs, train_labels, bs=256, size=224)

    paths = {
        "ZS": CKPT/"resnet50_ZS.pth",  # head-only (frozen backbone; trained head)
        "FT_STRUCT": CKPT/"resnet50_FT_STRUCT.pth",
        "FT_LEAKMIN": CKPT/"resnet50_FT_LEAKMIN.pth",
    }
    models = {}

    # ZS: head-only
    if paths["ZS"].exists():
        m = build_resnet50()
        m.load_state_dict(torch.load(paths["ZS"], map_location=device))
        m.eval(); models["ZS"] = m
        print("[LOAD] ZS checkpoint")
    else:
        print("[TRAIN] ZS head (2 epochs, lr=1e-3)")
        m = build_resnet50()
        m = train_head(m, dl, epochs=2, lr=1e-3)
        torch.save(m.state_dict(), paths["ZS"]); models["ZS"] = m

    # FT_STRUCT: last-block ft (2 ep, lr=1e-3)
    if paths["FT_STRUCT"].exists():
        m = build_resnet50(); m.load_state_dict(torch.load(paths["FT_STRUCT"], map_location=device))
        m.eval(); models["FT_STRUCT"] = m
        print("[LOAD] FT_STRUCT checkpoint")
    else:
        print("[TRAIN] FT_STRUCT last-block (2 epochs, lr=1e-3)")
        m = build_resnet50(); m = train_head(m, dl, epochs=2, lr=1e-3)
        m = finetune_last_block(m, dl, lr=1e-3, epochs=2)
        torch.save(m.state_dict(), paths["FT_STRUCT"]); models["FT_STRUCT"] = m

    # FT_LEAKMIN: last-block ft (4 ep, lr=3e-4)
    if paths["FT_LEAKMIN"].exists():
        m = build_resnet50(); m.load_state_dict(torch.load(paths["FT_LEAKMIN"], map_location=device))
        m.eval(); models["FT_LEAKMIN"] = m
        print("[LOAD] FT_LEAKMIN checkpoint")
    else:
        print("[TRAIN] FT_LEAKMIN last-block (4 epochs, lr=3e-4)")
        m = build_resnet50(); m = train_head(m, dl, epochs=2, lr=1e-3)
        m = finetune_last_block(m, dl, lr=3e-4, epochs=4)
        torch.save(m.state_dict(), paths["FT_LEAKMIN"]); models["FT_LEAKMIN"] = m

    return models

def ensure_prior_metrics_index():
    prior = []
    for fn in ["metrics_ZS.csv","metrics_FT_STRUCT.csv","metrics_FT_LEAKMIN.csv","metrics_all.csv"]:
        p = BASE/"outputs_ft"/fn
        if p.exists(): prior.append({"name": fn, "path": str(p)})
    idx = pd.DataFrame(prior)
    if len(prior):
        idx.to_csv(OUT/"_PRIOR_index.csv", index=False)
        print("[INFO] Found prior metrics in outputs_ft/ (not re-run).")
    else:
        print("[INFO] No prior metrics found; proceeding with new ablations only.")

def eval_runtime_memory(model, img_list, methods, size=224, n_sample=20, regime="ZS"):
    res=[]
    target_layer = find_last_conv(model)
    torch.cuda.empty_cache()
    for method in methods:
        if method not in AVAILABLE_CAMS: continue
        # warm-up
        _ = get_cam_maps_batch(model, img_list[:min(4,len(img_list))], method, size=size, target_layer=target_layer)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0=time.perf_counter()
        _ = get_cam_maps_batch(model, img_list[:min(n_sample,len(img_list))], method,
                               size=size, target_layer=target_layer, scorecam_batch=32)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter()-t0
        ms = dt*1000.0/max(1,min(n_sample, len(img_list)))
        mem = 0.0
        if device.type=="cuda":
            mem = torch.cuda.max_memory_allocated()/1024**2
        res.append(dict(regime=regime, method=method, ms_per_img=ms, cuda_MB=mem))
    df=pd.DataFrame(res); df.to_csv(OUT/f"runtime_memory_{regime}.csv", index=False)
    return df

def run_faithfulness(models, eval_imgs, eval_masks, methods, tag="ZS", size=224, steps=20):
    """Insertion/Deletion faithfulness + structure-aware variants (batched per steps)"""
    K = min(N_FAITH, len(eval_imgs))
    chosen = list(range(0, len(eval_imgs), max(1, len(eval_imgs)//K)))[:K]
    rows = []

    for method in methods:
        if method not in AVAILABLE_CAMS:
            continue
        print(f"[FAITH] {tag} {method}")
        m = models[tag]
        target_layer = find_last_conv(m)

        # precompute CAMs for chosen images (batched)
        cams_all = []
        for start in tqdm(range(0, len(chosen), BATCH_CAM), ncols=100, desc=f"{method} {tag} (CAM batch)"):
            batch_ids = chosen[start:start+BATCH_CAM]
            imgs_b = [eval_imgs[i] for i in batch_ids]
            cams_b = get_cam_maps_batch(m, imgs_b, method, size=size, target_layer=target_layer,
                                        scorecam_batch=32)
            cams_all.extend(cams_b)

        for j, idx in enumerate(tqdm(chosen, ncols=100, desc=f"{method} {tag}")):
            img = eval_imgs[idx]
            Mf, Mt, Mb = eval_masks[idx]
            cam = cams_all[j]

            # plain insertion/deletion
            _, _, _, AUC_ins, AUC_del = insertion_deletion_curves_batched(m, img, cam, Mb, steps=steps, size=size)

            # structure-aware (restrict ranking to QR vs BG)
            _, _, _, AUC_ins_QR, AUC_del_QR = insertion_deletion_curves_batched(
                m, img, cam, Mb, steps=steps, size=size, structure_mask=Mb
            )
            _, _, _, AUC_ins_BG, AUC_del_BG = insertion_deletion_curves_batched(
                m, img, cam, Mb, steps=steps, size=size, structure_mask=(1 - Mb)
            )

            rows.append(dict(
                regime=tag, method=method, idx=idx,
                AUC_ins=AUC_ins, AUC_del=AUC_del,
                AUC_ins_QR=AUC_ins_QR, AUC_del_QR=AUC_del_QR,
                AUC_ins_BG=AUC_ins_BG, AUC_del_BG=AUC_del_BG
            ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"faithfulness_{tag}.csv", index=False)

    sm = df.groupby(["regime", "method"]).mean(numeric_only=True).reset_index()
    sm.to_csv(OUT / f"faithfulness_{tag}_summary.csv", index=False)

    # quick plot: insertion AUC (QR vs BG)
    for method in sm["method"].unique():
        sdf = sm[sm.method == method]
        if len(sdf):
            plt.figure()
            plt.bar(["AUC_ins_QR", "AUC_ins_BG"],
                    [sdf["AUC_ins_QR"].iloc[0], sdf["AUC_ins_BG"].iloc[0]])
            plt.title(f"{tag} {method}: insertion AUC (QR vs BG)")
            plt.ylabel("AUC")
            plt.savefig(FIG / f"faith_{tag}_{method}_AUCins_QR_BG.png", dpi=160)
            plt.close()

    return sm

def run_causal_occlusion(models, eval_imgs, eval_masks, methods, tag="ZS", size=224):
    """Occlude finder/timing/background; measure logit drops; correlate with FMR/TMR"""
    rows=[]
    for method in methods:
        if method not in AVAILABLE_CAMS: continue
        print(f"[CAUSAL] {tag} {method}")
        m = models[tag]; target_layer = find_last_conv(m)

        # precompute CAMs batched for speed
        cams_all = []
        for start in tqdm(range(0, min(N_EVAL,len(eval_imgs)), BATCH_CAM), ncols=100, desc=f"{method} {tag} (CAM batch)"):
            batch_ids = list(range(start, min(start+BATCH_CAM, min(N_EVAL,len(eval_imgs)))))
            imgs_b = [eval_imgs[i] for i in batch_ids]
            cams_b = get_cam_maps_batch(m, imgs_b, method, size=size, target_layer=target_layer,
                                        scorecam_batch=32)
            cams_all.extend(cams_b)

        for i in tqdm(range(min(N_EVAL, len(eval_imgs))), ncols=100, desc=f"{method} {tag}"):
            img = eval_imgs[i]; Mf,Mt,Mb = eval_masks[i]
            cam = cams_all[i]
            # base logit
            _, logit_base = logits_for_batch(m, [img], size=size)
            logit_base = float(logit_base[0])

            # resize masks to image space for area match
            H, W = img.shape[:2]
            Mf_r = ensure_mask_size(Mf, H, W)
            Mt_r = ensure_mask_size(Mt, H, W)
            Mb_r = ensure_mask_size(Mb, H, W)

            area_f = int(Mf_r.sum())
            area_t = int(Mt_r.sum())

            yy_bg, xx_bg = np.where((1 - Mb_r) > 0)
            bgF = np.zeros((H, W), dtype=np.uint8)
            bgT = np.zeros((H, W), dtype=np.uint8)

            if len(yy_bg) > 0:
                if area_f > 0:
                    selF = np.random.choice(len(yy_bg), size=min(area_f, len(yy_bg)), replace=False)
                    bgF[yy_bg[selF], xx_bg[selF]] = 1
                if area_t > 0:
                    selT = np.random.choice(len(yy_bg), size=min(area_t, len(yy_bg)), replace=False)
                    bgT[yy_bg[selT], xx_bg[selT]] = 1

            img_F  = occlude_mask(img, Mf_r, 255)
            img_T  = occlude_mask(img, Mt_r, 255)
            img_bF = occlude_mask(img, bgF,  255)
            img_bT = occlude_mask(img, bgT,  255)

            pF, logF   = logits_for_batch(m, [img_F],  size=size)
            pT, logT   = logits_for_batch(m, [img_T],  size=size)
            pbF, logbF = logits_for_batch(m, [img_bF], size=size)
            pbT, logbT = logits_for_batch(m, [img_bT], size=size)
            logF  = float(logF[0]);  logT  = float(logT[0])
            logbF = float(logbF[0]); logbT = float(logbT[0])

            dropF = max(0.0, logit_base - logF)
            dropT = max(0.0, logit_base - logT)
            dropbF= max(0.0, logit_base - logbF)
            dropbT= max(0.0, logit_base - logbT)

            # structural mass
            H,W = img.shape[:2]
            cam_r = resize_to(cam, (H,W))
            FMR,TMR,BL = FMR_TMR_BL(cam_r, Mf,Mt,Mb)

            rows.append(dict(regime=tag, method=method, idx=i,
                             drop_finder=dropF, drop_timing=dropT,
                             drop_bgF=dropbF, drop_bgT=dropbT,
                             FMR=FMR, TMR=TMR, BL=BL))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/f"causal_occ_{tag}.csv", index=False)

    # correlations
    cors=[]
    for method in methods:
        d = df[df.method==method]
        if len(d):
            rhoF,_ = spearmanr(d["FMR"], d["drop_finder"])
            rhoT,_ = spearmanr(d["TMR"], d["drop_timing"])
            cors.append(dict(regime=tag, method=method, rho_FMR_vs_dropF=rhoF, rho_TMR_vs_dropT=rhoT))
    sm=pd.DataFrame(cors)
    sm.to_csv(OUT/f"causal_occ_{tag}_corr_summary.csv", index=False)
    return df, sm

def run_stress_monotonic(models, methods, tag="FT_STRUCT", size=224):
    """Progressively destroy structure (increase occlusion %) and check monotonic metrics)"""
    oc_perc = [0.0, 0.1, 0.2, 0.4, 0.6]
    rows=[]
    # clean QR set to degrade
    base_qr, base_masks, _ = build_eval_set(n_qr=min(120,N_EVAL), n_neg=0)
    for method in methods:
        if method not in AVAILABLE_CAMS: continue
        print(f"[STRESS] {tag} {method}")
        m = models[tag]; tl = find_last_conv(m)
        for p in oc_perc:
            # build degraded batch and CAM as big batch
            imgs_p = []
            for i in range(len(base_qr)):
                img = base_qr[i].copy(); Mf,Mt,Mb = base_masks[i]
                yy,xx = np.where(Mb)
                k = int(len(yy)*p)
                if k>0:
                    sel = np.random.choice(len(yy), size=k, replace=False)
                    img[yy[sel], xx[sel]] = 255
                imgs_p.append(img)
            cams = []
            for start in tqdm(range(0, len(imgs_p), BATCH_CAM), ncols=100, desc=f"{method} p={p}"):
                cams.extend(get_cam_maps_batch(m, imgs_p[start:start+BATCH_CAM], method,
                                               size=size, target_layer=tl, scorecam_batch=32))
            # metrics
            for i in range(len(base_qr)):
                img = imgs_p[i]; Mf,Mt,Mb = base_masks[i]
                cam_r = resize_to(cams[i], img.shape[:2])
                FMR,TMR,BL = FMR_TMR_BL(cam_r, Mf,Mt,Mb)
                rows.append(dict(regime=tag, method=method, p=p, FMR=FMR, TMR=TMR, BL=BL))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/f"stress_monotonic_{tag}.csv", index=False)

    sm = df.groupby(["method","p"]).mean(numeric_only=True).reset_index()
    for met in sm["method"].unique():
        d = sm[sm.method==met]
        plt.figure(); plt.plot(d["p"], d["FMR"], marker='o'); plt.title(f"{tag} {met} FMR vs occlusion"); plt.xlabel("% occlusion"); plt.ylabel("FMR")
        plt.savefig(FIG/f"stress_{tag}_{met}_FMR.png", dpi=160); plt.close()
        plt.figure(); plt.plot(d["p"], d["TMR"], marker='o'); plt.title(f"{tag} {met} TMR vs occlusion"); plt.xlabel("% occlusion"); plt.ylabel("TMR")
        plt.savefig(FIG/f"stress_{tag}_{met}_TMR.png", dpi=160); plt.close()
        plt.figure(); plt.plot(d["p"], d["BL"], marker='o'); plt.title(f"{tag} {met} BL vs occlusion"); plt.xlabel("% occlusion"); plt.ylabel("BL")
        plt.savefig(FIG/f"stress_{tag}_{met}_BL.png", dpi=160); plt.close()
    return df

def run_threshold_robustness(models, eval_imgs, eval_masks, methods, tag="FT_STRUCT", size=224):
    """IoU between thresholded CAM and QR mask vs threshold ∈ [0,1] (batched CAMs)"""
    ths = np.linspace(0.05, 0.95, 19)
    rows=[]
    top_methods = methods if len(methods)<=3 else [m for m in methods if m in ["eigengradcam","layercam"]]
    for method in top_methods:
        if method not in AVAILABLE_CAMS: continue
        print(f"[THRESH] {tag} {method}")
        m = models[tag]; tl = find_last_conv(m)
        # batch CAMs first
        cams_all=[]
        for start in tqdm(range(0, min(N_EVAL, len(eval_imgs)), BATCH_CAM), ncols=100, desc=f"{method} (CAM batch)"):
            ii = list(range(start, min(start+BATCH_CAM, min(N_EVAL, len(eval_imgs)))))
            imgs_b = [eval_imgs[i] for i in ii]
            cams_b = get_cam_maps_batch(m, imgs_b, method, size=size, target_layer=tl, scorecam_batch=32)
            cams_all.extend(cams_b)
        for i in tqdm(range(min(N_EVAL, len(eval_imgs))), ncols=100, desc=f"{method} (IoU)"):
            img = eval_imgs[i]; Mf,Mt,Mb = eval_masks[i]
            H,W = img.shape[:2]; cam_r=resize_to(cams_all[i], (H,W))
            Mb_r  = ensure_mask_size(Mb, H, W)
            for t in ths:
                mask = (cam_r >= t).astype(np.uint8)
                inter = int((mask & Mb_r).sum())
                union = int((mask | Mb_r).sum()) + 1e-6
                iou = float(inter/union)
                rows.append(dict(regime=tag, method=method, idx=i, thr=t, IoU=iou))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/f"thr_robust_{tag}.csv", index=False)
    sm = df.groupby(["method","thr"]).mean(numeric_only=True).reset_index()
    for met in sm["method"].unique():
        d = sm[sm.method==met]
        plt.figure(); plt.plot(d["thr"], d["IoU"], marker='o'); plt.title(f"{tag} {met} IoU vs threshold"); plt.xlabel("threshold"); plt.ylabel("IoU (CAM vs QR box)")
        plt.savefig(FIG/f"thr_{tag}_{met}_IoU_curve.png", dpi=160); plt.close()
    return df

def run_cross_resolution(models, eval_imgs, eval_masks, methods, tag="FT_STRUCT"):
    """CAM metrics at input sizes 224/256/320 to test stability (batched)"""
    sizes = [224,256,320]
    rows=[]
    for method in methods:
        if method not in AVAILABLE_CAMS: continue
        print(f"[XRES] {tag} {method}")
        m = models[tag]; tl = find_last_conv(m)
        for size in sizes:
            cams_all=[]
            for start in tqdm(range(0, min(N_EVAL, len(eval_imgs)), BATCH_CAM), ncols=100, desc=f"{method} {size} (CAM batch)"):
                ii = list(range(start, min(start+BATCH_CAM, min(N_EVAL, len(eval_imgs)))))
                imgs_b = [eval_imgs[i] for i in ii]
                cams_b = get_cam_maps_batch(m, imgs_b, method, size=size, target_layer=tl, scorecam_batch=32)
                cams_all.extend(cams_b)
            for i in tqdm(range(min(N_EVAL, len(eval_imgs))), ncols=100, desc=f"{method} {size} (metrics)"):
                img = eval_imgs[i]; Mf,Mt,Mb = eval_masks[i]
                H,W=img.shape[:2]; cam_r=resize_to(cams_all[i], (H,W))
                FMR,TMR,BL = FMR_TMR_BL(cam_r, Mf,Mt,Mb)
                rows.append(dict(regime=tag, method=method, size=size, FMR=FMR, TMR=TMR, BL=BL))
    df=pd.DataFrame(rows)
    df.to_csv(OUT/f"xres_{tag}.csv", index=False)
    sm = df.groupby(["method"]).agg({"FMR":["mean","std"],"TMR":["mean","std"],"BL":["mean","std"]})
    sm.to_csv(OUT/f"xres_{tag}_summary.csv")
    return df

def main():
    print("[STEP] Checking prior metrics…")
    ensure_prior_metrics_index()

    print("[STEP] Building small train set for checkpoints (fast)…")
    tr_qr, _, _ = build_eval_set(n_qr=600, n_neg=0)
    tr_neg = [make_negative() for _ in range(600)]
    tr_imgs = tr_qr + tr_neg
    tr_lbls = [1]*len(tr_qr) + [0]*len(tr_neg)

    print("[STEP] Loading or training checkpoints (ZS / FT_STRUCT / FT_LEAKMIN)…")
    models = load_or_train_models(tr_imgs, tr_lbls)

    print("[STEP] Building evaluation set…")
    ev_qr, ev_masks, _ = build_eval_set(n_qr=N_EVAL, n_neg=0)

    # ---------------- Faithfulness ----------------
    for tag in REGIMES:
        fsum = OUT/f"faithfulness_{tag}_summary.csv"
        if fsum.exists():
            print(f"[SKIP] Faithfulness {tag} (exists)")
        else:
            run_faithfulness(models, ev_qr, ev_masks, METHODS, tag=tag, size=224, steps=20)

    # ---------------- Causal occlusion ----------------
    for tag in REGIMES:
        csum = OUT/f"causal_occ_{tag}_corr_summary.csv"
        if csum.exists():
            print(f"[SKIP] Causal occlusion {tag} (exists)")
        else:
            run_causal_occlusion(models, ev_qr, ev_masks, METHODS, tag=tag, size=224)

    # ---------------- Monotonic stress (use FT_STRUCT) ----------------
    smp = OUT/"stress_monotonic_FT_STRUCT.csv"
    if smp.exists():
        print("[SKIP] Stress monotonic FT_STRUCT (exists)")
    else:
        run_stress_monotonic(models, METHODS, tag="FT_STRUCT", size=224)

    # ---------------- Threshold robustness (FT_STRUCT) ----------------
    thrp = OUT/"thr_robust_FT_STRUCT.csv"
    if thrp.exists():
        print("[SKIP] Threshold robustness FT_STRUCT (exists)")
    else:
        run_threshold_robustness(models, ev_qr, ev_masks, METHODS, tag="FT_STRUCT", size=224)

    # ---------------- Cross-resolution (FT_STRUCT) ----------------
    xresp = OUT/"xres_FT_STRUCT.csv"
    if xresp.exists():
        print("[SKIP] Cross-resolution FT_STRUCT (exists)")
    else:
        run_cross_resolution(models, ev_qr, ev_masks, METHODS, tag="FT_STRUCT")

    # ---------------- Runtime & Memory ----------------
    for tag in REGIMES:
        rp = OUT/f"runtime_memory_{regime_to_file(tag)}.csv" if False else OUT/f"runtime_memory_{tag}.csv"
        if rp.exists():
            print(f"[SKIP] Runtime/memory {tag} (exists)")
        else:
            eval_runtime_memory(models[tag], ev_qr, METHODS, size=224, n_sample=64, regime=tag)

    # ---------------- Index outputs ----------------
    index_rows=[]
    for p in sorted(OUT.glob("*.csv")):
        index_rows.append({"file": str(p), "kind": "csv"})
    for p in sorted(FIG.glob("*.png")):
        index_rows.append({"file": str(p), "kind": "figure"})
    pd.DataFrame(index_rows).to_csv(OUT/"_EXTRA_index.csv", index=False)
    print("[DONE] Extra ablations complete.")
    print(f"[INDEX] {OUT / '_EXTRA_index.csv'}")

def regime_to_file(tag: str) -> str:
    return tag

if __name__ == "__main__":
    main()
