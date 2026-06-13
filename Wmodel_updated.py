#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wmodel.py - Site-specific vs pooled melanoma classifiers with patient-group splits
and a built-in data-efficiency control experiment.

What this script can do (default behavior):
- Train pooled + per-site models in one run (using ALL available data per split)
- Use group-aware (patient) splits to prevent leakage
- Evaluate ALL models on a COMMON pooled test set (same exact test rows for every model)
- Produce cross-site metrics (train-domain model x test-site slice) so you can build heatmaps later
- Optionally enforce equal training budgets across sites (data-efficiency control) via --equalize-sites

Outputs (in one timestamped folder under OUT_ROOT):
- sites_present.json: sites found, baseline budgets, site split counts
- sampling_plan.json: budgets and actual sampled counts
- data_manifest.csv: exact isic_id used for each model and split (train/val/test)
- summary.csv: pooled and site-specific metrics with counts
- Per-model folders: pooled/, site_trunk/, site_extremity/, etc.
    - model_best.pth, model_last.pth
    - metrics.json (history + final test metrics + confusion matrix numbers)
    - Optional confusion matrix PNGs if SAVE_CONFUSION = True

Run:
    python Wmodel.py train

Optional overrides are available via CLI, but everything works with defaults.
"""

import os
import sys
import math
import json
import random
import zlib  # CHANGED: used for stable per-site sampling seeds (avoid Python hash())
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import torch.distributed as dist

import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

import timm
from tqdm import tqdm


# ------------------------------ Hardcoded defaults (edit here) ------------------------------

COMMON_TEST_SET = True  # set True to evaluate ALL models on the exact same pooled test set

DEFAULTS = {
    # Data paths
    "METADATA_FILE": "metadata.csv",
    "DATA_ROOT": "data",
    "MASK_DIRNAME": "masks",

    # Columns
    "SITE_COL": "anatom_site_general",
    "GROUP_COL": "group_id",
    "SPLIT_COL": None,  # if provided, must contain train/val/test

    # Site mode
    "SITE_MODE": "both",  # per-site | pooled | both
    "SITES": ["headneck", "trunk", "extremity"],  # default sites to train (set None for all)

    # Train hyperparams
    "BATCH_SIZE": 32,
    "EPOCHS": 25,
    "LR": 7e-5,
    "WEIGHT_DECAY": 1e-4,
    "FOCUS_STRENGTH": 0.4,
    "INPUT_SIZE": 512,
    "AUGMENT": False,
    "PATIENCE": 10,
    "WARMUP_EPOCHS": 1,
    "SEED": 1337,

    # Splits
    "TEST_SIZE": 0.2,
    "VAL_SIZE": 0.1,

    # Evaluation
    "COMMON_TEST_SET": COMMON_TEST_SET,

    # System
    "DDP": False,
    "OUT_ROOT": "results/site_vs_pooled",

    # Data-efficiency control (disabled by default)
    "EQUALIZE_SITES": True,
    "BASELINE_SITE": "headneck",
    "POOLED_MULTIPLIER": 1,

    # Output controls
    "SAVE_CONFUSION": False,   # keep graphs minimal by default

    # Extra outputs for cross-site heatmaps / error analysis
    "SAVE_TEST_PREDICTIONS": True,
    "SAVE_CROSS_SITE_PREDICTIONS": True,
}


# ------------------------------ Utils ------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# CHANGED: stable 0-9999 hash so per-site sampling is deterministic across runs and DDP processes.
def stable_hash_0_9999(text: str) -> int:
    """Stable 0-9999 hash (used for deterministic per-site sampling seeds)."""
    h = zlib.crc32(str(text).encode("utf-8")) & 0xffffffff
    return int(h % 10000)



def log_main(is_main: bool, *args, **kwargs):
    if is_main:
        print(*args, **kwargs, flush=True)


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pad_to_square(im: Image.Image, fill=0) -> Image.Image:
    # FIXED: correct symmetric padding to square (old version had wrong bottom/right padding)
    w, h = im.size
    if w == h:
        return im
    diff = abs(w - h)
    pad1 = diff // 2
    pad2 = diff - pad1
    if w > h:
        # pad top/bottom
        padding = (0, pad1, 0, pad2)
    else:
        # pad left/right
        padding = (pad1, 0, pad2, 0)
    return TF.pad(im, padding, fill=fill)


def normalize_site(s: str) -> str:
    """
    Robust site normalization for ISIC-style metadata.

    Returns canonical tokens like:
        - headneck
        - trunk
        - extremity
        - unknown
        - <other compact token>
    """
    if s is None:
        return "unknown"
    # Handle pandas NaN
    try:
        if isinstance(s, float) and np.isnan(s):
            return "unknown"
    except Exception:
        pass

    s0 = str(s).strip().lower()
    if s0 in {"", "nan", "none"}:
        return "unknown"

    # Compact form for matching
    c = s0.replace(" ", "").replace("_", "").replace("-", "").replace("/", "").replace("&", "and")

    # Head/neck variants
    if c in {"headneck", "headandneck", "headneckregion", "headandneckregion", "neck"}:
        return "headneck"
    # Many files use head/neck exactly; handle it even if '/' not stripped in upstream
    if "head" in c and "neck" in c:
        return "headneck"

    # Extremity variants
    if c in {
        "upperextremity",
        "lowerextremity",
        "upperextremities",
        "lowerextremities",
        "extremity",
        "extremities",
        "upperlimb",
        "lowerlimb",
        "arm",
        "leg",
    }:
        return "extremity"
    if "extremity" in c:
        return "extremity"

    # Trunk variants
    if c in {"trunk", "torso"}:
        return "trunk"

    # Unknown-ish tokens
    if c in {"unknown", "na", "notreported", "unspecified"}:
        return "unknown"

    return c


# ------------------------------ Metadata encoder ------------------------------

class MetadataEncoder:
    """
    Fixed layout metadata vector:
        [ z_age, onehot_sex(2), onehot_site(S) ]
    S = number of unique normalized sites in metadata (excluding unknown)
    """
    def __init__(self, sites: List[str]):
        sites = [normalize_site(s) for s in sites if isinstance(s, str)]
        self.sites = sorted(list({s for s in sites if s and s != "unknown"}))
        self.site_to_idx = {s: i for i, s in enumerate(self.sites)}

    @property
    def dim(self) -> int:
        return 1 + 2 + len(self.sites)

    def encode(self, row: dict) -> torch.Tensor:
        age = row.get("age_approx", 50)
        age = float(age) if pd.notna(age) else 50.0
        z_age = (age - 50.0) / 20.0

        sex = str(row.get("sex", "")).strip().lower()
        onehot_sex = [1.0, 0.0] if sex == "male" else [0.0, 1.0] if sex == "female" else [0.0, 0.0]

        site_raw = row.get("anatom_site_general", None)
        site = normalize_site(str(site_raw)) if site_raw is not None else "unknown"
        onehot_site = [0.0] * len(self.sites)
        if site in self.site_to_idx:
            onehot_site[self.site_to_idx[site]] = 1.0

        vec = [z_age] + onehot_sex + onehot_site
        return torch.tensor(vec, dtype=torch.float32)


# ------------------------------ Dataset ------------------------------

class LesionDataset(Dataset):
    """
    Returns:
        x4: 4 x H x W tensor (RGB normalized + binary mask)
        meta: metadata feature tensor
        y: int label {0,1}
        isic_id: string
        site_norm: normalized site string
        group_id: grouping key (patient/group id)
    """
    def __init__(self,
                 df: pd.DataFrame,
                 data_root: Path,
                 mask_dirname: str,
                 input_size: int,
                 meta_encoder: MetadataEncoder,
                 group_col: str,
                 augment: bool):
        self.data_root = data_root
        self.mask_dir = data_root / mask_dirname
        self.size = (int(input_size), int(input_size))
        self.meta_encoder = meta_encoder
        self.group_col = str(group_col)
        self.augment = bool(augment)

        # FIXED: keep row metadata aligned with actual existing files
        rows = []
        paths = []
        labels = []

        df = df.reset_index(drop=True)
        for _, r in df.iterrows():
            isic_id = r.get("isic_id", None)
            if pd.isna(isic_id):
                continue

            img_fp = self.data_root / f"{isic_id}.jpg"
            mask_fp = self.mask_dir / f"{isic_id}_segmentation.png"

            if not img_fp.exists():
                continue
            if not mask_fp.exists():
                continue

            diag = str(r.get("diagnosis", "")).strip().lower()
            y = 0 if diag == "benign" else 1

            rows.append(r.to_dict())
            paths.append(img_fp)
            labels.append(int(y))

        self.rows = rows
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_fp = self.paths[idx]
        r = self.rows[idx]
        y = int(self.labels[idx])

        img = Image.open(img_fp).convert("RGB")

        mask_fp = self.mask_dir / f"{img_fp.stem}_segmentation.png"
        mask = Image.open(mask_fp).convert("L")

        img = pad_to_square(img).resize(self.size, Image.BILINEAR)
        mask = pad_to_square(mask).resize(self.size, Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() < 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            if y == 1 and random.random() < 0.3:
                angle = random.choice([0, 90, 180])
                img, mask = TF.rotate(img, angle), TF.rotate(mask, angle)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        mask_t = (TF.to_tensor(mask) > 0.5).float()
        x4 = torch.cat([img_t, mask_t], 0)

        meta = self.meta_encoder.encode(r)

        isic_id = str(r.get("isic_id", img_fp.stem))
        site_norm = str(r.get("_site_norm", normalize_site(r.get("anatom_site_general", None))))
        group_id = r.get(self.group_col, r.get("group_id", ""))

        return x4, meta, y, isic_id, site_norm, group_id

# ------------------------------ Model ------------------------------

class ConvNeXtDual(nn.Module):
    def __init__(self, meta_dim: int, drop=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            in_chans=4,
            num_classes=0,
            drop_rate=float(drop),
            drop_path_rate=float(drop),
            global_pool="avg",
        )
        hidden = 512
        self.meta_to_gamma_beta = nn.Sequential(
            nn.Linear(int(meta_dim), hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1536),  # 2 x 768
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x4, meta):
        feat = self.backbone(x4)                  # [B, 768]
        gamma_beta = self.meta_to_gamma_beta(meta)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # [B, 768] each
        feat = gamma * feat + beta
        return self.fc(feat)


# ------------------------------ Evaluation ------------------------------

@torch.no_grad()
def evaluate(model, loader, device, ce_weight):
    model.eval()
    ce = nn.CrossEntropyLoss(weight=ce_weight, reduction="mean")

    tot = 0.0
    y_true = []
    y_prob = []
    y_pred = []

    isic_ids = []
    site_norms = []
    group_ids = []

    for x4, meta, y, isic_id, site_norm, group_id in loader:
        x4, meta, y = x4.to(device), meta.to(device), y.to(device)
        logits = model(x4, meta)
        loss_b = ce(logits, y).item()
        tot += loss_b * y.size(0)

        prob = torch.softmax(logits, 1)[:, 1]
        pred = torch.argmax(logits, 1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_prob.extend(prob.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

        # keep identifiers for downstream analysis (heatmaps, error analysis, etc.)
        isic_ids.extend([str(x) for x in isic_id])
        site_norms.extend([str(x) for x in site_norm])
        group_ids.extend([str(x) for x in group_id])

    N = len(loader.dataset)
    if N <= 0:
        return 0.0, 0.0, 0.0, 0.0, [], [], [], [], [], []

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = float((y_true_arr == y_pred_arr).mean())

    # safe guards when only one class exists in y_true
    f1 = float(f1_score(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0
    auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0

    return float(tot / max(N, 1)), acc, f1, auc, y_true, y_pred, y_prob, isic_ids, site_norms, group_ids

# ------------------------------ Splitting ------------------------------



def _class_counts_from_labels(labels: List[int]) -> Dict[str, int]:
    labels = [int(x) for x in labels]
    n_pos = int(sum(labels))
    n_neg = int(len(labels) - n_pos)
    return {"n_pos": n_pos, "n_neg": n_neg, "n_total": int(len(labels))}


def _save_predictions_table(out_csv_gz: Path,
                            model_name: str,
                            train_site: str,
                            test_site: str,
                            y_true: List[int],
                            y_pred: List[int],
                            y_prob: List[float],
                            isic_ids: List[str],
                            site_norms: List[str],
                            group_ids: List[str]):
    dfp = pd.DataFrame({
        "model": [str(model_name)] * len(y_true),
        "train_site": [str(train_site)] * len(y_true),
        "test_site": [str(test_site)] * len(y_true),
        "isic_id": [str(x) for x in isic_ids],
        "site_norm": [str(x) for x in site_norms],
        "group_id": [str(x) for x in group_ids],
        "y_true": [int(x) for x in y_true],
        "y_pred": [int(x) for x in y_pred],
        "y_prob": [float(x) for x in y_prob],
    })
    out_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    dfp.to_csv(out_csv_gz, index=False, compression="gzip")


def eval_saved_model_on_test_rows(work_dir: Path,
                                 test_rows: pd.DataFrame,
                                 all_sites: List[str],
                                 cfg,
                                 local_rank: int,
                                 model_name: str,
                                 train_site: str,
                                 test_site: str,
                                 save_dir: Optional[Path] = None):
    """
    Load work_dir/model_best.pth and evaluate on a provided test_rows dataframe.

    Used for cross-site heatmap-ready metrics when COMMON_TEST_SET is enabled:
    every model is evaluated on the same pooled test split, and we also compute
    metrics on per-site slices of that shared test set.

    Returns a dict with acc/f1/auc/n_test and class counts, or None if empty.
    """
    if test_rows is None or len(test_rows) == 0:
        return None

    meta_enc = MetadataEncoder(all_sites)
    ds = LesionDataset(test_rows, cfg.data_root, cfg.mask_dirname, cfg.input_size, meta_enc,
                      group_col=cfg.group_col, augment=False)
    if len(ds) == 0:
        return None

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    model = ConvNeXtDual(meta_dim=meta_enc.dim).to(device)
    state = torch.load(work_dir / "model_best.pth", map_location=device)
    model.load_state_dict(state, strict=True)

    ce_weight = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    loss, acc, f1, auc, y_true, y_pred, y_prob, isic_ids, site_norms, group_ids = evaluate(model, dl, device, ce_weight)

    cm = confusion_matrix(y_true, y_pred).tolist()
    cls = _class_counts_from_labels(y_true)

    out = {
        "model": str(model_name),
        "train_site": str(train_site),
        "test_site": str(test_site),
        "loss": float(loss),
        "acc": float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "n_test": int(len(ds)),
        "n_test_pos": int(cls["n_pos"]),
        "n_test_neg": int(cls["n_neg"]),
        "confusion_matrix": cm,
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        # metrics JSON
        with open(save_dir / f"eval_{str(test_site)}.json", "w") as f:
            json.dump(out, f, indent=2)
        # predictions table (compressed)
        if getattr(cfg, "save_cross_site_predictions", True):
            _save_predictions_table(
                save_dir / f"preds_{str(test_site)}.csv.gz",
                model_name=str(model_name),
                train_site=str(train_site),
                test_site=str(test_site),
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                isic_ids=isic_ids,
                site_norms=site_norms,
                group_ids=group_ids,
            )

    return out
def make_group_splits(df: pd.DataFrame,
                      label_col: str,
                      group_col: str,
                      test_size: float,
                      val_size: float,
                      seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns boolean masks aligned to df.index:
        train_mask, val_mask, test_mask

    FIXED: validation indices must be mapped back to the original indices.
    """
    groups = df[group_col].values
    labels = (df[label_col].astype(str).str.lower() != "benign").astype(int).values
    idx = np.arange(len(df))

    # split off test
    gss = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
    tr_idx, te_idx = next(gss.split(idx, labels, groups=groups))

    # split val from remaining train pool
    val_frac_of_tr = float(val_size) / max(1e-12, (1.0 - float(test_size)))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_tr, random_state=int(seed) + 1)
    tr2_rel, va_rel = next(gss2.split(tr_idx, labels[tr_idx], groups=groups[tr_idx]))

    tr_final_idx = tr_idx[tr2_rel]
    va_idx = tr_idx[va_rel]

    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)

    train_mask[tr_final_idx] = True
    val_mask[va_idx] = True
    test_mask[te_idx] = True

    return train_mask, val_mask, test_mask


# ------------------------------ Data-efficiency sampling helpers ------------------------------

def _allocate_proportional_int(counts_dict: Dict[tuple, int], target_total: int) -> Dict[tuple, int]:
    if target_total <= 0:
        return {k: 0 for k in counts_dict.keys()}

    total_avail = int(sum(counts_dict.values()))
    if target_total >= total_avail:
        return {k: int(v) for k, v in counts_dict.items()}

    raw = {k: (counts_dict[k] * target_total / total_avail) for k in counts_dict.keys()}
    base = {k: int(math.floor(raw[k])) for k in raw.keys()}
    used = int(sum(base.values()))
    remainder = int(target_total - used)

    frac_order = sorted(raw.keys(), key=lambda k: (raw[k] - math.floor(raw[k])), reverse=True)
    for k in frac_order:
        if remainder <= 0:
            break
        if base[k] < counts_dict[k]:
            base[k] += 1
            remainder -= 1

    if remainder > 0:
        for k in counts_dict.keys():
            if remainder <= 0:
                break
            cap = int(counts_dict[k] - base[k])
            if cap <= 0:
                continue
            add = int(min(cap, remainder))
            base[k] += add
            remainder -= add

    return base


def stratified_sample_exact(df: pd.DataFrame, n: int, seed: int, strata_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    n = int(n)

    if n >= len(df):
        return df, {"target_n": int(n), "used_n": int(len(df))}

    rng = np.random.RandomState(int(seed))

    keys = df[strata_cols].apply(lambda r: tuple(r.values.tolist()), axis=1)
    df["_stratum_key"] = keys

    counts = df["_stratum_key"].value_counts().to_dict()
    alloc = _allocate_proportional_int(counts, n)

    picked_idx = []
    for key, k_n in alloc.items():
        k_n = int(k_n)
        if k_n <= 0:
            continue
        sub = df[df["_stratum_key"] == key]
        rs = int(rng.randint(0, 2**31 - 1))
        picked_idx.append(sub.sample(n=k_n, random_state=rs).index.values)

    if len(picked_idx) == 0:
        out = df.sample(n=n, random_state=int(seed))
    else:
        out = df.loc[np.concatenate(picked_idx)].copy()

    out.drop(columns=["_stratum_key"], inplace=True, errors="ignore")
    return out, {"target_n": int(n), "used_n": int(len(out))}


def sample_equal_thirds_by_site(df: pd.DataFrame,
                                n: int,
                                seed: int,
                                sites: List[str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Sample EXACTLY ~1/3 from each requested site (trunk/extremity/headneck) up to total n.
    - Primary constraint: equal thirds by site (not label-stratified).
    - If a site has fewer than its quota, we take all from that site and
      backfill the remainder from the remaining requested sites at random.

    Returns:
        sampled_df, info dict
    """
    df = df.copy()
    n = int(n)

    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy(), {
            "target_n": int(n),
            "used_n": 0,
            "alloc_target_by_site": {},
            "used_by_site": {},
            "backfilled_n": 0,
        }

    # Keep only requested sites that exist in df
    req_sites = [normalize_site(s) for s in sites]
    avail_sites = [s for s in req_sites if s in df["_site_norm"].unique().tolist()]
    if len(avail_sites) == 0:
        # Fallback: just random sample from df
        out = df.sample(n=min(n, len(df)), random_state=int(seed)).copy()
        return out, {
            "target_n": int(n),
            "used_n": int(len(out)),
            "alloc_target_by_site": {},
            "used_by_site": dict(out["_site_norm"].value_counts().to_dict()),
            "backfilled_n": 0,
        }

    # Equal third-ish allocation
    k = int(len(avail_sites))
    base = int(n // k)
    rem = int(n - base * k)

    rng = np.random.RandomState(int(seed))
    shuffled = avail_sites.copy()
    rng.shuffle(shuffled)

    alloc = {s: int(base) for s in avail_sites}
    for s in shuffled[:rem]:
        alloc[s] += 1

    picked = []
    used_by_site = {}
    for s in avail_sites:
        sub = df[df["_site_norm"] == s]
        want = int(alloc.get(s, 0))
        take = int(min(want, len(sub)))
        if take <= 0:
            used_by_site[s] = 0
            continue
        rs = int(rng.randint(0, 2**31 - 1))
        part = sub.sample(n=take, random_state=rs).copy()
        picked.append(part)
        used_by_site[s] = int(len(part))

    out = pd.concat(picked, ignore_index=True) if len(picked) > 0 else df.iloc[0:0].copy()

    # Backfill if any site was short (should be rare with baseline-derived budgets)
    backfilled = 0
    short = int(n - len(out))
    if short > 0:
        # only backfill from requested sites (keep pooled strictly within these sites)
        pool = df[df["_site_norm"].isin(avail_sites)].copy()
        if len(out) > 0:
            pool = pool[~pool["isic_id"].isin(out["isic_id"].tolist())]
        if len(pool) > 0:
            rs = int(rng.randint(0, 2**31 - 1))
            extra = pool.sample(n=min(short, len(pool)), random_state=rs).copy()
            out = pd.concat([out, extra], ignore_index=True)
            backfilled = int(len(extra))

    # Final safety cap
    if len(out) > n:
        out = out.sample(n=n, random_state=int(seed)).copy()

    return out, {
        "target_n": int(n),
        "used_n": int(len(out)),
        "alloc_target_by_site": {k: int(v) for k, v in alloc.items()},
        "used_by_site": {k: int(v) for k, v in used_by_site.items()},
        "backfilled_n": int(backfilled),
    }

def make_equal_parts_common_test_set(
    df_test: pd.DataFrame,
    sites: List[str],
    seed: int,
    stratify_by_label: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Build a COMMON test set that is composed of EQUAL parts from each site.
    We downsample each site's test rows to n_per_site = min(counts across sites).
    Optionally stratify within each site by label (_y) for more stable class balance.
    """
    df_test = df_test.copy()
    req_sites = [normalize_site(s) for s in sites]
    avail_sites = [s for s in req_sites if s in df_test["_site_norm"].unique().tolist()]

    if len(avail_sites) == 0 or len(df_test) == 0:
        return df_test.iloc[0:0].copy(), {
            "used_n_total": 0,
            "n_per_site": 0,
            "used_by_site": {},
            "used_by_site_label_counts": {},
            "avail_sites": avail_sites,
        }

    counts = {s: int((df_test["_site_norm"] == s).sum()) for s in avail_sites}
    n_per_site = int(min(counts.values())) if len(counts) > 0 else 0

    if n_per_site <= 0:
        return df_test.iloc[0:0].copy(), {
            "used_n_total": 0,
            "n_per_site": 0,
            "used_by_site": {k: 0 for k in counts.keys()},
            "used_by_site_label_counts": {},
            "avail_sites": avail_sites,
        }

    rng = np.random.RandomState(int(seed))
    picked = []
    used_by_site = {}
    used_by_site_label_counts = {}

    for s in avail_sites:
        sub = df_test[df_test["_site_norm"] == s].copy()
        rs = int(rng.randint(0, 2**31 - 1))

        if stratify_by_label and "_y" in sub.columns:
            part, _ = stratified_sample_exact(sub, n_per_site, rs, ["_y"])
        else:
            part = sub.sample(n=n_per_site, random_state=rs).copy()

        picked.append(part)
        used_by_site[s] = int(len(part))
        if "_y" in part.columns:
            used_by_site_label_counts[s] = {k: int(v) for k, v in part["_y"].value_counts().to_dict().items()}

    out = pd.concat(picked, ignore_index=True)
    return out, {
        "used_n_total": int(len(out)),
        "n_per_site": int(n_per_site),
        "used_by_site": {k: int(v) for k, v in used_by_site.items()},
        "used_by_site_label_counts": used_by_site_label_counts,
        "avail_sites": avail_sites,
    }


# ------------------------------ Training core ------------------------------

@dataclass
class TrainConfig:
    data_root: Path
    metadata_file: Path
    mask_dirname: str

    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    focus_strength: float
    input_size: int
    augment: bool
    patience: int
    warmup_epochs: int

    out_root: Path

    site_mode: str
    sites: Optional[List[str]]

    site_col: str
    group_col: str
    split_col: Optional[str]

    test_size: float
    val_size: float

    common_test_set: bool

    seed: int
    ddp: bool

    equalize_sites: bool
    baseline_site: str
    pooled_multiplier: int

    save_confusion: bool
    save_test_predictions: bool
    save_cross_site_predictions: bool


def build_loaders(df: pd.DataFrame,
                  cfg: TrainConfig,
                  meta_encoder: MetadataEncoder,
                  is_distributed: bool,
                  rank: int,
                  world_size: int):

    ds_train = LesionDataset(df[df["split"] == "train"], cfg.data_root, cfg.mask_dirname,
                             cfg.input_size, meta_encoder, group_col=cfg.group_col, augment=True)
    ds_val = LesionDataset(df[df["split"] == "val"], cfg.data_root, cfg.mask_dirname,
                           cfg.input_size, meta_encoder, group_col=cfg.group_col, augment=False)
    ds_test = LesionDataset(df[df["split"] == "test"], cfg.data_root, cfg.mask_dirname,
                            cfg.input_size, meta_encoder, group_col=cfg.group_col, augment=False)

    if is_distributed:
        smp_train = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        smp_train = None

    # IMPORTANT: evaluate full val/test on each rank (rank0 writes results),
    # avoids needing distributed metric aggregation.
    smp_val = None
    smp_test = None

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, sampler=smp_train,
                          shuffle=(smp_train is None),
                          num_workers=4, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, sampler=smp_val,
                        shuffle=False, num_workers=4, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, sampler=smp_test,
                         shuffle=False, num_workers=4, pin_memory=True)
    return dl_train, dl_val, dl_test

def cosine_warmup_lambda(epoch: int, cfg: TrainConfig):
    if epoch < cfg.warmup_epochs:
        return float(epoch + 1) / float(max(1, cfg.warmup_epochs))
    T = max(1, cfg.epochs - cfg.warmup_epochs)
    t = epoch - cfg.warmup_epochs
    return 0.5 * (1 + math.cos(math.pi * t / T))


def _save_confusion_png(out_dir: Path, y_true, y_pred, title: str):
    # minimal dependency: matplotlib only if needed
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            color = "white" if val > cm.max() / 2 else "black"
            plt.text(j, i, str(val), ha="center", va="center", color=color)
    plt.xticks([0, 1], ["Benign", "Malignant"])
    plt.yticks([0, 1], ["Benign", "Malignant"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / f"{title.replace(' ', '_')}_cm.png")
    plt.close()


def train_one(df_split: pd.DataFrame,
              all_sites: List[str],
              cfg: TrainConfig,
              work_dir: Path,
              is_main: bool,
              local_rank: int,
              world_size: int):

    meta_enc = MetadataEncoder(all_sites)
    is_distributed = bool(cfg.ddp and world_size > 1)

    dl_train, dl_val, dl_test = build_loaders(df_split, cfg, meta_enc,
                                              is_distributed=is_distributed,
                                              rank=local_rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    model = ConvNeXtDual(meta_dim=meta_enc.dim).to(device)

    # FIXED: do not re-init process group here; run() handles init/cleanup
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: cosine_warmup_lambda(e, cfg))
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    ce_weight = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)

    # extra split label counts (helps interpret F1/AUC)
    split_counts = {
        "train": _class_counts_from_labels(getattr(dl_train.dataset, "labels", [])),
        "val": _class_counts_from_labels(getattr(dl_val.dataset, "labels", [])),
        "test": _class_counts_from_labels(getattr(dl_test.dataset, "labels", [])),
    }

    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": []}
    best_auc = 0.0
    patience_ctr = 0

    for epoch in range(cfg.epochs):
        if is_distributed and hasattr(dl_train.sampler, "set_epoch"):
            dl_train.sampler.set_epoch(epoch)

        model.train()
        run_loss = 0.0
        n_seen = 0

        pbar = tqdm(dl_train, desc=f"E{epoch+1}/{cfg.epochs}", disable=not is_main)

        for x4, meta, y, *_ in pbar:
            x4, meta, y = x4.to(device), meta.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(x4, meta)

                base_loss = F.cross_entropy(logits, y, weight=ce_weight, reduction="none")
                mask_area = x4[:, 3, :, :].mean([1, 2]).clamp(min=1e-6)
                weighted = base_loss * (1.0 + cfg.focus_strength * (1.0 - mask_area))

                k = max(1, int(0.3 * len(weighted)))
                loss = torch.topk(weighted, k=k, largest=True).values.mean()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss += float(loss.item()) * int(y.size(0))
            n_seen += int(y.size(0))

            if is_main:
                pbar.set_postfix(loss=f"{run_loss / max(1, n_seen):.4f}")

        train_loss = float(run_loss / max(1, n_seen))
        val_loss, val_acc, val_f1, val_auc, *_ = evaluate(model, dl_val, device, ce_weight)
        scheduler.step()

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(float(val_loss))
        hist["val_acc"].append(float(val_acc))
        hist["val_f1"].append(float(val_f1))
        hist["val_auc"].append(float(val_auc))

        _to_save = model.module if hasattr(model, "module") else model

        if is_main:
            torch.save(_to_save.state_dict(), work_dir / "model_last.pth")

            improved = float(val_auc) > float(best_auc)
            if improved:
                best_auc = float(val_auc)
                patience_ctr = 0
                torch.save(_to_save.state_dict(), work_dir / "model_best.pth")
            else:
                patience_ctr += 1

        if is_distributed:
            dist.barrier()
            obj = [best_auc, patience_ctr]
            dist.broadcast_object_list(obj, src=0)
            best_auc, patience_ctr = obj

        stop = bool(patience_ctr >= cfg.patience)
        if is_distributed:
            obj = [stop]
            dist.broadcast_object_list(obj, src=0)
            stop = bool(obj[0])

        if stop:
            break

    # Load best checkpoint before test
    best_p = work_dir / "model_best.pth"
    if best_p.exists():
        target = model.module if hasattr(model, "module") else model
        state = torch.load(best_p, map_location=device)
        target.load_state_dict(state, strict=True)

    test_loss, test_acc, test_f1, test_auc, y_true, y_pred, y_prob, isic_ids, site_norms, group_ids = evaluate(model, dl_test, device, ce_weight)

    # Save per-image test predictions.
    # IMPORTANT: val/test are evaluated fully on each rank, so only rank0 should write a single file.
    if getattr(cfg, "save_test_predictions", True) and is_main:
        preds_dir = work_dir / "predictions"
        safe_mkdir(preds_dir)
        out_csv_gz = preds_dir / "test_predictions.csv.gz"
        _save_predictions_table(
            out_csv_gz,
            model_name=str(work_dir.name),
            train_site=str(work_dir.name),
            test_site="test_loader",
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            isic_ids=isic_ids,
            site_norms=site_norms,
            group_ids=group_ids,
        )

    if is_main:
        cm = confusion_matrix(y_true, y_pred).tolist()

        payload = {
            "history": hist,
            "best_val_auc": float(best_auc),
            "epochs_ran": int(len(hist.get("train_loss", []))),
            "counts": {
                "n_train": int(len(dl_train.dataset)),
                "n_val": int(len(dl_val.dataset)),
                "n_test": int(len(dl_test.dataset)),
                "train": split_counts["train"],
                "val": split_counts["val"],
                "test": split_counts["test"],
            },
            "test": {
                "loss": float(test_loss),
                "acc": float(test_acc),
                "f1": float(test_f1),
                "auc": float(test_auc),
            },
            "confusion_matrix": cm,
        }

        with open(work_dir / "metrics.json", "w") as f:
            json.dump(payload, f, indent=2)

        if cfg.save_confusion:
            _save_confusion_png(work_dir, y_true, y_pred, "test_final")

    return {
        "loss": float(test_loss),
        "acc": float(test_acc),
        "f1": float(test_f1),
        "auc": float(test_auc),
        "n_train": int(len(dl_train.dataset)),
        "n_val": int(len(dl_val.dataset)),
        "n_test": int(len(dl_test.dataset)),
        "n_test_pos": int(split_counts["test"]["n_pos"]),
        "n_test_neg": int(split_counts["test"]["n_neg"]),
    }

# ------------------------------ Orchestrator ------------------------------

def run(cfg: TrainConfig):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", "0"))

    is_distributed = bool(cfg.ddp and world_size > 1)
    if is_distributed and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    is_main = (rank == 0)
    set_seed(cfg.seed)

    df = pd.read_csv(cfg.metadata_file)

    required_cols = ["isic_id", "diagnosis", cfg.site_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if cfg.group_col not in df.columns:
        raise ValueError(f"Missing required group column: {cfg.group_col}")

    # Basic clean
    df = df[df["isic_id"].notna()].copy()

    # Build binary label early (used for stratified sampling)
    df["_y"] = (df["diagnosis"].astype(str).str.lower() != "benign").astype(int)

    # Filter to images and masks that exist (keep data clean)
    df["img_fp"] = df["isic_id"].apply(lambda x: str(cfg.data_root / f"{x}.jpg"))
    df["mask_fp"] = df["isic_id"].apply(lambda x: str(cfg.data_root / cfg.mask_dirname / f"{x}_segmentation.png"))

    before = len(df)
    df = df[df["img_fp"].apply(lambda p: Path(p).exists())]
    df = df[df["mask_fp"].apply(lambda p: Path(p).exists())]
    after = len(df)
    if is_main and after < before:
        print(f"Skipped {before - after} rows without image or mask files.", flush=True)

    # Normalize sites AFTER filtering so present_sites reflects what is actually usable
    df["_site_norm"] = df[cfg.site_col].apply(normalize_site)
    present_sites = sorted([s for s in df["_site_norm"].dropna().unique().tolist() if s and s != "unknown"])

    # Decide which sites to use
    if cfg.sites:
        wanted = sorted({normalize_site(s) for s in cfg.sites})
        missing = [s for s in wanted if s not in present_sites]
        if missing and is_main:
            print(f"Warning: requested sites not present after filtering: {missing}", flush=True)
        sites_to_use = [s for s in wanted if s in present_sites]
    else:
        sites_to_use = present_sites

    # Guardrail: if you asked for per-site (or both) but there are no usable sites, fail loudly
    if cfg.site_mode in {"per-site", "both"} and len(sites_to_use) == 0:
        msg = (
            "No usable sites found for per-site training. "
            "Check SITE_COL, normalize_site mapping, and that your metadata has non-missing anatomic sites."
        )
        raise ValueError(msg)

    # Prepare or honor split column ONCE globally
    if cfg.split_col and cfg.split_col in df.columns:
        df["split"] = df[cfg.split_col].astype(str).str.lower()
    else:
        tr, va, te = make_group_splits(
            df,
            label_col="diagnosis",
            group_col=cfg.group_col,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            seed=cfg.seed,
        )
        split = np.array(["train"] * len(df), dtype=object)
        split[va] = "val"
        split[te] = "test"
        df["split"] = split

    # Common test set control: if enabled, ALL models use the exact same pooled test split
    # Common test set control: if enabled, ALL models use the exact same pooled test split
    df_test_common = None
    common_test_plan = None
    
    if cfg.common_test_set:
        df_test_pool = df[(df["split"] == "test") & (df["_site_norm"].isin(sites_to_use))].copy()
    
        df_test_common, common_test_plan = make_equal_parts_common_test_set(
            df_test=df_test_pool,
            sites=sites_to_use,
            seed=cfg.seed + 999,
            stratify_by_label=True,
        )
    
        if len(df_test_common) == 0:
            raise ValueError("COMMON_TEST_SET is True but common test set balancing produced 0 rows.")
    
        if is_main:
            print(
                f"COMMON_TEST_SET enabled (equal parts) | n_test_common={len(df_test_common)} "
                f"| n_per_site={common_test_plan.get('n_per_site', 0)} "
                f"| used_by_site={common_test_plan.get('used_by_site', {})}",
                flush=True
            )


    # Baseline budgets from baseline site train/val counts
    baseline_site_norm = normalize_site(cfg.baseline_site)
    if baseline_site_norm not in present_sites:
        candidates = sites_to_use if len(sites_to_use) > 0 else present_sites
        if len(candidates) > 0:
            site_counts = {s: int((df["_site_norm"] == s).sum()) for s in candidates}
            baseline_site_norm = min(site_counts.keys(), key=lambda k: site_counts[k])

    base_train_df = df[(df["_site_norm"] == baseline_site_norm) & (df["split"] == "train")]
    base_val_df = df[(df["_site_norm"] == baseline_site_norm) & (df["split"] == "val")]

    baseline_train_n = int(len(base_train_df))
    baseline_val_n = int(len(base_val_df))

    if baseline_train_n <= 0 or baseline_val_n <= 0:
        if is_main:
            print("Warning: baseline site has 0 samples in train or val; disabling equalize_sites for this run.", flush=True)
        cfg.equalize_sites = False

    # NOTE: pooled_multiplier is no longer used for pooled budgets when equalize_sites is enabled.
    # Keep it pinned to 1 to avoid confusion in logs/JSON.
    if cfg.equalize_sites:
        cfg.pooled_multiplier = 1

    pooled_train_target = int(baseline_train_n) if cfg.equalize_sites else int((df["split"] == "train").sum())
    pooled_val_target = int(baseline_val_n) if cfg.equalize_sites else int((df["split"] == "val").sum())

    if is_main:
        print(f"Site mode: {cfg.site_mode}", flush=True)
        print(f"Sites present (usable): {present_sites}", flush=True)
        print(f"Sites used: {sites_to_use}", flush=True)
        print(
            f"Equalize sites: {cfg.equalize_sites} | baseline_site: {baseline_site_norm} "
            f"| baseline_train_n: {baseline_train_n} | baseline_val_n: {baseline_val_n} "
            f"| pooled_train_target: {pooled_train_target} | pooled_val_target: {pooled_val_target} "
            f"| pooled_sampling: equal_thirds_by_site",
            flush=True
        )

    all_sites = present_sites

    manifest_frames = []
    sampling_plan = {
        "seed": int(cfg.seed),
        "equalize_sites": bool(cfg.equalize_sites),
        "baseline_site": str(baseline_site_norm),
        "baseline_train_n": int(baseline_train_n),
        "baseline_val_n": int(baseline_val_n),
        "pooled_multiplier": int(cfg.pooled_multiplier),
        "pooled_train_target": int(pooled_train_target),
        "pooled_val_target": int(pooled_val_target),
        "actual": {},
    }
    if cfg.common_test_set:
        sampling_plan["common_test_set_plan"] = common_test_plan
    rows = []

    # ------------------ Pooled model ------------------
    pooled_counts = {}
    if cfg.site_mode in {"pooled", "both"}:
        work_dir = cfg.out_root / "pooled"
        if is_main:
            safe_mkdir(work_dir)

        df_tr = df[df["split"] == "train"].copy()
        df_va = df[df["split"] == "val"].copy()
        # Test set selection
        if cfg.common_test_set:
            df_te = df_test_common.copy()
        else:
            df_te = df[df["split"] == "test"].copy()  # keep full test

        # Keep pooled training strictly within the requested sites
        df_tr = df_tr[df_tr["_site_norm"].isin(sites_to_use)].copy()
        df_va = df_va[df_va["_site_norm"].isin(sites_to_use)].copy()
        if not cfg.common_test_set:
            df_te = df_te[df_te["_site_norm"].isin(sites_to_use)].copy()

        if cfg.equalize_sites:
            df_tr, info_tr = sample_equal_thirds_by_site(df_tr, min(len(df_tr), pooled_train_target), cfg.seed + 101, sites_to_use)
            df_va, info_va = sample_equal_thirds_by_site(df_va, min(len(df_va), pooled_val_target), cfg.seed + 202, sites_to_use)
        else:
            info_tr = {"target_n": int(len(df_tr)), "used_n": int(len(df_tr))}
            info_va = {"target_n": int(len(df_va)), "used_n": int(len(df_va))}

        df_pooled_used = pd.concat([df_tr, df_va, df_te], ignore_index=True)

        pooled_counts = {
            "train_used_n": int(len(df_tr)),
            "val_used_n": int(len(df_va)),
            "test_used_n": int(len(df_te)),
            "train_target_n": int(info_tr["target_n"]),
            "val_target_n": int(info_va["target_n"]),
            "train_alloc_target_by_site": info_tr.get("alloc_target_by_site", {}),
            "train_used_by_site": info_tr.get("used_by_site", {}),
            "val_alloc_target_by_site": info_va.get("alloc_target_by_site", {}),
            "val_used_by_site": info_va.get("used_by_site", {}),
        }
        sampling_plan["actual"]["pooled"] = pooled_counts

        mf = df_pooled_used[["isic_id", cfg.group_col, "_site_norm", "diagnosis", "_y", "split"]].copy()
        mf["model"] = "pooled"
        manifest_frames.append(mf)

        pooled_metrics = train_one(df_pooled_used, all_sites, cfg, work_dir, is_main, local_rank, world_size)

        if cfg.common_test_set:
            rows.append({
                "site": "common_test",
                "test_site": "all",
                "train_site": "pooled",
                "model": "pooled",
                "acc": float(pooled_metrics["acc"]),
                "f1": float(pooled_metrics["f1"]),
                "auc": float(pooled_metrics["auc"]),
                "n_train": int(pooled_metrics.get("n_train", 0)),
                "n_val": int(pooled_metrics.get("n_val", 0)),
                "n_test": int(pooled_metrics.get("n_test", 0)),
                "n_test_pos": int(pooled_metrics.get("n_test_pos", 0)),
                "n_test_neg": int(pooled_metrics.get("n_test_neg", 0)),
            })

            # Also write pooled metrics per test-site slice of the SAME common test set
            # so downstream code can build a train_model x test_site heatmap.
            if is_main:
                cross_dir = work_dir / "cross_site_eval"
                for t in sites_to_use:
                    test_rows = df_test_common[df_test_common["_site_norm"] == t].copy()
                    res = eval_saved_model_on_test_rows(
                        work_dir=work_dir,
                        test_rows=test_rows,
                        all_sites=all_sites,
                        cfg=cfg,
                        local_rank=local_rank,
                        model_name="pooled",
                        train_site="pooled",
                        test_site=str(t),
                        save_dir=cross_dir / f"test_{t}",
                    )
                    if res is None:
                        continue
                    rows.append({
                        "site": str(t),
                        "test_site": str(t),
                        "train_site": "pooled",
                        "model": "pooled",
                        "acc": float(res["acc"]),
                        "f1": float(res["f1"]),
                        "auc": float(res["auc"]),
                        "n_train": int(pooled_metrics.get("n_train", pooled_counts.get("train_used_n", 0))),
                        "n_val": int(pooled_metrics.get("n_val", pooled_counts.get("val_used_n", 0))),
                        "n_test": int(res["n_test"]),
                        "n_test_pos": int(res.get("n_test_pos", 0)),
                        "n_test_neg": int(res.get("n_test_neg", 0)),
                    })
        if not cfg.common_test_set:
            # Evaluate pooled model per site on FULL held-out test rows
            for s in sites_to_use:
                test_rows = df[(df["split"] == "test") & (df["_site_norm"] == s)]
                if len(test_rows) == 0:
                    continue

                meta_enc = MetadataEncoder(all_sites)
                ds_site_test = LesionDataset(test_rows, cfg.data_root, cfg.mask_dirname, cfg.input_size, meta_enc, group_col=cfg.group_col, augment=False)

                smp = DistributedSampler(ds_site_test, num_replicas=world_size, rank=local_rank, shuffle=False) if is_distributed else None
                dl = DataLoader(ds_site_test, batch_size=cfg.batch_size, sampler=smp, shuffle=False, num_workers=2, pin_memory=True)

                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)
                    device = torch.device("cuda", local_rank)
                else:
                    device = torch.device("cpu")

                model = ConvNeXtDual(meta_dim=meta_enc.dim).to(device)
                state = torch.load(work_dir / "model_best.pth", map_location=device)
                model.load_state_dict(state, strict=True)

                ce_weight = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
                _, acc, f1, auc, *_ = evaluate(model, dl, device, ce_weight)

                rows.append({
                    "site": str(s),
                    "model": "pooled",
                    "acc": float(acc),
                    "f1": float(f1),
                    "auc": float(auc),
                    "n_train": int(pooled_counts.get("train_used_n", 0)),
                    "n_val": int(pooled_counts.get("val_used_n", 0)),
                    "n_test": int(len(ds_site_test)),
                })

    # ------------------ Per-site models ------------------
    if cfg.site_mode in {"per-site", "both"}:
        for s in sites_to_use:
            df_site_all = df[df["_site_norm"] == s].copy()
            if len(df_site_all) == 0:
                if is_main:
                    print(f"Skipping site {s}: no rows after filtering.", flush=True)
                continue

            df_tr = df_site_all[df_site_all["split"] == "train"].copy()
            df_va = df_site_all[df_site_all["split"] == "val"].copy()
            if cfg.common_test_set:
                df_te = df_test_common.copy()
            else:
                df_te = df_site_all[df_site_all["split"] == "test"].copy()  # keep full test

            if len(df_tr) == 0 or len(df_va) == 0 or len(df_te) == 0:
                if is_main:
                    print(
                        f"Skipping site {s}: split sizes train={len(df_tr)} val={len(df_va)} test={len(df_te)}",
                        flush=True
                    )
                continue

            if cfg.equalize_sites:
                df_tr, info_tr = stratified_sample_exact(
                    df_tr, min(len(df_tr), baseline_train_n),
                    cfg.seed + 1000 + stable_hash_0_9999(f"{s}|train"),
                    ["_y"]
                )
                df_va, info_va = stratified_sample_exact(
                    df_va, min(len(df_va), baseline_val_n),
                    cfg.seed + 2000 + stable_hash_0_9999(f"{s}|val"),
                    ["_y"]
                )
            else:
                info_tr = {"target_n": int(len(df_tr)), "used_n": int(len(df_tr))}
                info_va = {"target_n": int(len(df_va)), "used_n": int(len(df_va))}

            df_site_used = pd.concat([df_tr, df_va, df_te], ignore_index=True)

            sampling_plan["actual"][f"site_{s}"] = {
                "train_used_n": int(len(df_tr)),
                "val_used_n": int(len(df_va)),
                "test_used_n": int(len(df_te)),
                "train_target_n": int(info_tr["target_n"]),
                "val_target_n": int(info_va["target_n"]),
            }

            mf = df_site_used[["isic_id", cfg.group_col, "_site_norm", "diagnosis", "_y", "split"]].copy()
            mf["model"] = f"site_{s}"
            manifest_frames.append(mf)

            work_dir = cfg.out_root / f"site_{s}"
            if is_main:
                safe_mkdir(work_dir)

            metrics = train_one(df_site_used, all_sites, cfg, work_dir, is_main, local_rank, world_size)

            rows.append({
                "site": "common_test" if cfg.common_test_set else str(s),
                "test_site": "all" if cfg.common_test_set else str(s),
                "train_site": str(s),
                "model": f"site_{s}",
                "acc": float(metrics["acc"]),
                "f1": float(metrics["f1"]),
                "auc": float(metrics["auc"]),
                "n_train": int(metrics.get("n_train", 0)),
                "n_val": int(metrics.get("n_val", 0)),
                "n_test": int(metrics.get("n_test", 0)),
                "n_test_pos": int(metrics.get("n_test_pos", 0)),
                "n_test_neg": int(metrics.get("n_test_neg", 0)),
            })

            # If using COMMON_TEST_SET, also record performance on each test-site slice
            # from the SAME shared pooled test set (cross-site heatmap ready).
            if cfg.common_test_set and is_main:
                cross_dir = work_dir / "cross_site_eval"
                for t in sites_to_use:
                    test_rows = df_test_common[df_test_common["_site_norm"] == t].copy()
                    res = eval_saved_model_on_test_rows(
                        work_dir=work_dir,
                        test_rows=test_rows,
                        all_sites=all_sites,
                        cfg=cfg,
                        local_rank=local_rank,
                        model_name=f"site_{s}",
                        train_site=str(s),
                        test_site=str(t),
                        save_dir=cross_dir / f"test_{t}",
                    )
                    if res is None:
                        continue
                    rows.append({
                        "site": str(t),
                        "test_site": str(t),
                        "train_site": str(s),
                        "model": f"site_{s}",
                        "acc": float(res["acc"]),
                        "f1": float(res["f1"]),
                        "auc": float(res["auc"]),
                        "n_train": int(metrics.get("n_train", 0)),
                        "n_val": int(metrics.get("n_val", 0)),
                        "n_test": int(res["n_test"]),
                        "n_test_pos": int(res.get("n_test_pos", 0)),
                        "n_test_neg": int(res.get("n_test_neg", 0)),
                    })
    # ------------------ Aggregated outputs ------------------
    if is_main:
        site_split_counts = (
            df[df["_site_norm"].isin(sites_to_use)]
            .groupby(["_site_norm", "split"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )

        with open(cfg.out_root / "sites_present.json", "w") as f:
            json.dump({
                "present_sites": present_sites,
                "used_sites": sites_to_use,
                "baseline_site": baseline_site_norm,
                "baseline_train_n": baseline_train_n,
                "baseline_val_n": baseline_val_n,
                "counts_by_site_and_split": site_split_counts,
            }, f, indent=2)

        if len(manifest_frames) > 0:
            manifest = pd.concat(manifest_frames, ignore_index=True)
            manifest.to_csv(cfg.out_root / "data_manifest.csv", index=False)

        with open(cfg.out_root / "sampling_plan.json", "w") as f:
            json.dump(sampling_plan, f, indent=2)

        summary_csv = cfg.out_root / "summary.csv"
        pd.DataFrame(rows).to_csv(summary_csv, index=False)
        print(f"Saved summary to {summary_csv}", flush=True)

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ------------------------------ CLI ------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Site-specific vs pooled melanoma classifiers with group-aware splits.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")

    # Data
    t.add_argument("--metadata-file", type=str, default=DEFAULTS["METADATA_FILE"])
    t.add_argument("--data-root", type=str, default=DEFAULTS["DATA_ROOT"])
    t.add_argument("--mask-dirname", type=str, default=DEFAULTS["MASK_DIRNAME"])
    t.add_argument("--site-col", type=str, default=DEFAULTS["SITE_COL"])
    t.add_argument("--group-col", type=str, default=DEFAULTS["GROUP_COL"])
    t.add_argument("--split-col", type=str, default=DEFAULTS["SPLIT_COL"])

    # Sites
    t.add_argument("--site-mode", type=str, choices=["per-site", "pooled", "both"], default=DEFAULTS["SITE_MODE"])
    t.add_argument("--sites", type=str, nargs="*", default=DEFAULTS["SITES"])

    # Train
    t.add_argument("--batch-size", type=int, default=DEFAULTS["BATCH_SIZE"])
    t.add_argument("--epochs", type=int, default=DEFAULTS["EPOCHS"])
    t.add_argument("--lr", type=float, default=DEFAULTS["LR"])
    t.add_argument("--focus-strength", type=float, default=DEFAULTS["FOCUS_STRENGTH"])
    t.add_argument("--input-size", type=int, default=DEFAULTS["INPUT_SIZE"])
    t.add_argument("--augment", action="store_true", default=DEFAULTS["AUGMENT"])
    t.add_argument("--patience", type=int, default=DEFAULTS["PATIENCE"])
    t.add_argument("--warmup-epochs", type=int, default=DEFAULTS["WARMUP_EPOCHS"])
    t.add_argument("--seed", type=int, default=DEFAULTS["SEED"])
    t.add_argument("--weight-decay", type=float, default=DEFAULTS["WEIGHT_DECAY"])

    # Splits
    t.add_argument("--test-size", type=float, default=DEFAULTS["TEST_SIZE"])
    t.add_argument("--val-size", type=float, default=DEFAULTS["VAL_SIZE"])

    # System
    t.add_argument("--ddp", action="store_true", default=DEFAULTS["DDP"])
    t.add_argument("--out-root", type=str, default=DEFAULTS["OUT_ROOT"])

    # Data-efficiency controls
    t.add_argument("--equalize-sites", dest="equalize_sites", action="store_true", default=DEFAULTS["EQUALIZE_SITES"])
    t.add_argument("--no-equalize-sites", dest="equalize_sites", action="store_false")
    t.add_argument("--baseline-site", type=str, default=DEFAULTS["BASELINE_SITE"])
    t.add_argument("--pooled-multiplier", type=int, default=DEFAULTS["POOLED_MULTIPLIER"])

    # Output controls
    t.add_argument("--save-confusion", action="store_true", default=DEFAULTS["SAVE_CONFUSION"])
    t.add_argument("--save-test-predictions", dest="save_test_predictions", action="store_true", default=DEFAULTS["SAVE_TEST_PREDICTIONS"])
    t.add_argument("--no-save-test-predictions", dest="save_test_predictions", action="store_false")
    t.add_argument("--save-cross-site-predictions", dest="save_cross_site_predictions", action="store_true", default=DEFAULTS["SAVE_CROSS_SITE_PREDICTIONS"])
    t.add_argument("--no-save-cross-site-predictions", dest="save_cross_site_predictions", action="store_false")

    args = ap.parse_args()

    if args.cmd == "train":
        # timestamped run folder
        out_root = Path(args.out_root) / pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")

        cfg = TrainConfig(
            data_root=Path(args.data_root),
            metadata_file=Path(args.metadata_file),
            mask_dirname=str(args.mask_dirname),

            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            focus_strength=float(args.focus_strength),
            input_size=int(args.input_size),
            augment=bool(args.augment),
            patience=int(args.patience),
            warmup_epochs=int(args.warmup_epochs),

            out_root=out_root,

            site_mode=str(args.site_mode),
            sites=args.sites,

            site_col=str(args.site_col),
            group_col=str(args.group_col),
            split_col=args.split_col if args.split_col not in [None, "None", ""] else None,

            test_size=float(args.test_size),
            val_size=float(args.val_size),
            common_test_set=bool(DEFAULTS["COMMON_TEST_SET"]),

            seed=int(args.seed),
            ddp=bool(args.ddp),

            equalize_sites=bool(args.equalize_sites),
            baseline_site=str(args.baseline_site),
            pooled_multiplier=int(args.pooled_multiplier),

            save_confusion=bool(args.save_confusion),
            save_test_predictions=bool(args.save_test_predictions),
            save_cross_site_predictions=bool(args.save_cross_site_predictions),
        )

        run(cfg)


if __name__ == "__main__":
    main()
