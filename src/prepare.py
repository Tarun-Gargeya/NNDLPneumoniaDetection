"""Preprocess the raw dataset into data/processed/.

index.csv   label, patient id, split and image metadata (measured on the original image)
images.npy  (N, 256, 256) uint8 grayscale
masks.npy   (N, 256, 256) uint8 lung masks (torchxrayvision PSPNet)
"""
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SIZE = 256
SEED = 42


def patient_id(name: str, label: int) -> str:
    if label == 1:
        m = re.match(r"(person\d+)_", name)
    else:
        m = re.match(r"(NORMAL2-IM-\d+|IM-\d+)", name)
    return m.group(1) if m else Path(name).stem


def collect_files() -> pd.DataFrame:
    rows, seen = [], set()
    for p in sorted(RAW.rglob("*.jpeg")):
        if "__MACOSX" in p.parts:
            continue
        cls = p.parent.name.upper()
        if cls not in ("NORMAL", "PNEUMONIA"):
            continue
        key = (cls, p.name)
        if key in seen:  # the zip contains a duplicate copy of the dataset
            continue
        seen.add(key)
        label = int(cls == "PNEUMONIA")
        rows.append(dict(path=str(p.relative_to(ROOT)), file=p.name, label=label,
                         patient=patient_id(p.name, label), orig_split=p.parent.parent.name))
    return pd.DataFrame(rows)


def patient_split(df: pd.DataFrame) -> pd.Series:
    """70/15/15 split by patient, stratified by class."""
    rng = np.random.default_rng(SEED)
    split_of = {}
    for label in (0, 1):
        pats = df.loc[df.label == label, "patient"].unique()
        rng.shuffle(pats)
        n = len(pats)
        a, b = int(0.70 * n), int(0.85 * n)
        split_of.update({p: "train" for p in pats[:a]})
        split_of.update({p: "val" for p in pats[a:b]})
        split_of.update({p: "test" for p in pats[b:]})
    return df.patient.map(split_of)


def load(path: str):
    g = cv2.imread(str(ROOT / path), cv2.IMREAD_GRAYSCALE)
    h, w = g.shape
    meta = dict(h=h, w=w, log_area=float(np.log(h * w)), aspect=w / h,
                brightness=float(g.mean() / 255), contrast=float(g.std() / 255))
    small = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    seg_in = cv2.resize(g, (512, 512), interpolation=cv2.INTER_AREA)
    return meta, small, seg_in


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect_files()
    df["split"] = patient_split(df)
    print(f"{len(df)} images, {df.patient.nunique()} patients")
    print(df.groupby(["split", "label"]).size().unstack())

    device = "cuda"
    seg = xrv.baseline_models.chestx_det.PSPNet().to(device).eval()
    lung_idx = [seg.targets.index("Left Lung"), seg.targets.index("Right Lung")]

    n = len(df)
    images = np.zeros((n, SIZE, SIZE), np.uint8)
    masks = np.zeros((n, SIZE, SIZE), np.uint8)
    metas = []
    bs = 16
    with ThreadPoolExecutor(8) as pool:
        loaded = pool.map(load, df.path)
        batch, batch_idx = [], []

        def flush():
            x = np.stack([xrv.datasets.normalize(s, 255) for s in batch])[:, None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                logits = seg(torch.from_numpy(x).float().to(device))[:, lung_idx]
            m = (logits.float().sigmoid() > 0.5).any(1, keepdim=True).float()
            m = torch.nn.functional.interpolate(m, size=SIZE, mode="nearest")
            masks[batch_idx] = m[:, 0].byte().cpu().numpy()
            batch.clear(); batch_idx.clear()

        for i, (meta, small, seg_in) in enumerate(tqdm(loaded, total=n, desc="prepare")):
            metas.append(meta)
            images[i] = small
            batch.append(seg_in); batch_idx.append(i)
            if len(batch) == bs:
                flush()
        if batch:
            flush()

    df = pd.concat([df, pd.DataFrame(metas)], axis=1)
    df["lung_frac"] = masks.reshape(n, -1).mean(1)
    # use a full mask where segmentation failed
    bad = df.lung_frac < 0.05
    masks[bad.values] = 1
    print(f"lung masks: median coverage {df.lung_frac.median():.2f}, {bad.sum()} fallbacks")

    df.to_csv(OUT / "index.csv", index=False)
    np.save(OUT / "images.npy", images)
    np.save(OUT / "masks.npy", masks)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
