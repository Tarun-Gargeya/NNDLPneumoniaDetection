"""Evaluate all trained variants on the test set and write metrics and figures to results/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

from common import (RESULTS, SHORTCUT_COLS, VARIANTS, PneumoniaNet, eval_resize, grad_cam,
                    load_data, normalize, outside_fraction)
from shortcut_check import matched_subset, metadata_auc

COLORS = {"baseline": "#2a78d6", "camguide": "#eb6834", "camguide_aug": "#1baf7a", "ours": "#eda100"}
LABELS = {"baseline": "Baseline B3", "camguide": "+ CAM guidance",
          "camguide_aug": "+ CAM + acq. aug", "ours": "Ours (+ adversary)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "figure.dpi": 130})


def run(model, imgs, msks, y, idx, dev, keep_cams=()):
    probs, feats, inside, cams = [], [], [], {}
    for i in range(0, len(idx), 32):
        b = idx[i:i + 32]
        x = eval_resize(torch.from_numpy(imgs[b]).to(dev).float().unsqueeze(1) / 255)
        m = F.interpolate(torch.from_numpy(msks[b]).to(dev).float().unsqueeze(1), size=x.shape[-1])
        yt = torch.tensor(y[b], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, fmap, pooled, _ = model(normalize(x))
            cam = grad_cam(logits.float(), fmap, yt, create_graph=False).float()
        inside.append((1 - outside_fraction(cam, m)).detach().cpu())
        probs.append(logits.float().softmax(1)[:, 1].detach().cpu())
        feats.append(pooled.float().detach().cpu())
        for j, k in enumerate(b):
            if k in keep_cams:
                c = F.interpolate(cam[j][None, None], size=x.shape[-1], mode="bilinear")[0, 0]
                cams[k] = (c / (c.max() + 1e-6)).detach().cpu().numpy()
    return torch.cat(probs).numpy(), torch.cat(feats).numpy(), torch.cat(inside).numpy(), cams


def main() -> None:
    dev = "cuda"
    df, imgs, msks = load_data()
    imgs, msks = np.asarray(imgs), np.asarray(msks)
    y = df.label.values
    va = np.where(df.split == "val")[0]
    te = np.where(df.split == "test")[0]
    test_df = df.iloc[te]
    matched = matched_subset(test_df)
    mpos = np.isin(te, matched)
    rng = np.random.default_rng(1)
    gallery = list(rng.choice(te[y[te] == 1], 3, replace=False)) + list(rng.choice(te[y[te] == 0], 2, replace=False))
    figs = RESULTS / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    rows, rocs, all_cams, cms = [], {}, {}, {}
    meta_full, meta_matched, _ = metadata_auc(df)
    rows.append(dict(variant="metadata_only", test_auc=meta_full, matched_auc=meta_matched))

    for v, (_, _, adv) in VARIANTS.items():
        ckpt = RESULTS / v / "model.pth"
        if not ckpt.exists():
            print(f"skip {v}: no checkpoint")
            continue
        model = PneumoniaNet(pretrained=False, adversary=adv).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.eval()
        pv, fv, _, _ = run(model, imgs, msks, y, va, dev)
        pt, ft, inside, cams = run(model, imgs, msks, y, te, dev, keep_cams=set(gallery))
        all_cams[v] = cams
        yt = y[te]
        pred = (pt > 0.5).astype(int)

        # threshold giving 95% recall on validation
        thr = np.sort(pv[y[va] == 1])[int(0.05 * (y[va] == 1).sum())]
        pred95 = (pt >= thr).astype(int)

        # linear probe: fit on val features, score on test
        sc = StandardScaler().fit(fv)
        r2 = {}
        for col in SHORTCUT_COLS:
            probe = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc.transform(fv), df[col].values[va])
            r2[col] = r2_score(df[col].values[te], probe.predict(sc.transform(ft)))

        # within-class correlation between the logit and image metadata
        logit = np.log(np.clip(pt, 1e-6, 1 - 1e-6) / np.clip(1 - pt, 1e-6, 1))
        rho = {col: np.mean([abs(spearmanr(logit[yt == c], test_df[col].values[yt == c])[0]) for c in (0, 1)])
               for col in SHORTCUT_COLS}

        rows.append(dict(
            variant=v,
            test_auc=roc_auc_score(yt, pt),
            matched_auc=roc_auc_score(yt[mpos], pt[mpos]),
            accuracy=accuracy_score(yt, pred), precision=precision_score(yt, pred),
            recall=recall_score(yt, pred), f1=f1_score(yt, pred),
            recall_at95thr=recall_score(yt, pred95),
            specificity_at95thr=recall_score(1 - yt, 1 - pred95),
            lung_cam_inside=float(inside.mean()),
            probe_r2_size=r2["log_area"], probe_r2_aspect=r2["aspect"],
            rho_size=rho["log_area"], rho_aspect=rho["aspect"],
        ))
        rocs[v] = roc_curve(yt, pt)[:2]
        cms[v] = confusion_matrix(yt, pred)
        print(f"{v}: test AUC {rows[-1]['test_auc']:.4f}, matched AUC {rows[-1]['matched_auc']:.4f}, "
              f"lung CAM {rows[-1]['lung_cam_inside']:.3f}, probe R2 aspect {r2['aspect']:.3f}")
        del model
        torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "metrics.csv", index=False)
    (RESULTS / "metrics.md").write_text(
        f"Test set: {len(te)} images; matched-acquisition subset: {len(matched)} images.\n\n"
        + res.round(4).to_markdown(index=False))
    print(res.round(4).to_string(index=False))
    trained = [v for v in VARIANTS if v in rocs]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for v in trained:
        fpr, tpr = rocs[v]
        auc = res.set_index("variant").loc[v, "test_auc"]
        ax.plot(fpr, tpr, color=COLORS[v], lw=2, label=f"{LABELS[v]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC, patient-level test set")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "roc.png"); plt.close(fig)

    r = res.set_index("variant").loc[trained]
    panels = [("test_auc", "Test AUC"), ("matched_auc", "AUC, shortcut removed\n(matched-acquisition test)"),
              ("lung_cam_inside", "Grad-CAM energy\ninside lungs"), ("probe_r2_aspect", "Aspect ratio recoverable\nfrom features (probe R²)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 4.2))
    for ax, (col, title) in zip(axes, panels):
        vals = r[col].values
        bars = ax.bar(range(len(trained)), vals, color=[COLORS[v] for v in trained], width=0.65,
                      edgecolor="white", linewidth=2)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{val:.3f}", ha="center",
                    va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(trained)), [LABELS[v] for v in trained], rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9, color=INK)
        if col.endswith("auc"):
            ax.set_ylim(0.9, 1.0)
        else:
            ax.set_ylim(0, 1.05 if col == "lung_cam_inside" else max(vals.max() * 1.15, 0.05))
        ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(figs / "summary.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, len(trained), figsize=(3.2 * len(trained), 3))
    for ax, v in zip(np.atleast_1d(axes), trained):
        cm = cms[v]
        ax.imshow(cm, cmap="Blues")
        for (i, j), n in np.ndenumerate(cm):
            ax.text(j, i, n, ha="center", va="center", color="white" if n > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["Normal", "Pneum."]); ax.set_yticks([0, 1], ["Normal", "Pneum."])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(LABELS[v], fontsize=9); ax.grid(False)
    fig.tight_layout(); fig.savefig(figs / "confusion.png"); plt.close(fig)

    fig, axes = plt.subplots(len(gallery), 1 + len(trained), figsize=(2.3 * (1 + len(trained)), 2.3 * len(gallery)))
    for i, k in enumerate(gallery):
        img = F.interpolate(torch.from_numpy(imgs[k]).float()[None, None], size=224)[0, 0].numpy()
        msk = F.interpolate(torch.from_numpy(msks[k]).float()[None, None], size=224)[0, 0].numpy()
        axes[i, 0].imshow(img, cmap="gray"); axes[i, 0].contour(msk, levels=[0.5], colors="#1baf7a", linewidths=1)
        axes[i, 0].set_ylabel("Pneumonia" if y[k] else "Normal", fontsize=9)
        for j, v in enumerate(trained, 1):
            axes[i, j].imshow(img, cmap="gray"); axes[i, j].imshow(all_cams[v][k], cmap="jet", alpha=0.4)
            axes[i, j].contour(msk, levels=[0.5], colors="white", linewidths=0.6)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0, 0].set_title("X-ray + lung mask", fontsize=9)
    for j, v in enumerate(trained, 1):
        axes[0, j].set_title(LABELS[v], fontsize=9)
    fig.tight_layout(); fig.savefig(figs / "gradcam_gallery.png"); plt.close(fig)
    print("figures in", figs)


if __name__ == "__main__":
    main()
