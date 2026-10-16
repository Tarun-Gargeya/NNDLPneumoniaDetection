"""Train one model variant (see VARIANTS in common.py).

Phase 1 trains the head on a frozen backbone; phase 2 fine-tunes the top of the backbone.
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from common import (RESULTS, SHORTCUT_COLS, VARIANTS, PneumoniaNet, augment, eval_resize,
                    grad_cam, load_data, normalize, outside_fraction)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--epochs1", type=int, default=6)
    ap.add_argument("--epochs2", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=5e-5)
    ap.add_argument("--lam_cam", type=float, default=1.0)
    ap.add_argument("--lam_adv", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    use_cam, use_acq, use_adv = VARIANTS[args.variant]

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"
    out = RESULTS / args.variant
    out.mkdir(parents=True, exist_ok=True)

    df, images, masks = load_data()
    imgs = torch.from_numpy(np.asarray(images)).to(dev)
    msks = torch.from_numpy(np.asarray(masks)).to(dev)
    y_all = torch.tensor(df.label.values, device=dev)
    tr_idx = np.where(df.split == "train")[0]
    va_idx = np.where(df.split == "val")[0]
    s = df[SHORTCUT_COLS].values
    mu, sd = s[tr_idx].mean(0), s[tr_idx].std(0)
    s_all = torch.tensor((s - mu) / sd, device=dev, dtype=torch.float32)

    n_pos = int(df.label.values[tr_idx].sum()); n_neg = len(tr_idx) - n_pos
    class_w = torch.tensor([n_pos / n_neg, 1.0], device=dev)
    print(f"[{args.variant}] train {len(tr_idx)} (normal {n_neg}, pneumonia {n_pos}), "
          f"val {len(va_idx)}, class weights {class_w.tolist()}")

    model = PneumoniaNet(pretrained=True, adversary=use_adv).to(dev)

    def batch(idx, train):
        x = imgs[idx].float().unsqueeze(1) / 255
        m = msks[idx].float().unsqueeze(1)
        if train:
            x, m = augment(x, m, acq=use_acq)
        else:
            x = eval_resize(x)
            m = F.interpolate(m, size=x.shape[-1], mode="nearest")
        return normalize(x), m, y_all[idx], s_all[idx]

    @torch.no_grad()
    def validate():
        model.eval()
        probs, losses = [], []
        for i in range(0, len(va_idx), 64):
            x, _, y, _ = batch(va_idx[i:i + 64], False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)[0].float()
            losses.append(F.cross_entropy(logits, y, weight=class_w, reduction="sum").item())
            probs.append(logits.softmax(1)[:, 1].cpu())
        p = torch.cat(probs).numpy()
        yv = df.label.values[va_idx]
        return sum(losses) / len(va_idx), roc_auc_score(yv, p), ((p > 0.5) == yv).mean()

    history, best_auc = [], -1.0
    for phase, epochs in ((1, args.epochs1), (2, args.epochs2)):
        for p in model.backbone.parameters():
            p.requires_grad = False
        if phase == 2:
            for p in model.fine_tune_params():
                p.requires_grad = True
        groups = [{"params": model.head.parameters(), "lr": args.lr1 if phase == 1 else args.lr2}]
        if phase == 2:
            groups.append({"params": model.fine_tune_params(), "lr": args.lr2})
        if use_adv:
            groups.append({"params": model.adversary.parameters(), "lr": 1e-3})
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=2)
        steps_per_epoch = math.ceil(len(tr_idx) / args.bs)
        stale = 0

        for epoch in range(epochs):
            model.train()
            model.backbone.eval()  # keep BatchNorm stats frozen
            if phase == 2:
                for mod in (*model.backbone.blocks[-2:], model.backbone.conv_head, model.backbone.bn2):
                    mod.train()
            perm = np.random.permutation(tr_idx)
            t0, agg = time.time(), dict(cls=0.0, cam=0.0, adv=0.0)
            for step in range(steps_per_epoch):
                idx = perm[step * args.bs:(step + 1) * args.bs]
                x, m, y, s_t = batch(idx, True)
                prog = (epoch * steps_per_epoch + step) / max(1, epochs * steps_per_epoch)
                grl = 2 / (1 + math.exp(-10 * prog)) - 1 if phase == 2 else 0.0
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, fmap, _, adv = model(x, grl=grl)
                    logits = logits.float()
                    loss = cls = F.cross_entropy(logits, y, weight=class_w)
                    if use_cam:
                        cam = outside_fraction(grad_cam(logits, fmap, y, create_graph=True).float(), m).mean()
                        loss = loss + args.lam_cam * cam
                        agg["cam"] += cam.item()
                    if use_adv:
                        a = F.mse_loss(adv.float(), s_t)
                        loss = loss + args.lam_adv * a
                        agg["adv"] += a.item()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                agg["cls"] += cls.item()

            vloss, vauc, vacc = validate()
            sched.step(vloss)
            row = dict(phase=phase, epoch=epoch + 1, val_loss=vloss, val_auc=vauc, val_acc=vacc,
                       **{f"train_{k}": v / steps_per_epoch for k, v in agg.items()},
                       secs=time.time() - t0)
            history.append(row)
            print(f"  phase {phase} ep {epoch + 1:2d} | cls {row['train_cls']:.3f} cam {row['train_cam']:.3f} "
                  f"adv {row['train_adv']:.3f} | val loss {vloss:.3f} auc {vauc:.4f} acc {vacc:.3f} "
                  f"| {row['secs']:.0f}s", flush=True)
            if vauc > best_auc:
                best_auc, stale = vauc, 0
                torch.save(model.state_dict(), out / "model.pth")
            else:
                stale += 1
                if stale >= args.patience:
                    print("  early stop")
                    break
        model.load_state_dict(torch.load(out / "model.pth"))

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    print(f"[{args.variant}] best val AUC {best_auc:.4f}")


if __name__ == "__main__":
    main()
