"""How well does image metadata alone (size, aspect ratio, brightness) predict the label?

Also defines the matched test subset, in which both classes have the same
distribution of original size and aspect ratio.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import PROC, RESULTS

META = ["log_area", "aspect", "brightness", "contrast"]


def matched_subset(test: pd.DataFrame, bins: int = 5, seed: int = 0) -> pd.Index:
    """Bin by (log_area, aspect) quantiles and keep equal class counts per bin."""
    rng = np.random.default_rng(seed)
    cell = (pd.qcut(test.log_area, bins, labels=False, duplicates="drop").astype(str) + "_"
            + pd.qcut(test.aspect, bins, labels=False, duplicates="drop").astype(str))
    keep = []
    for _, g in test.groupby(cell):
        n0, n1 = g[g.label == 0], g[g.label == 1]
        k = min(len(n0), len(n1))
        if k:
            keep += list(rng.choice(n0.index, k, replace=False)) + list(rng.choice(n1.index, k, replace=False))
    return pd.Index(sorted(keep))


def metadata_auc(df: pd.DataFrame, cols=META):
    tr, te = df[df.split == "train"], df[df.split == "test"]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[cols], tr.label)
    matched = matched_subset(te)
    return (roc_auc_score(te.label, clf.predict_proba(te[cols])[:, 1]),
            roc_auc_score(te.loc[matched, "label"], clf.predict_proba(te.loc[matched, cols])[:, 1]),
            len(matched))


def main() -> None:
    df = pd.read_csv(PROC / "index.csv")
    RESULTS.mkdir(exist_ok=True)
    out = {}
    print(df.groupby("label")[META].mean().T.rename(columns={0: "normal", 1: "pneumonia"}).round(3))
    for name, cols in [("all metadata", META), ("size only", ["log_area", "aspect"]),
                       ("brightness only", ["brightness", "contrast"])]:
        full, matched, n = metadata_auc(df, cols)
        out[name] = dict(test_auc=full, matched_test_auc=matched)
        print(f"{name:16s} test AUC {full:.3f} | matched-acquisition test AUC {matched:.3f}")
    out["matched_test_size"] = n
    out["test_size"] = int((df.split == "test").sum())
    print(f"matched test subset: {n} of {out['test_size']} test images")
    (RESULTS / "shortcut_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
