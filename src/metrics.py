"""Shared binary-classification metrics.

Convention: class 1 = tampered (positive class).

FAR (false accept rate) = fraction of **tampered** documents accepted as
genuine, i.e. predicted genuine.  FAR = 1 - recall(tampered).
FRR (false reject rate) = fraction of genuine documents flagged as tampered.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score,
                             average_precision_score, roc_auc_score)


def predict_labels(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_prob >= threshold).astype(int)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Precision/recall/F1/FAR/FRR/accuracy at `threshold`.

    All rates use the tampered class as positive.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = predict_labels(y_prob, threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_tampered = int((y_true == 1).sum())
    n_genuine = int((y_true == 0).sum())
    far = float(fn / n_tampered) if n_tampered else float("nan")
    frr = float(fp / n_genuine) if n_genuine else float("nan")

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "far": far,
        "frr": frr,
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_genuine": n_genuine,
        "n_tampered": n_tampered,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if n_tampered and n_genuine:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def format_metrics(m: dict, title: str = "") -> str:
    head = f"{title}\n" if title else ""
    return (head +
            f"  n={m['n']}  (genuine {m['n_genuine']}, tampered {m['n_tampered']})\n"
            f"  accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  "
            f"recall={m['recall']:.4f}  f1={m['f1']:.4f}\n"
            f"  FAR={m['far']:.4f} (tampered accepted as genuine)  "
            f"FRR={m['frr']:.4f} (genuine rejected)  @threshold={m['threshold']:.2f}\n"
            f"  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
            f"confusion TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
