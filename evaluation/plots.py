from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def save_classification_plots(
    y_true,
    y_score,
    *,
    model_name: str,
    out_dir: str | Path,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_score) >= 0.5).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    RocCurveDisplay.from_predictions(y_true, y_score, ax=axes[0], name=model_name)
    axes[0].set_title("ROC Curve")

    PrecisionRecallDisplay.from_predictions(
        y_true, y_score, ax=axes[1], name=model_name
    )
    axes[1].set_title("Precision–Recall Curve")

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Benign", "Malicious"],
        cmap="Blues",
        ax=axes[2],
        colorbar=False,
    )
    axes[2].set_title("Confusion Matrix")

    fig.suptitle(model_name)
    fig.tight_layout()
    path = out_dir / f"{model_name.lower().replace(' ', '_')}_diagnostics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_evolution_plot(history: pd.DataFrame, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in ["F1", "MCC", "PR-AUC", "Defense Score"]:
        if metric in history.columns:
            ax.plot(history["Cycle"], history[metric], marker="o", label=metric)

    ax.set_xlabel("Defensive Evolution Cycle")
    ax.set_ylabel("Score")
    ax.set_title("Quantum Cyber Anathema: Recursive Defensive Evolution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path
