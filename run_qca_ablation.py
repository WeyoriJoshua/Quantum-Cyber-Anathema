from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_config import CONFIG
from qca.ablation import run_ablation


def parse_args():
    p = argparse.ArgumentParser(description="Run validation-selected QCA component ablation on the matched benchmark.")
    p.add_argument("--data", default=str(CONFIG.benchmark_splits_path))
    p.add_argument("--out", default=str(CONFIG.project_root / "results" / "qca_ablation" / CONFIG.experiment_tag))
    p.add_argument("--qca-config", default=str(CONFIG.project_root / "results" / "qca" / CONFIG.experiment_tag / "selected_qca_config.json"))
    p.add_argument("--beta-uncertainty", type=float, default=None)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--cycles", type=int, default=CONFIG.qca_cycles)
    p.add_argument("--epochs-per-cycle", type=int, default=CONFIG.qca_epochs_per_cycle)
    p.add_argument("--warmup-epochs", type=int, default=CONFIG.qca_warmup_epochs)
    p.add_argument("--warmup-patience", type=int, default=CONFIG.qca_warmup_patience)
    p.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    p.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    p.add_argument("--memory-capacity", type=int, default=600)
    p.add_argument("--seed", type=int, default=CONFIG.seed)
    return p.parse_args()


def main():
    args = parse_args()
    beta = args.beta_uncertainty
    if beta is None:
        config_path = Path(args.qca_config)
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            beta = float(cfg["beta_uncertainty"])
            print(f"Loaded beta_uncertainty={beta} from {config_path}")
        else:
            raise FileNotFoundError(
                "QCA selected configuration not found. Run run_qca.py first, "
                "or pass --beta-uncertainty explicitly."
            )

    summary = run_ablation(
        args.data, args.out,
        beta_uncertainty=float(beta),
        layers=args.layers,
        cycles=args.cycles,
        epochs_per_cycle=args.epochs_per_cycle,
        warmup_epochs=args.warmup_epochs,
        warmup_patience=args.warmup_patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        memory_capacity=args.memory_capacity,
        seed=args.seed,
        verbose=False,
    )
    cols = [
        "Ablation", "Selected Cycle", "Validation Defense Score",
        "Validation DIG", "Validation REI", "Validation QECR",
        "Test F1", "Test MCC", "Test PR-AUC", "Test ROC-AUC",
        "Test Recall", "Test Precision",
    ]
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
