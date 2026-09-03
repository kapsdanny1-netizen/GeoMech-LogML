#!/usr/bin/env python3
"""Command-line end-to-end pipeline: generate (or load) data → train all models
under well-wise CV → validate uncertainty → write metrics, ablation and report.

Examples
--------
    python scripts/train_eval.py                          # synthetic, defaults
    python scripts/train_eval.py --wells 10 --seed 7      # more wells
    python scripts/train_eval.py --csv path/to/all_wells.csv
    python scripts/train_eval.py --out-dir outputs/run1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geomech_logml.config import GLOBAL_SEED
from geomech_logml.data.synthetic import SyntheticConfig, generate_dataset
from geomech_logml.data.las_io import load_any
from geomech_logml.pipeline import (
    ExperimentConfig,
    interval_coverage_summary,
    run_ablation,
    run_experiment,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="GeoMech-LogML end-to-end training/evaluation")
    ap.add_argument("--csv", type=str, default=None,
                    help="optional CSV/LAS file with wells (else synthetic data)")
    ap.add_argument("--wells", type=int, default=8, help="number of synthetic wells")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    ap.add_argument("--step", type=float, default=0.5, help="synthetic depth step (m)")
    ap.add_argument("--feature-set", type=str, default="eng_with_vp")
    ap.add_argument("--cv", type=str, default="well_kfold",
                    choices=["well_kfold", "leave_one_well_out"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--out-dir", type=str, default="outputs/run")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---- data -----------------------------------------------------------------
    if args.csv:
        df = load_any(args.csv)
        print(f"Loaded {df['WELL'].nunique()} wells from {args.csv}")
    else:
        cfg_syn = SyntheticConfig(n_wells=args.wells, step_m=args.step, seed=args.seed)
        df = generate_dataset(cfg_syn)
        print(f"Generated {cfg_syn.n_wells} synthetic wells "
              f"({len(df)} rows, seed={args.seed})")

    # ---- experiment -------------------------------------------------------------
    exp_cfg = ExperimentConfig(feature_set=args.feature_set, cv_strategy=args.cv,
                               n_splits=args.n_splits, alpha=args.alpha, seed=args.seed)
    print("Running well-wise CV for all models ...")
    res = run_experiment(df, exp_cfg)

    res.metrics.to_csv(out / "metrics_pooled_oof.csv", index=False)
    for key, cv in res.cv.items():
        cv.fold_metrics.to_csv(out / f"fold_metrics_{key}.csv", index=False)
    res.curves.to_csv(out / "curve_predictions.csv", index=False)
    if res.honest_conformal is not None:
        interval_coverage_summary(res).to_csv(out / "interval_coverage.csv", index=False)

    print("\n=== Pooled out-of-fold (blind-well) metrics ===")
    print(res.metrics.to_string(index=False))
    if res.honest_conformal is not None:
        print("\n=== Prediction-interval validation (honest, out-of-well) ===")
        print(interval_coverage_summary(res).to_string(index=False))

    # ---- ablation ---------------------------------------------------------------
    if not args.no_ablation:
        print("\nRunning feature-set ablation (with vs without Vp) ...")
        abl = run_ablation(df, exp_cfg)
        abl.to_csv(out / "ablation_vp.csv", index=False)
        print(abl.round(3).to_string(index=False))

    with open(out / "config.json", "w") as fh:
        json.dump(res.config.__dict__, fh, indent=2, default=str)
    print(f"\nArtifacts written to {out.resolve()}")


if __name__ == "__main__":
    main()
