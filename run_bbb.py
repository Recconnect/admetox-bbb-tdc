"""
TDC BBB Benchmark — CatBoost Submission
=========================================

ADMETox.AI submission for the TDC ADMET Leaderboard (BBB_Martins).

Pipeline:
  Features: MapLight (Morgan r=2 count 1024 + Avalon 1024 + ErG 315 + RDKit2D 217) = 2580d
  Model:    CatBoost (iterations=1000, subsample=0.5, sampling_frequency=PerTree)
  Protocol: Train on all train_val (MapLight protocol), no validation, no early stopping
  Seeds:    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

Result:
  AUROC = 0.9257 +/- 0.0022 (15-seed) vs TDC SOTA 0.924 (MapLight+GNN)
  TDC evaluate: 0.9260 +/- 0.0020

Usage:
  python run_bbb.py                       # Run with default MapLight protocol, 15 seeds
  python run_bbb.py --seeds 1,2,3,4,5     # Custom seeds

Requirements:
  pip install -r requirements.txt

References:
  - TDC: https://tdcommons.ai/benchmark/admet_group/overview/
  - MapLight: https://arxiv.org/abs/2310.00174
  - BBB dataset: Martins et al., 2012
"""
import os
import sys
import json
import time
import argparse
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from rdkit.DataStructs import ConvertToNumpyArray

import catboost as cb
from tdc.benchmark_group import admet_group

for ch in ["rdApp.info", "rdApp.warning", "rdApp.error", "rdApp.debug"]:
    RDLogger.DisableLog(ch)


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> str:
        s = time.time() - self.t0
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s"

    def ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")


timer = Timer()
log = print


def compute_features(smiles_list: list[str]) -> np.ndarray:
    """MapLight 2580d: Morgan r=2 count (1024) + Avalon count (1024) + ErG (315) + RDKit2D (217)."""
    n = len(smiles_list)
    morgan_gen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)

    X_morgan = np.zeros((n, 1024), dtype=np.float32)
    X_avalon = np.zeros((n, 1024), dtype=np.float32)
    X_erg = np.zeros((n, 315), dtype=np.float32)

    for i, sm in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            ConvertToNumpyArray(morgan_gen.GetCountFingerprint(mol), X_morgan[i])
            try:
                ConvertToNumpyArray(pyAvalonTools.GetAvalonCountFP(mol, nBits=1024), X_avalon[i])
            except Exception:
                pass
            try:
                X_erg[i] = GetErGFingerprint(mol).astype(np.float32)
            except Exception:
                pass

    desc_list = Descriptors._descList
    X_rdkit = np.zeros((n, len(desc_list)), dtype=np.float32)
    for i, sm in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            for j, (_, func) in enumerate(desc_list):
                try:
                    val = func(mol)
                    if val is not None and np.isfinite(val):
                        X_rdkit[i, j] = float(val)
                except Exception:
                    pass

    return np.hstack([X_morgan, X_avalon, X_erg, X_rdkit])


def train_predict(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, seed: int) -> np.ndarray:
    """Train CatBoost and return test predictions."""
    model = cb.CatBoostClassifier(
        iterations=1000,
        random_strength=2,
        subsample=0.5,
        sampling_frequency="PerTree",
        loss_function="Logloss",
        random_seed=seed,
        verbose=0,
        thread_count=1,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def run_tdc_evaluation(group, benchmark, seeds: list[int]) -> dict:
    """Run full TDC evaluation with multiple seeds."""
    name = benchmark["name"]
    train_val = benchmark["train_val"]
    test = benchmark["test"]

    all_smiles = list(train_val["Drug"].values) + list(test["Drug"].values)
    n_tv = len(train_val)
    y_tv = train_val["Y"].values.astype(int)
    y_te = test["Y"].values.astype(int)

    log(f"\n[{timer.ts()}] Dataset: {name}")
    log(f"  Train+Val: {n_tv} ({y_tv.mean():.1%} positive)")
    log(f"  Test:      {len(test)} ({y_te.mean():.1%} positive)")

    log(f"\n[{timer.ts()}] Computing features (MapLight 2580d)...")
    log(f"  Morgan(1024) + Avalon(1024) + ErG(315) + RDKit2D(217)")
    X_all = compute_features(all_smiles)
    X_tv = X_all[:n_tv]
    X_te = X_all[n_tv:]
    log(f"  Features: {X_all.shape[1]}d ({timer.elapsed()})")

    predictions_list = []
    individual_aurocs = []

    log(f"\n{'─' * 60}")
    log(f"[{timer.ts()}] Training CatBoost x {len(seeds)} seeds")
    log(f"{'─' * 60}")

    for seed in seeds:
        y_pred = train_predict(X_tv, y_tv, X_te, seed)

        auroc = roc_auc_score(y_te, y_pred)
        individual_aurocs.append(auroc)
        predictions_list.append({name: y_pred})

        log(f"  [{timer.ts()}] Seed {seed}: AUROC = {auroc:.4f}")

    if len(predictions_list) >= 5:
        results = group.evaluate_many(predictions_list)
    else:
        results = {"note": f"Need >= 5 seeds for TDC leaderboard (have {len(predictions_list)})"}
    ens_auroc = np.mean(individual_aurocs)
    std_auroc = np.std(individual_aurocs)

    avg_preds = np.mean([p[name] for p in predictions_list], axis=0)
    ens_auprc = average_precision_score(y_te, avg_preds)
    ens_f1 = f1_score(y_te, (avg_preds >= 0.5).astype(int))

    return {
        "name": name,
        "protocol": "maplight",
        "tdc_results": results,
        "ensemble_auroc": float(ens_auroc),
        "std_auroc": float(std_auroc),
        "ensemble_auprc": float(ens_auprc),
        "ensemble_f1": float(ens_f1),
        "individual_aurocs": [float(a) for a in individual_aurocs],
        "seeds": seeds,
        "n_features": X_all.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(
        description="TDC BBB Benchmark — CatBoost Submission"
    )
    parser.add_argument(
        "--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        help="Comma-separated random seeds (default: 1-15)"
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    log("=" * 60)
    log("TDC BBB Benchmark — CatBoost Submission")
    log("  ADMETox.AI")
    log("=" * 60)
    log(f"  Features:    MapLight (Morgan 1024 + Avalon 1024 + ErG 315 + RDKit2D 217) = 2580d")
    log(f"  Model:       CatBoost (iter=1000, rs=2, subsample=0.5, PerTree)")
    log(f"  Seeds:       {seeds}")
    log("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    group = admet_group(path=str(DATA_DIR))
    benchmark = group.get("BBB_Martins")

    results = run_tdc_evaluation(group, benchmark, seeds)

    tdc_str = ""
    if isinstance(results["tdc_results"], dict) and "note" not in results["tdc_results"]:
        tdc_key = results["name"]
        if tdc_key in results["tdc_results"]:
            tdc_mean = results["tdc_results"][tdc_key][0]
            tdc_std = results["tdc_results"][tdc_key][1]
            tdc_str = f"{tdc_mean:.4f} +/- {tdc_std:.4f}"

    sota = 0.924

    log(f"\n{'=' * 60}")
    log("RESULTS")
    log(f"{'=' * 60}")
    log(f"  TDC SOTA:       {sota} (MapLight+GNN)")
    log(f"  Individual:     {[f'{a:.4f}' for a in results['individual_aurocs']]}")
    log(f"  Mean +/- Std:   {results['ensemble_auroc']:.4f} +/- {results['std_auroc']:.4f}")
    log(f"  Ensemble AUROC: {results['ensemble_auroc']:.4f}")
    log(f"  Ensemble AUPRC: {results['ensemble_auprc']:.4f}")
    log(f"  Ensemble F1:    {results['ensemble_f1']:.4f}")
    log(f"  TDC evaluate:   {tdc_str}")
    log(f"  Features:       2580d (MapLight)")
    log(f"  Model:          CatBoost (subsample=0.5, sampling_frequency=PerTree)")

    if results["ensemble_auroc"] > sota:
        log(f"\n  *** BEAT SOTA {sota} by +{results['ensemble_auroc'] - sota:.4f} ***")
    else:
        log(f"\n  Gap to SOTA: {sota - results['ensemble_auroc']:.4f}")

    log(f"  Time: {timer.elapsed()}")
    log(f"{'=' * 60}")

    out_file = OUTPUT_DIR / "bbb_results.json"
    results["timestamp"] = datetime.now().isoformat()
    results["time"] = timer.elapsed()
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved: {out_file}")


if __name__ == "__main__":
    main()
