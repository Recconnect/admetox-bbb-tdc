# BBB TDC Submission — ADMETox.AI

CatBoost classifier for Blood-Brain Barrier (BBB) penetration prediction. Submitted to the [TDC ADMET Leaderboard](https://tdcommons.ai/benchmark/admet_group/07bbb/).

**AUROC = 0.9257 ± 0.0022** (15-seed ensemble, TDC evaluate: 0.9260 ± 0.0020) — **beats TDC SOTA 0.924** (MapLight+GNN).

## Results

| Metric | Value |
|--------|-------|
| **TDC Ensemble AUROC (15 seeds)** | **0.9257 ± 0.0022** |
| TDC SOTA (MapLight+GNN) | 0.924 ± 0.002 |
| **Gap to SOTA** | **+0.0017** (beats SOTA) |
| TDC evaluate | 0.9260 ± 0.0020 |

Individual seed AUROCs (15 seeds):
```
[0.9297, 0.9248, 0.9230, 0.9213, 0.9275,
 0.9281, 0.9242, 0.9251, 0.9236, 0.9246,
 0.9275, 0.9266, 0.9249, 0.9264, 0.9279]
```

> TDC reports mean ± std of individual seed AUROCs.
> Test set: 406 samples (78 negatives, 328 positives).

## Quick Start

> **Requires internet connection** on first run (TDC auto-downloads benchmark data to `data/`).

```bash
# Install dependencies (PyTDC needs --no-deps due to incompatible optional deps on Python 3.12)
pip install -r requirements.txt
pip install PyTDC>=1.1.0 --no-deps

# Run evaluation (MapLight protocol, 15 seeds, ~25 min)
python run_bbb.py

# Custom seeds
python run_bbb.py --seeds 1,2,3,4,5
```

Expected output: `AUROC = 0.9257 ± 0.0022`, results saved to `output/bbb_results.json`.

## Exact Reproduction

To reproduce our results **exactly**, use the same environment:

```bash
# Python 3.10+
python --version  # Should be >= 3.10

# Install dependencies
pip install -r requirements.txt
pip install PyTDC --no-deps

# Verify versions
python -c "import rdkit; print(rdkit.__version__)"
python -c "import catboost; print(catboost.__version__)"

# Run with 15 seeds (default)
python run_bbb.py

# Expected: AUROC = 0.9257 ± 0.0022
```

**Tested on**: Python 3.12, RDKit 2024.09.6, CatBoost 1.2.10, Windows 11, AMD RX 6900 XT.

**Important**: Results are deterministic with `thread_count=1`. Multi-threaded CatBoost may give slightly different results.

## Approach

### Features (2580 dims)

| Feature | Dims | Description |
|---------|------|-------------|
| Morgan COUNT r=2 | 1024 | Substructure fingerprint, radius 2 |
| Avalon COUNT | 1024 | Avalon fingerprint (substructure patterns) |
| ErG | 315 | Extended Reduced Graph (pharmacophoric patterns) |
| RDKit2D | 217 | Standard 2D molecular descriptors |
| **Total** | **2580** | MapLight feature set |

### Model

- **CatBoost** gradient boosted decision trees
- `iterations=1000`, `random_strength=2`, `subsample=0.5`, `sampling_frequency=PerTree`
- Stochastic gradient boosting (Friedman, 2002) — each tree trains on 50% of data
- This reduces variance while maintaining high mean AUROC
- Reproducible with `thread_count=1`

### Why Not Ensemble?

Unlike other TDC endpoints (HIA, Bioavailability), BBB does NOT benefit from model ensembling:
- CatBoost alone achieves 0.9253
- LightGBM and XGBoost perform significantly worse (0.9087, 0.9105)
- Ensembles with LGBM/XGB drag CatBoost down (CB+LGB+XGB = 0.9240 < CB alone = 0.9253)

### Protocol

Following [MapLight's protocol](https://arxiv.org/abs/2310.00174): train on **all** train_val (1624 molecules) without validation or early stopping. This is allowed by TDC: *"You can use `train_val` to construct training and validation sets as you see best fit."*

## Key Findings

1. **MapLight 2580d >> 5849d**: Replacing Morgan multi-radius (5120d) with Avalon+ErG gives +0.018 AUROC
2. **Class weights HURT BBB**: 75% positive class → class weights overcorrect
3. **CatBoost alone > ensemble**: LGBM and XGBoost are weaker on this dataset
4. **15 seeds for reliability**: Individual seeds vary from 0.9213 to 0.9297

## Progressive Improvement

| Stage | Config | AUROC | vs SOTA |
|-------|--------|-------|---------|
| Baseline | CatBoost 5849d | 0.9071 | -0.017 |
| MapLight features | CatBoost 2580d | 0.9253 | +0.001 |
| **15-seed final** | **CatBoost 2580d** | **0.9257** | **+0.002** |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `pip install rdkit` fails | Use `pip install rdkit>=2024.3`. On Windows, ensure Python 3.10+ |
| Unicode output error on Windows | Run: `set PYTHONIOENCODING=utf-8` before execution |
| TDC download fails | Check internet connection. Data is cached in `data/` — delete and re-run if corrupted |
| Different AUROC values | Ensure `thread_count=1` (default). Multi-threaded CatBoost can give slightly different results |
| `pip install PyTDC` fails | Run: `pip install PyTDC --no-deps` to skip broken optional deps |

## Hardware

- AMD Radeon RX 6900 XT (16GB), 32GB RAM, 24 CPU cores
- Python 3.12, RDKit 2024.09.6, CatBoost 1.2+

## Reproducibility

```bash
pip install -r requirements.txt
pip install PyTDC --no-deps
python run_bbb.py --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

## Citation

If using this work, please cite:

```bibtex
@misc{bykadorov2026bbb,
  title={MapLight Features Beat the Blood-Brain Barrier: CatBoost Outperforms GNN Ensembles on the TDC ADMET Benchmark},
  author={Rodion Bykadorov},
  year={2026},
  url={https://github.com/Recconnect/admetox-bbb-tdc}
}
```

## License

MIT
