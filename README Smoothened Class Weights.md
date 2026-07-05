## Smoothened Class Weights Variant (`Smoothened_Class_Weights.ipynb`)

### Overview

This notebook is a variant of `ShipsEar.ipynb` that refines how class imbalance is handled during training. The preprocessing pipeline, DWSTr architecture (DWS block + SpecAugment + 6-block Transformer encoder), and evaluation cells are unchanged. The key differences are concentrated in the **training cell (Cell 13)**, where the class-weighting strategy and batch size were revised, yielding a noticeably stronger and more stable model.

### What Changed vs. `ShipsEar.ipynb`

| Aspect | Original (`ShipsEar.ipynb`) | Smoothened (`Smoothened_Class_Weights.ipynb`) |
|---|---|---|
| Class weighting | Standard `compute_class_weight('balanced', ...)` applied directly | Standard balanced weights are computed, then **square-root smoothed** before use |
| Batch size | 256 | **64** (lowered to avoid Colab GPU OOM errors) |
| Epochs run before early stopping | ~14 | **52** (training continued much longer and kept improving) |
| Test accuracy | 94.89% | **96.88%** |
| Test loss | 0.1518 | **0.0954** |

### Square-Root Weight Smoothing

The standard `'balanced'` class-weighting formula:

```
w_i = n_samples / (n_classes × n_samples_i)
```

produces very large weights for the rarest classes (e.g. Pilot Ship, Dredger). While this helps the model pay attention to minority classes, excessively large weights can cause **gradient spikes** during training — a single batch containing a few rare-class samples can dominate the loss and destabilize optimization.

To address this, the smoothed notebook applies a square-root transform to the standard weights:

```python
standard_weights = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=y_train
)

# Square-root smoothing: compress the range of weights
smoothed_weights = [math.sqrt(w) for w in standard_weights]

class_weight_dict = dict(enumerate(smoothed_weights))
```

**Effect:** Because the square root is a concave, monotonically increasing function, it preserves the relative ordering of class weights (rare classes still get more weight than common ones) but **compresses the spread** between them. For example, a raw weight of 23.9 for the rarest class becomes ≈4.9, while a weight near 1.0 for the majority class is barely changed. This keeps gradients from minority-class samples informative without letting them overwhelm the update.

### Reduced Batch Size

The batch size was lowered from 256 to 64:

```python
BATCH_SIZE = 64  # Lowered from 256 to prevent Colab GPU OOM errors
```

Besides resolving out-of-memory errors on the T4 GPU, a smaller batch size also means **more gradient updates per epoch** (1,654 steps vs. ~414 previously), which combined with the smoothed weights produced smoother, more consistent convergence.

### Training Behavior

With the same callbacks as before (`ModelCheckpoint` on `val_accuracy`, `EarlyStopping` with `patience=5` on `val_loss`, `ReduceLROnPlateau` with `patience=3`), training now ran for **52 epochs** before early stopping triggered (vs. ~14 previously), with the learning rate automatically reduced twice (0.001 → 0.0005 → 0.00025). The best weights were restored from **epoch 47**.

Validation accuracy climbed steadily throughout training, from ~42.6% after epoch 1 to a peak of **96.86%** at epoch 47, with validation loss dropping to **0.093**.

### Final Test Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **96.88%** |
| **Test Loss** | **0.0954** |
| Best epoch (restored) | 47 (of 52 run) |
| Batch size | 64 |
| Total parameters | 1,339,866 (unchanged architecture) |

### Per-Class Performance (Smoothened Weights)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Dredger | 99.57% | 99.72% | 99.64% | 703 |
| Fishboat | 95.60% | 99.35% | 97.44% | 1,378 |
| Motorboat | 93.99% | 95.48% | 94.73% | 2,720 |
| Mussel Boat | 94.30% | 96.87% | 95.57% | 1,948 |
| Natural Ambient Noise | 98.77% | 96.89% | 97.82% | 3,057 |
| Ocean Liner | 96.16% | 98.18% | 97.16% | 2,524 |
| Passengers | 98.32% | 95.85% | 97.07% | 11,410 |
| Pilot Ship | 97.59% | 98.64% | 98.11% | 369 |
| Roro | 97.81% | 98.44% | 98.12% | 4,040 |
| Sailboat | 89.18% | 96.70% | 92.79% | 1,091 |
| Trawler | 95.11% | 97.94% | 96.51% | 437 |
| Tugboat | 95.15% | 96.19% | 95.67% | 551 |

**Macro avg F1:** 96.72% · **Weighted avg F1:** 96.89%

**Key observations:**
- Overall accuracy improved by **+1.99 percentage points** (94.89% → 96.88%) with no architecture changes.
- **Motorboat** and **Sailboat** — the two weakest classes in the original run (precision 89.94% and 83.14% respectively) — improved the most, with Sailboat's precision rising from 83.14% to 89.18% and Motorboat's F1 rising from 92.23% to 94.73%.
- Rare classes (Pilot Ship, Trawler, Dredger) remain excellent, confirming that the smoothing did not sacrifice minority-class performance for overall accuracy — it improved both simultaneously.
- The tighter precision/recall balance across all 12 classes (all F1-scores now above 92%) suggests the smoothed weighting reduced the over-correction that can occur with raw `'balanced'` weights on datasets with a wide class-imbalance range (here, ~31:1 between the largest and smallest class).

### Takeaway

When class imbalance is severe (as in ShipsEar, ranging from ~37.7% for Passengers down to ~1.2% for Pilot Ship), naively applying `sklearn`'s `'balanced'` class weights can over-correct and destabilize training. Smoothing those weights with a square root — combined with a smaller batch size for more frequent, stable gradient updates — produced a model that is both more accurate overall and more balanced across classes than the original DWSTr implementation.

---

## Multi-Run Cross-Validation Pipeline (Cell 17)

### Overview

The notebook adds a final cell — the **"Ultimate Multi-Iteration Pipeline"** — that goes beyond a single train/val/test split to answer a more important question: *how consistent is the model's performance across different data splits?* Rather than reporting a single test accuracy from one lucky (or unlucky) split, this cell retrains the DWSTr model **from scratch 5 times**, each with an independently re-shuffled train/val/test partition, and reports the **mean ± standard deviation** of test accuracy and loss across all runs.

### How It Works

```python
NUM_RUNS = 5                  # Total training cycles to execute
MAX_EPOCHS = 100
BATCH_SIZE = 64
INITIAL_LR = 0.001
TEST_SPLIT = 0.20             # 20% held out for testing
VAL_SPLIT = 0.20              # 20% of the remainder for validation
ES_PATIENCE = 5
LR_PATIENCE = 3
LR_FACTOR = 0.5
```

For each of the 5 runs, the pipeline:

1. **Recombines the full dataset.** All 151,142 preprocessed segments (`X_train`, `X_val`, `X_test` from earlier cells) are concatenated back into one pool, `X_full`/`y_full`.
2. **Re-splits with a run-specific seed.** `train_test_split` is called twice (test split, then val split) using `random_state=run` (i.e. seeds 1–5), so each run trains and tests on a *different* stratified partition of the data rather than reusing the same fixed split every time.
3. **Clones a fresh, untrained model.** `keras.models.clone_model(model)` creates a new DWSTr instance with re-initialized weights, then recompiles it with the Adam optimizer — ensuring each run starts from scratch and results aren't biased by weights carried over from a previous run.
4. **Recomputes smoothed class weights** for that run's specific training split (since the class distribution shifts slightly with each re-split).
5. **Trains with the same callback strategy** as before (`ModelCheckpoint`, `EarlyStopping` patience 5, `ReduceLROnPlateau` patience 3), using a lightweight `MinimalLogger` callback that only prints progress every 10 epochs to keep output clean.
6. **Evaluates and saves full artifacts per run** — the best checkpoint (`model_weights.keras`), a text log with the classification report, a training history plot, and a confusion matrix — all written to `mc_cross_val_results/Run_0N/`.
7. **Aggregates results** across all 5 runs into a final mean ± standard deviation for both test accuracy and test loss.

### Results Across the 5 Runs

| Run | Test Accuracy | Test Loss | Epochs to Converge (approx.) |
|---|---|---|---|
| 1 | 94.37% | 0.1603 | ~50 |
| 2 | 93.48% | 0.1836 | ~40 |
| 3 | **96.50%** | **0.1050** | ~80 |
| 4 | 94.84% | 0.1534 | ~40 |
| 5 | 91.71% | 0.2368 | ~30 |

**Final aggregated result (5 runs):**

| Metric | Value |
|---|---|
| **Mean Test Accuracy** | **94.18% ± 1.58%** |
| **Mean Test Loss** | **0.1678 ± 0.0429** |

### Why This Matters

- **Single-split results can be optimistic or pessimistic by chance.** The earlier single-run result of 96.88% test accuracy (Cell 14) falls within the observed range across the 5 independent runs (91.71%–96.50%), but sits near the high end — meaning that particular train/test split happened to be relatively favorable. The multi-run average of **94.18% ± 1.58%** is a more honest, statistically grounded estimate of the model's true generalization performance.
- **Run-to-run variance reveals training sensitivity.** A spread of ~4.8 percentage points between the best (96.50%) and worst (91.71%) run indicates that final accuracy is meaningfully affected by which samples land in train/val/test, as well as by early-stopping timing (Run 3, which trained the longest at ~80 epochs, achieved the best result, while Run 5 stopped earliest at ~30 epochs and scored the lowest).
- **This is standard practice for small-to-moderate datasets.** With only ~90 source audio files (even though segmentation expands this to 151K samples derived from a limited number of underlying recordings), a single split risks reporting a number that doesn't generalize. Reporting mean ± std over multiple runs is a more rigorous way to validate model performance and is directly comparable to k-fold cross-validation in spirit, though implemented here as repeated random stratified splits rather than strict folds.
- **Fresh model cloning avoids leakage between runs.** By calling `clone_model()` and recompiling for every run, each iteration is a true independent trial — no weights, optimizer state, or learning-rate schedule carry over from the previous run.

### Artifacts Produced

For each run, the pipeline saves a self-contained results folder:

```
mc_cross_val_results/
├── Run_01/
│   ├── model_weights.keras
│   ├── evaluation_log.txt          # Full classification report + accuracy/loss
│   ├── training_history.png        # Accuracy & loss curves
│   └── confusion_matrix.png
├── Run_02/
│   └── ...
...
└── Run_05/
    └── ...
```

This makes it possible to inspect any individual run's confusion matrix or per-class metrics after the fact, rather than only having access to the final aggregated numbers.
