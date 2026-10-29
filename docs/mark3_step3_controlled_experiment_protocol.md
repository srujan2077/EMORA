# EMORA Mark 3 Step 3 Controlled Experiment Protocol

## Status

This protocol and the accompanying scripts are implementation deliverables
only. Full training is intentionally not started until this protocol and code
are reviewed and approved.

## Objective

Compare two temporal-window construction policies with the same model,
optimizer, training budget, seed, loss, class-weight formula, batch size,
checkpoint-selection rule, and validation procedure:

- **Candidate A — metadata-aware:** use the existing train and validation
  windows unchanged.
- **Candidate B — split-at-gap:** for each target, retain the longest
  contiguous suffix ending at that target. Original IDs, segment numbers,
  embeddings, and labels are preserved. Missing observations are not
  fabricated and segment numbers are not renumbered.

The split-at-gap policy is a hypothesis, not an assumed improvement.

## Fixed configuration

The controlled runs use the existing Mark 3 temporal configuration:

| Setting | Value |
|---|---:|
| Seed | 42 |
| Batch size | 32 |
| Maximum epochs | 40 |
| Early-stopping patience | 7 |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| DataLoader workers | 0 |
| Model | `TemporalFusionMark3` |
| Hidden dimension | 256 |
| Attention heads | 8 |
| Transformer layers | 2 |
| Dropout | 0.15 |
| Loss | `BCEWithLogitsLoss(pos_weight=...)` |

For each candidate, positive class weights are computed from that candidate's
training labels only:

```text
positive_counts = train_labels.sum(axis=0)
negative_counts = len(train_labels) - positive_counts
pos_weight = clamp(negative_counts / positive_counts, min=1.0, max=10.0)
```

Validation and test labels are never used for class-weight computation.

## Training and selection procedure

1. Load only `train_windows.pt` and `val_windows.pt`.
2. Verify each candidate's train/validation video-ID sets are disjoint.
3. Set Python, NumPy, PyTorch, and CUDA seeds to 42.
4. Construct the candidate's train and validation samples.
5. Train with the fixed configuration above.
6. After each epoch, evaluate validation Macro-F1 at threshold 0.5.
7. Save the candidate checkpoint only when validation Macro-F1 improves by
   more than `1e-6`.
8. Stop after 7 epochs without improvement, or after epoch 40.
9. Load the selected candidate checkpoint.
10. Calibrate one threshold per emotion on the full validation set only,
    searching 0.05 through 0.95 in increments of 0.01. Ties prefer the
    threshold closest to 0.5.
11. Report the selected checkpoint's validation metrics at threshold 0.5 and
    with the calibrated thresholds.

The primary comparison is validation Macro-F1 and Micro-F1 at threshold 0.5.
Per-emotion F1, subset accuracy, calibrated-threshold metrics, loss curves,
best epoch, class weights, and context-length distributions are secondary
outputs.

## Leakage safeguards

- The test split is not loaded, inspected, or evaluated.
- Train and validation video IDs are checked for overlap before training.
- Class weights use training labels only.
- Checkpoint selection uses validation Macro-F1@0.5 only.
- Threshold calibration happens after checkpoint selection and uses validation
  labels only.
- No context uses a future segment.
- Candidate B never renumbers segment numbers or fabricates embeddings.
- Candidate A reads the existing windows without modifying them.
- Candidate outputs use a new experiment-specific directory and refuse to
  overwrite existing files.
- The target-only Step 2 analysis was diagnostic only: it used a model trained
  with full context and was not an independently trained baseline.

## Reproducibility checks

Each candidate report records:

- seed and all fixed hyperparameters;
- candidate name and policy definition;
- source input paths;
- train/validation sample counts;
- train/validation video counts and overlap count;
- class weights;
- context-length distributions;
- epoch-by-epoch training and validation losses;
- validation Macro-F1/Micro-F1@0.5;
- selected epoch and selected checkpoint path;
- calibrated thresholds and final validation metrics.

Run both candidates with distinct output directories:

```powershell
& '.venv\Scripts\python.exe' 'scripts\train_mark3_step3_controlled.py' `
  --candidate metadata_aware `
  --output-dir 'checkpoints\mark3_step3\metadata_aware'

& '.venv\Scripts\python.exe' 'scripts\train_mark3_step3_controlled.py' `
  --candidate split_at_gap `
  --output-dir 'checkpoints\mark3_step3\split_at_gap'
```

After both runs complete, compare reports without loading test data:

```powershell
& '.venv\Scripts\python.exe' 'scripts/compare_mark3_step3_results.py' `
  --metadata-report 'checkpoints\mark3_step3\metadata_aware\results.json' `
  --split-report 'checkpoints\mark3_step3\split_at_gap\results.json' `
  --output 'reports\mark3_step3_controlled_comparison_YYYYMMDD_HHMMSS.json'
```

## Small pilot before full training

The pilot is a smoke test only, not evidence for selecting a policy. It uses
the same code and architecture but a deliberately reduced budget and separate
output directory:

```powershell
& '.venv\Scripts\python.exe' 'scripts/train_mark3_step3_controlled.py' `
  --candidate metadata_aware `
  --output-dir 'checkpoints\mark3_step3_pilot\metadata_aware' `
  --max-epochs 1 `
  --patience 1 `
  --max-train-batches 2 `
  --max-val-batches 2
```

The pilot must be discarded for scientific comparison. Run the full
experiment only after reviewing the protocol, code, and pilot output.
