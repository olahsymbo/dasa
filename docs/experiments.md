# Experiments

Datasets, splits, models, the seven compared methods, the training protocol, and the
full config reference.

---

## Datasets

Both datasets download automatically on first use into `data_root` (default `./data`).

### Oxford-IIIT Pet — `pet`

Pet photographs with pixel-level trimap annotations. Chosen because many images contain
fine fur boundaries, substantial pose variation, and large differences in object scale —
the kind of heterogeneity that makes uniform augmentation wasteful.

The raw trimap labels are remapped in `PetSegDataset._mask`:

| Raw trimap value | Meaning | Class id used here |
| --- | --- | --- |
| 3 | boundary / unclassified | **0** |
| 1 | pet foreground | **1** |
| 2 | background | **2** |

So class 0 is the boundary band, and it is consistently the hardest — every result CSV
shows `class_0_IoU` well below the other two. That is expected, and it is why the paper
reports class-wise IoU alongside the means.

- Train pool: official `trainval` split
- Test: official `test` split
- Classes: 3

### Pascal VOC 2012, binary — `voc_binary`

VOC segmentation labels collapsed to foreground vs. background in
`VOCBinarySegDataset._mask`: any annotated object class becomes foreground (1), and both
true background and the 255 void border become background (0).

This creates a strong pixel imbalance in favour of background, which makes it a useful
stress test — a model can post a respectable mean score while badly missing objects.
That is why foreground IoU (`class_1_IoU`) is the headline metric here rather than mIoU.

- Train pool: `image_set="train"`
- Test: `image_set="val"`
- Classes: 2

### Splits

Both datasets follow the same scheme in `run_one`:

```python
ids = list(range(len(base)))                       # optionally truncated by max_samples
tr, va = train_test_split(ids, test_size=0.15, random_state=cfg["seed"])
```

- **Train (85%)** — used for training and for difficulty estimation
- **Validation (15%)** — evaluated every epoch, written to `history_*.csv`; not used for
  early stopping or model selection, only for monitoring
- **Test** — the dataset's own held-out split, evaluated once at the end, written to
  `result_*.csv`

Validation is loaded through `val_clean()`, an un-augmented view of the training pool.

---

## Models

All three are built through `build_model(name, num_classes)` and expose the same
interface: take `(B, 3, H, W)`, return per-pixel logits `(B, C, H, W)` at input
resolution.

| `models` entry | Architecture | Params (3-class) | Notes |
| --- | --- | --- | --- |
| `unet` | `SmallUNet` — 3-level encoder/decoder, 32→256 channels | 1.93 M | Trained from scratch. `Dropout2d(0.1)` in every `DoubleConv` |
| `deeplabv3` | `deeplabv3_resnet50`, ImageNet-pretrained | 42.0 M | Final classifier conv replaced for the class count; `["out"]` unwrapped |
| `segformer_b0` | `nvidia/segformer-b0-finetuned-ade-512-512` | 3.71 M | Head reinitialized via `ignore_mismatched_sizes=True`; logits bilinearly upsampled to input size |

The three cover distinct model families — compact conv encoder-decoder, strong conv with
atrous context, and lightweight transformer — which is what lets the paper claim the
method is architecture-agnostic rather than tuned to one representation type.

SegFormer-B0 gets its own batch size (`segformer_batch_size`) because its memory demand
is higher than its parameter count suggests.

---

## The seven methods

Every method draws from the same base transformation family. Only the allocation of
strength changes.

| Method | Adaptive? | Allocation |
| --- | --- | --- |
| `baseline` | no | Uniform `s = 0.5` for all samples |
| `strong_uniform` | no | Uniform `s = 1.0` for all samples |
| `random_weighted` | yes | Random per-sample weights — non-uniform but difficulty-blind |
| `loss_only` | yes | Per-sample cross-entropy |
| `rarity_only` | yes | Inverse class frequency |
| `boundary_only` | yes | Mask contour density |
| `DASA_EM` | yes | All four signals, 0.35 / 0.35 / 0.20 / 0.10 |

**`baseline` is not "no augmentation."** It is medium-strength augmentation applied
uniformly — flip at `p = 0.5`, then rotation/translation/colour jitter at `p = 0.6` with
half-magnitude parameters. It is the standard fixed pipeline the paper compares against,
not an unaugmented control.

**`strong_uniform`** is the "just turn augmentation up globally" control. It is the most
informative negative result in the paper: it helps DeepLabV3 (+0.083 mIoU on Pet) and
hurts both U-Net (−0.028) and SegFormer-B0 (−0.025).

**`random_weighted`** isolates whether *any* non-uniform allocation helps, or whether the
difficulty information specifically is doing the work.

**`loss_only` / `rarity_only` / `boundary_only`** are single-factor ablations motivated
by hard-example mining, class rebalancing, and boundary-aware objectives respectively.
They are ablations of DASA, not reimplementations of those published methods.

Adaptive methods run the warm-up → estimate → train loop described in
[method.md](method.md#5-the-training-loop); non-adaptive methods just train straight
through for `baseline_epochs`.

---

## Training protocol

Shared across all methods:

- Optimizer: `AdamW(lr, weight_decay)`
- Loss: `nn.CrossEntropyLoss()`, unweighted
- No LR schedule, no early stopping, no gradient clipping, no mixed precision
- Validation evaluated after every epoch
- Test evaluated once, after all training
- `seed_all(cfg["seed"])` is called at the start of every single run, so each
  `(dataset, model, method)` cell starts from the same seed

Metrics come from `evaluate` in [src/metrics.py](../src/metrics.py): per-class IoU and
Dice are computed **per sample**, then averaged over samples with `np.nanmean`. Classes
absent from both prediction and ground truth for a given sample contribute `nan` and are
skipped rather than counted as zero. This is a per-image macro average, not a
dataset-level confusion-matrix IoU — the numbers are not directly comparable to
benchmark leaderboards that accumulate intersections and unions globally.

The driver loops `dataset × model × method`, catches CUDA OOM per cell (records an
`error: OOM` row and continues), and rewrites `ALL_RESULTS_SUMMARY.csv` after every cell
so a run that dies partway still leaves usable results behind.

---

## Config reference

Every key, its meaning, and its value in each shipped config.

| Key | Meaning | `smoke` | `paper_lite` | `full_optional` |
| --- | --- | --- | --- | --- |
| `seed` | Seed for split, numpy, torch | 42 | 42 | 42 |
| `data_root` | Dataset download/cache dir | `./data` | `./data` | `./data` |
| `output_root` | Root for csv/checkpoints/figures/logs | `./outputs` | `./outputs` | `./outputs` |
| `dataset` / `datasets` | Which datasets to run | `pet` | `[pet, voc_binary]` | `[pet, voc_binary]` |
| `image_size` | Square resize applied to image and mask | 96 | 128 | 192 |
| `max_samples` | Truncate the train pool (debug aid) | 300 | `null` | `null` |
| `models` | Architectures to run | `[unet]` | all three | all three |
| `methods` | Strategies to run | `[baseline, DASA_EM]` | all seven | all seven |
| `batch_size` | Batch size for U-Net and DeepLabV3 | 16 | 40 | 12 |
| `segformer_batch_size` | Batch size override for SegFormer-B0 | — | 8 | 4 |
| `baseline_epochs` | Epochs for non-adaptive methods, and warm-up epochs for adaptive ones | 1 | 10 | 20 |
| `em_rounds` | Adaptation rounds `M` | 1 | 3 | 4 |
| `epochs_per_m_step` | Training epochs per round | 1 | 3 | 5 |
| `lr` | AdamW learning rate | 1e-3 | 1e-3 | 5e-4 |
| `weight_decay` | AdamW weight decay | 1e-4 | 1e-3 | 1e-4 |
| `mc_passes` | Stochastic passes `K` for ambiguity | 2 | 4 | 6 |
| `budget_strength` | Sampling weight scale — `w = 1 + budget·d` | 3.0 | 4.0 | 4.0 |
| `num_workers` | DataLoader workers | 2 | 4 | 4 |
| `save_checkpoints` | Write `.pt` per cell | true | true | true |

Note `dataset` (singular) vs `datasets` (plural): the driver reads
`cfg.get("datasets", [cfg.get("dataset", "pet")])`, so either form works. `smoke` uses
the singular.

### Grid size

`len(datasets) × len(models) × len(methods)` cells:

- `smoke` — 1 × 1 × 2 = **2 cells**
- `paper_lite` — 2 × 3 × 7 = **42 cells**
- `full_optional` — 2 × 3 × 7 = **42 cells**, each substantially more expensive

### Configs vs. the paper protocol

The paper's stated setup is: image size 128×128, **20** baseline epochs, 3 rounds × 3
epochs, lr 1e-3, weight decay 1e-3, `K = 4` MC passes, batch size **20** for
U-Net/DeepLabV3 and 8 for SegFormer-B0.

`paper_lite` matches on image size, rounds, epochs-per-round, learning rate, weight
decay, MC passes, and the SegFormer batch size. It **differs** on two:

| Setting | Paper | `paper_lite` |
| --- | --- | --- |
| `baseline_epochs` | 20 | 10 |
| `batch_size` (U-Net, DeepLabV3) | 20 | 40 |

Both differences reduce cost, and both affect the numbers — halving warm-up epochs while
doubling batch size means fewer gradient steps before adaptation begins. This is why the
CSVs in `outputs/csv/` do not reproduce the paper's tables (see
[outputs.md](outputs.md#what-is-currently-in-outputscsv)).

`full_optional` is *not* the paper protocol either — it raises image size to 192, adds a
fourth round, and lowers the learning rate. It is a larger, more expensive exploration,
not a reproduction target.

**To reproduce the paper exactly**, copy `paper_lite` and set `baseline_epochs: 20` and
`batch_size: 20`:

```yaml
# configs/paper_exact.yaml
seed: 42
data_root: ./data
output_root: ./outputs
datasets: [pet, voc_binary]
image_size: 128
max_samples: null
models: [unet, deeplabv3, segformer_b0]
methods: [baseline, strong_uniform, random_weighted, loss_only, rarity_only, boundary_only, DASA_EM]
batch_size: 20
segformer_batch_size: 8
baseline_epochs: 20
em_rounds: 3
epochs_per_m_step: 3
lr: 0.001
weight_decay: 0.001
mc_passes: 4
budget_strength: 4.0
num_workers: 4
save_checkpoints: true
```

```bash
python -m src.run_experiments --config configs/paper_exact.yaml
```

Expect it to be slower than `paper_lite` — roughly double the warm-up cost per cell, plus
smaller batches.
