# DASA: Difficulty-Aware Sample Allocation for Adaptive Data Augmentation

Reference implementation and experiment pack for:

> **Difficulty-Aware Sample Allocation for Adaptive Data Augmentation in Semantic Segmentation**
> Olasimbo Ayodeji Arigbabu, Abimbola Ismail Arigbabu
> arXiv:[2608.25710](https://arxiv.org/abs/2608.25710) \[cs.CV], 26 Aug 2026

Most augmentation pipelines apply the same transformation budget to every training
image, or adapt it from a single signal such as loss. 

DASA treats augmentation as a **resource to allocate**: it scores each training sample on four complementary
difficulty signals such as prediction ambiguity, optimization difficulty, class rarity, and
boundary complexity, combines them into a normalized score `d_i ∈ [0, 1]`, and maps
that score to a per-sample augmentation strength `s_i`. Easy samples stay lightly
perturbed; hard samples get stronger geometric and photometric transformations.

The method is architecture-agnostic, requires no changes to the segmentation network and is
evaluated here on U-Net, DeepLabV3-ResNet50, and SegFormer-B0 across Oxford-IIIT Pet
(3-class trimap) and binary Pascal VOC.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/run_smoke.sh          # ~minutes, 1 model × 2 methods, sanity check
bash scripts/run_paper_lite.sh     # the realistic full-grid run
```

Datasets download automatically to `./data` on first use via `torchvision.datasets`
(`OxfordIIITPet`, `VOCSegmentation`). A CUDA GPU is detected automatically; the code
falls back to CPU, which is impractically slow for anything beyond the smoke config.

**Run the smoke config first.** If it fails, stop the instance before burning budget on
the full grid.

Figures are generated separately, after a run has produced checkpoints and a summary CSV:

```bash
python -m src.make_paper_figures --dataset pet --model deeplabv3
```

---

## What gets compared

Seven training strategies, all sharing the same base augmentation family so that
differences are attributable to *allocation* rather than to a different set of transforms:

| Method | Allocation rule |
| --- | --- |
| `baseline` | Uniform, fixed medium strength (`s = 0.5`) |
| `strong_uniform` | Uniform, maximum strength (`s = 1.0`) for every sample |
| `random_weighted` | Non-uniform but difficulty-blind — random per-sample weights |
| `loss_only` | Per-sample cross-entropy only |
| `rarity_only` | Inverse class frequency only |
| `boundary_only` | Mask contour density only |
| `DASA_EM` | All four signals, weighted 0.35 / 0.35 / 0.20 / 0.10 |

`DASA_EM` is the method the paper calls **DASA**. The `_EM` suffix is the code's name for
the round-based alternation between difficulty estimation and training (Algorithm 1 in
the paper); it is not a separate variant.

---

## Results

Headline numbers from the paper (image size 128×128, 20 baseline epochs, 3 rounds × 3
epochs for adaptive methods):

**Oxford-IIIT Pet** — mIoU, best per model in bold

| Model | Baseline | Strong uniform | Loss-only | Boundary-only | DASA |
| --- | --- | --- | --- | --- | --- |
| U-Net | 0.692 | 0.664 | 0.704 | 0.698 | **0.714** |
| DeepLabV3 | 0.633 | 0.716 | 0.725 | 0.717 | **0.740** |
| SegFormer-B0 | 0.711 | 0.686 | 0.730 | 0.718 | **0.730** |

**Binary Pascal VOC** — foreground IoU, where DASA wins for every architecture

| Model | Baseline | Best single-signal | DASA |
| --- | --- | --- | --- |
| U-Net | 0.219 | 0.423 (rarity) | **0.442** |
| DeepLabV3 | 0.453 | 0.461 (boundary) | **0.482** |
| SegFormer-B0 | 0.446 | 0.440 (rarity) | **0.468** |

Two things worth reading off these tables. First, `strong_uniform` *hurts* U-Net and
SegFormer-B0 while helping DeepLabV3 — globally turning augmentation up is not reliably
good, which is the paper's core motivation. Second, no single-signal ablation wins
everywhere; DASA's advantage comes from the signals being complementary.

---

## Citation

```bibtex
@article{arigbabu2026dasa,
  title   = {Difficulty-Aware Sample Allocation for Adaptive Data Augmentation
             in Semantic Segmentation},
  author  = {Arigbabu, Olasimbo Ayodeji and Arigbabu, Abimbola Ismail},
  journal = {arXiv preprint arXiv:2608.25710},
  year    = {2026}
}
```
