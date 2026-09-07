# Semi-Supervised Virtual Staining: AF → IHC via Pyramid-Constrained CycleGAN

> Semi-supervised CycleGAN with a multi-scale **Pyramid Loss** for virtual staining from
> label-free **autofluorescence (AF)** to **immunohistochemistry (IHC)**,
> targeting prostate cancer biomarkers (AMACR, p63/CK34βE12).
> Built on top of [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

**Author:** Zerui Kang · **Supervisor:** Prof. Terence Wong (HKUST)

---

## Overview

Chemical IHC staining is costly, slow, and tissue-destructive. Virtual staining offers a digital
alternative, but AF→IHC translation faces two unique challenges:

1. **Geometric misalignment** — AF and IHC images come from *adjacent physical sections*,
   so pixel-level registration is impossible; supervised methods (Pix2Pix) produce severe artifacts.
2. **Information-density asymmetry** — AF is a dense morphological signal while IHC is a
   *sparse* protein-expression signal; unsupervised CycleGAN suffers from structural
   hallucination and mode collapse (the "blank whiteboard" failure).

This project introduces a **semi-supervised CycleGAN** framework:

- A dynamic mask mixes **unpaired** data (domain adaptation via cycle/identity losses)
  with a small fraction (~30%) of **loosely paired** data (structural supervision).
- A **multi-scale Pyramid Loss** (5-level average-pooling pyramid) anchors macroscopic
  glandular topology while tolerating local misalignment.
- An Otsu-based adaptive pipeline cleans WSI patches (white background > 40% discarded),
  yielding **157,131** high-quality training images from multi-spectral AF (265/280/340 nm)
  and IHC whole-slide images.

## Key Results

| Direction | Finding |
|---|---|
| **IHC → AF (reverse)** | **+32.1% relative SSIM** over vanilla CycleGAN; pyramid constraint preserves glandular topology even under adversarial mode collapse |
| **AF → IHC (forward)** | Outperforms vanilla CycleGAN on all metrics (best PSNR); local mode collapse observed and analyzed — traced to the L1 "low-entropy shortcut" of blank-output |

Representative qualitative comparisons and loss-convergence curves are in [`assets/`](assets/).
The full analysis is in the [final project report](docs/Final_project_report.pdf).

## Code Changes vs. Upstream

| File | Change |
|---|---|
| `models/cycle_gan_model.py` | Semi-supervised mask logic; supervised path with L1 + Pyramid Loss |
| `models/networks.py` | Network adjustments for paired/unpaired mixed input |
| `options/train_options.py` | New flags: paired ratio, pyramid weight/depth, mask control |
| `data/unaligned_dataset.py` | Mixed paired/unpaired loader with dynamic mask |
| `train.py` | Training loop integration |

## Setup & Usage

```bash
conda env create -f environment.yml
conda activate pytorch-CycleGAN-and-pix2pix

# Semi-supervised training (30% paired sampling, pyramid weight 80)
python train.py --dataroot ./datasets/prostate_data --name semi_supervised_v1 \
    --model cycle_gan --lambda_pyramid 80.0 --paired_ratio 0.3
```

Trained on a single NVIDIA RTX 4090 (24 GB), PyTorch 2.8.0 + CUDA 12.8.

**Data & checkpoints** are not included in this repository due to size and clinical data
policy. Full experiment outputs (all baselines, both directions) are available at:
<云盘链接放这里>.

## Limitations & Future Work

- Forward AF→IHC mapping exhibits local mode collapse under extreme density asymmetry;
  the next iteration replaces cycle-consistency with a one-sided contrastive framework
  (CUT/Patch-NCE-based) with a two-stage pretrained encoder — see
  [docs/新模型.pdf](docs/新模型.pdf) for the proposed architecture.
- Evaluation is single-marker and single-center; multi-marker, multi-center validation is planned.

## Acknowledgements

This codebase is built on [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
(BSD license, see `LICENSE`; original README kept as `README_upstream.md`).
Clinical samples and imaging support: Prince of Wales Hospital and the Translational and
Advanced Bioimaging Laboratory, HKUST.

## Contact

Zerui Kang — <邮箱> · Issue reports welcome.
