# RAGDE-Net: Radiometric-Agnostic Geometric Discrepancy Estimation for Multimodal Remote Sensing Image Registration

[Paper](https://github.com/Xinyu-Gong/RAGDE-Net) | [Dataset (TRSR)](https://github.com/Xinyu-Gong/RAGDE-Net) | [Code](https://github.com/Xinyu-Gong/RAGDE-Net)

This repository contains the official implementation of **RAGDE-Net**, a coarse-to-fine multimodal remote sensing image registration framework that decouples radiometric variation from geometric misalignment.

> **Xinyu Gong**, Faling Chen, Yunpeng Liu, Zelin Shi
> *IEEE Transactions on Geoscience and Remote Sensing (Under Review)*

## Overview

Multimodal remote sensing image registration is fundamentally hindered by the coupling of radiometric differences and geometric misalignment. RAGDE-Net addresses this with three coordinated components:

1. **Modality Translation Generator (MTG)** — Synthesizes structure-preserving pseudo-cross-modal image pairs with controllable geometric perturbations, decoupling radiometric variation from geometric misalignment.

2. **Geometric Discrepancy Evaluator (GDE)** — Trained offline on pseudo pairs to produce response maps that are **invariant to modality** yet **monotonically sensitive to misalignment**.

3. **Coarse-to-Fine Registration** — During online registration, both affine and deformable stages share the identical pretrained GDE for unified loss supervision and geometry-aware forward guidance.

## Key Features

- **Radiometric-agnostic**: GDE isolates geometric error from modality-specific appearance variation
- **Unified objective**: Shared GDE across coarse and fine stages avoids heterogeneous supervision
- **TRSR Dataset**: A newly constructed tri-modal (Visible-SAR-Infrared) remote sensing registration benchmark
- **State-of-the-art**: Consistently outperforms existing methods on visible-SAR and visible-infrared tasks

## TODO

- [ ] Release MTG training code
- [ ] Release GDE training and evaluation code
- [ ] Release coarse-to-fine registration pipeline
- [ ] Release pretrained model weights
- [ ] Release TRSR dataset download link
- [ ] Add inference demo and usage instructions

Code and models will be released gradually. Stay tuned!

## Citation

```bibtex
@article{gong2025ragde,
  title={Multimodal Remote Sensing Image Registration via Radiometric-Agnostic Geometric Discrepancy Estimation},
  author={Gong, Xinyu and Chen, Faling and Liu, Yunpeng and Shi, Zelin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  note={Under Review}
}
```

## License

This project is licensed under the MIT License.
