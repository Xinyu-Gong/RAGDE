# RAGDE-Net

Official PyTorch implementation of **Multimodal Remote Sensing Image Registration via Radiometric-Agnostic Geometric Discrepancy Estimation**.

RAGDE-Net is a coarse-to-fine registration framework built around a reusable Geometric Discrepancy Evaluator (GDE). The GDE learns dense, geometry-sensitive responses from structure-preserving pseudo cross-modal pairs, then supplies the same response guidance and geometric supervision to both affine and deformable registration stages.

## Method

The training pipeline has three sequential stages:

1. **MTG** learns bidirectional modality translation with CycleGAN losses and a multi-scale modality-independent neighborhood descriptor (MMIND) constraint.
2. **GDE** learns a dense discrepancy map from two controlled perturbations of one source image. One view is translated by the frozen MTG to introduce cross-modal appearance variation.
3. **Registration** freezes and shares the GDE across a ResNet-34 affine regressor and a U-Net deformable regressor. Both stages use additive response guidance and the same GDE-based residual objective.

The implementation follows Eqs. (3)-(26) in the paper. Defaults reported by the paper are collected in [`configs/ragde.yaml`](configs/ragde.yaml).

## Installation

Python 3.9 or newer and PyTorch 2.0 or newer are required.

```bash
git clone https://github.com/Xinyu-Gong/RAGDE.git
cd RAGDE
python -m pip install -e .
```

For development checks:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Data Layout

RAGDE expects spatially aligned, single-channel image pairs before synthetic perturbation. Images are paired by filename stem.

```text
data/TRSR/visible_sar/
├── train/
│   ├── visible/
│   │   ├── 0001.png
│   │   └── ...
│   └── sar/
│       ├── 0001.png
│       └── ...
└── test/
    ├── visible/
    └── sar/
```

Supported formats are PNG, JPEG, BMP, and TIFF. Inputs are resized to `256 x 256`, converted to grayscale, and normalized to `[-1, 1]`.

TRSR contains visible, SAR, and infrared patches derived from Sentinel-2, Sentinel-1, and Landsat 8/9 TIRS Band 10 data. Dataset release information will be added here when the archive is publicly available.

## Training

Copy the default configuration and set `data.train_root`, `data.val_root`, `data.fixed_dir`, and `data.moving_dir` for the desired modality pair.

```bash
ragde train mtg --config configs/ragde.yaml
ragde train gde --config configs/ragde.yaml
ragde train registration --config configs/ragde.yaml
```

Each stage consumes the checkpoint produced by the preceding stage. Paths can be overridden without editing YAML:

```bash
ragde train gde --config configs/ragde.yaml \
  --set mtg.checkpoint=outputs/mtg/last.pt
```

Resume a stage with optimizer and epoch state:

```bash
ragde train registration --config configs/ragde.yaml \
  --resume outputs/registration/last.pt
```

## Evaluation

Evaluation synthesizes the paper's controlled affine and smooth local perturbations from aligned test pairs and reports reprojection error (RE), corner RMSE, and displacement RMSE.

```bash
ragde evaluate \
  --config configs/ragde.yaml \
  --checkpoint outputs/registration/best.pt \
  --output outputs/registration/metrics.json
```

Use `ragde smoke --config configs/ragde.yaml` for a dataset-free check of all network interfaces before a long training run. Dataset pairing and paths are validated when a training or evaluation command starts.

## Reproducibility Notes

- Paper settings: Adam, learning rate `1e-4`, batch size `8`, `gamma=2`, `tau=10`, transition epoch `80`, and affine loss floor `0.1`.
- Synthetic affine ranges: rotation `[-30, 30]` degrees, translation `[-20%, 20%]`, scaling `[-15%, 15%]`, plus slight shear.
- Local perturbations use Gaussian-smoothed random displacement fields with `sigma=4` pixels.
- MTG is needed only while constructing GDE training pairs. Registration inference uses the affine regressor, deformable regressor, and shared GDE.
- The paper reports `256 x 256` inputs and 2,600/400 train/test pairs per benchmark. Dataset splitting is intentionally external to prevent accidental leakage.

## Citation

```bibtex
@article{gong2026ragde,
  title={Multimodal Remote Sensing Image Registration via Radiometric-Agnostic Geometric Discrepancy Estimation},
  author={Gong, Xinyu and Chen, Faling and Liu, Yunpeng and Shi, Zelin},
  year={2026}
}
```

Publication metadata will be updated after the final journal record is available.

## License

Released under the [MIT License](LICENSE).
