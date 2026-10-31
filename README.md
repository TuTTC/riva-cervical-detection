# ISBI 2026 RIVA Cervical Cytology Challenge (Track A): Detector Ensemble for Pap Smear Images

Object detection of (pre)cancerous cells in cervical cytology (Pap smear) images. Our solution fuses three heterogeneous detectors with **Weighted Boxes Fusion (WBF)** and ranked **8th of 66 teams in the preliminary phase**, evaluated with mAP@0.50:0.95.

<!-- TODO: add the challenge link and the final-phase result if available. -->

## Approach

```
Pap smear image
   ├── YOLO (tiled training, SAHI sliced inference) ──┐
   ├── Swin Transformer detector (mmdetection)  ──────┼── WBF ──> final boxes + class + score
   └── Faster R-CNN (paper reproduction) ─────────────┘
```

1. **YOLO with tiling.** Whole-slide images are large, so we cut them into overlapping 1024x1024 tiles for training (`pipeline/tiling/offline_tiling.py`) and use sliced inference at test time (`pipeline/tiling/inference_sahi.py`).
2. **Swin Transformer detector** (`swin_large/`), built on mmdetection.
3. **Faster R-CNN** following the *Holistic and Historical Instance Comparison* paper (`frcnn_paper/`): ResNet-50 FPN, box augmentation, contrastive instance comparison.
4. **Ensemble with WBF** (`pipeline/ensemble_3_models_wbf.py`): per-image fusion of the three prediction files with model weights 4 : 3 : 1 (YOLO : Swin : Faster R-CNN), IoU threshold 0.55 and skip-box threshold 0.01.
5. **Explored, not part of the main submission** (see script headers): a two-stage variant that re-classifies detected cells with an EfficientNet-B5 trained with focal loss (`pipeline/classifier/`).

## Results

<!-- TODO: fill in with numbers from the challenge leaderboard. The metric is mAP@0.50:0.95.

| Model | Preliminary score |
|---|---|
| YOLO (tiled + SAHI) | ? |
| Swin Transformer | ? |
| Faster R-CNN (paper) | ? |
| **WBF ensemble (4:3:1)** | ? (rank 8/66) |

Script comments mention the values 0.13, 0.12, 0.1075 and 0.18855 for the single models and the ensemble; confirm which metric they refer to before publishing. Add a short ablation of the WBF weights if you have one. -->

## Repository layout

```
frcnn_paper/   Faster R-CNN reproduction (own README inside)
swin_large/    Swin Transformer detector (own README inside)
pipeline/
├── tiling/        offline tiling, tiled YOLO training, SAHI inference
├── training/      YOLOv11 and RT-DETR training
├── detection/     Faster R-CNN training scripts
├── classifier/    EfficientNet-B5 second-stage classifier (exploratory)
├── inference/     one- and two-stage inference
├── utils/         cell cropping, feature fusion
└── ensemble_*.py  WBF ensembles over submission CSV files
```

## Data

The challenge data are **not** redistributed here. Register on the official challenge page, accept its terms, download the dataset, and place it under `data/` (git-ignored):

```
data/riva-cervical-cytology-challenge-isbi-2026/
├── images/
└── annotations/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r frcnn_paper/requirements.txt
pip install -r swin_large/requirements.txt
pip install ultralytics sahi ensemble-boxes
```

## Reproduce

> Paths inside `pipeline/` currently point to a local drive layout. Edit the path constants at the top of each script before running. <!-- TODO: replace with a shared config. -->

```bash
python pipeline/tiling/offline_tiling.py          # 1. tile the images
python pipeline/tiling/train_tiled_detection.py   # 2. train YOLO on tiles
python swin_large/run.py                          # 3. train the Swin detector
python frcnn_paper/run.py --mode all              # 4. train Faster R-CNN
python pipeline/ensemble_3_models_wbf.py          # 5. fuse the three prediction CSVs with WBF
```

## Acknowledgements

- Faster R-CNN reproduction based on *Holistic and Historical Instance Comparison for Cervical Cell Detection*.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [mmdetection](https://github.com/open-mmlab/mmdetection), [SAHI](https://github.com/obss/sahi), [ensemble-boxes](https://github.com/ZFTurbo/Weighted-Boxes-Fusion).
- The RIVA challenge organizers for the data.

## Contact

Tran Thi Cam Tu, University of Information Technology, VNU-HCM, `camtutran2608@gmail.com`.
