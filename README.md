# SAID — Spectral-Attention Integrated Detector

> Fog-resistant object detection achieving >93% mAP on the RTTS dataset.
> Evolution of Li et al. (2025) YOLOX+AOD-Net baseline (88.84% mAP).

## Architecture

| Component | Description |
|---|---|
| **Base** | YOLOv12-X (Ultralytics) |
| **A2C2f-FSA** | FFT frequency disentanglement — suppresses fog veil, enhances texture |
| **DSA** | Deformable Spatial Attention in C3k2 bottlenecks |
| **Loss** | WIoU v3 + Inner-MPDIoU (difficulty-adaptive, small-object focused) |

## Dataset

- **Stage 1 pre-train:** VOC-FOG (9,578 synthetic foggy images)
- **Stage 2 fine-tune:** RTTS (4,322 real-world foggy road images, 60/20/20 split)

## Project Structure

```
said/
├── a2c2f_fsa.py       # A2C2f-FSA module (FFT + DSA)
└── wiou_loss.py       # WIoU v3 + Inner-MPDIoU loss

train.py               # Two-stage training script
prepare_data.py        # RTTS dataset preparation
prepare_vocfog.py      # VOC-FOG → YOLO format conversion
smoke_test.py          # Pipeline validation
said_kaggle.ipynb      # Kaggle P100 training notebook
rtts.yaml              # RTTS dataset config
vocfog.yaml            # VOC-FOG dataset config
```

## Usage

```bash
# 1. Prepare datasets
python prepare_data.py       # RTTS → RTTS_Ready/
python prepare_vocfog.py     # voc-fog → VOC_FOG_YOLO/

# 2. Smoke test
python smoke_test.py

# 3. Train (local MPS / CUDA auto-detected)
python train.py --stage both --epochs 100 --batch 16

# 4. Train on Kaggle P100 → open said_kaggle.ipynb
```

## Training Schedule (Stage 2)

- Validate every **10 epochs** on RTTS val
- Test every **20 epochs** on RTTS test + VOC-FOG test  
- Save rolling-best checkpoint every **5 epochs**
- Save final weights on completion
