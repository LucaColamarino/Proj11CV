# Unknown Object Segmentation (UOS) with Conformal Prediction

This repository implements **Unknown Object Segmentation (UOS)** for urban driving scenarios, aiming to detect **out-of-distribution (OoD) obstacles** such as lost objects on the road.  
The system is based on **DeepLabV3+ (ResNet50 backbone)** and refined with **Conformal Prediction** to provide **statistically rigorous uncertainty estimates**.

Designed for datasets like **Cityscapes** and **Lost and Found**, the pipeline reduces the original 19 Cityscapes classes to **7 semantic macro-classes**.

---

## 📌 Key Features
- **DeepLabUOS** – customized DeepLabV3 model with an **Unknown Objectness Score (UOS)**
- **Conformal Prediction** – calibrates a threshold `q̂` for reliable unknown detection
- **Evaluation Metrics** – supports **mIoU**, **AUROC**, **AP**, **FPR95**
- **End-to-End Notebook** – training, calibration, and visualization included

---

## 📂 Project Structure

```
├── dataset.py                # Dataset loader, preprocessing & augmentations
├── model_uos.py              # DeepLabV3+ customized for UOS computation
├── conformal_prediction.py   # Conformal calibration and OoD mask
├── train_eval.py             # Evaluation metrics (mIoU, AUROC, AP, FPR95)
├── paper_impl.ipynb          # Full training & visualization notebook
├── requirements.txt          # Project dependencies
```

---

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

*(Make sure you have Python ≥ 3.8 and a working PyTorch installation with GPU support if available.)*

---

## 📀 Dataset Setup

The project expects the **Cityscapes/Lost and Found** structure:

```
dataset/
├── leftImg8bit/train/<city>/*.png
└── gtFine/train/<city>/*_labelTrainIds.png
```

### Macro-Class Mapping
The 19 Cityscapes labels are mapped to 7 macro-classes:

| ID | Class          |
|----|----------------|
| 0  | Road           |
| 1  | Flat Surfaces  |
| 2  | Human          |
| 3  | Vehicle        |
| 4  | Construction   |
| 5  | Objects        |
| 6  | Vegetation     |
| 7  | OoD / Unknown  |

---

## 🚀 Quick Start – Step by Step

### 1. Load Dataset

```python
from dataset import BaseSegmentationDataset, build_mapping_array

train_ds = BaseSegmentationDataset(
    img_dir="dataset/leftImg8bit/train",
    label_dir="dataset/gtFine/train",
    mapping_array=build_mapping_array(),
    resize=(256, 512),
    ood_mode=True
)
```

### 2. Initialize the Model

```python
from model_uos import DeepLabUOS
import torch

model = DeepLabUOS(n_classes=7).cuda()  # or .cpu()
```

### 3. Training

The `paper_impl.ipynb` notebook contains the full training loop.  
Simplified example:

```python
import torch.nn as nn
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

model.train()
for imgs, labels in train_loader:
    imgs, labels = imgs.cuda(), labels.cuda()
    optimizer.zero_grad()
    out = model(imgs)
    loss = criterion(out["logits"], labels)
    loss.backward()
    optimizer.step()
```

### 4. Calibrate Conformal Threshold

```python
from conformal_prediction import calibrate_conformal

qhat = calibrate_conformal(uos_scores_calib, alpha=0.05)
print("Calibrated q̂:", qhat)
```

### 5. Generate OoD Mask

```python
from conformal_prediction import conformal_mask

mask = conformal_mask(uos_map, qhat)  # Pixels with UOS ≥ q̂ flagged as OoD
```

### 6. Evaluate Performance

```python
from train_eval import evaluate_metrics, calculate_mIoU

metrics = evaluate_metrics(all_scores, all_gts)
print(metrics)  # {'AUROC': 0.91, 'AP': 0.87, 'FPR95': 0.12}

mIoU = calculate_mIoU(model, val_loader)
print("mIoU:", mIoU)
```

### 7. Visualize Results

Run the notebook:

```bash
jupyter notebook paper_impl.ipynb
```

---

## 📝 Citation

If you use this work, please cite:

```
@article{chen2017deeplabv3,
  title={Rethinking Atrous Convolution for Semantic Image Segmentation},
  author={Chen, Liang-Chieh et al.},
  journal={arXiv preprint arXiv:1706.05587},
  year={2017}
}

@article{angelopoulos2021conformal,
  title={Conformal prediction for reliable machine learning: Theory and applications},
  author={Angelopoulos, Anastasios N. et al.},
  journal={arXiv preprint arXiv:2107.07511},
  year={2021}
}
```

---

## ✅ Next Steps
- [ ] Train on full Cityscapes + Lost and Found
- [ ] Fine-tune temperature scaling for better UOS calibration
- [ ] Extend to real-time inference
