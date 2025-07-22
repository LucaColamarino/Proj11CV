# Unknown Object Segmentation (UOS) with Conformal Prediction

This repository implements **Unknown Object Segmentation (UOS)** for urban driving scenarios, aiming to detect **out-of-distribution (OoD) obstacles** such as lost objects on the road.  
The system is based on **DeepLabV3+ (ResNet50 backbone)** and refined with **Conformal Prediction** to provide **statistically rigorous uncertainty estimates**.

Designed for datasets like **Cityscapes** and **Lost and Found**, the pipeline reduces the original 19 Cityscapes classes to **7 semantic macro-classes**.

---

## Key Features
- **DeepLabUOS** – customized DeepLabV3 model with an **Unknown Objectness Score (UOS)**
- **Conformal Prediction** – calibrates a threshold `q̂` for reliable unknown detection
- **Evaluation Metrics** – supports **mIoU**, **AUROC**, **AP**, **FPR95**
- **End-to-End Notebook** – training, calibration, and visualization included

---

## Project Structure

```
├── dataset.py                # Dataset loader, preprocessing & augmentations
├── model_uos.py              # DeepLabV3+ customized for UOS computation
├── conformal_prediction.py   # Conformal calibration and OoD mask
├── train_eval.py             # Evaluation metrics (mIoU, AUROC, AP, FPR95)
├── main.ipynb                # Full training & visualization notebook
├── requirements.txt          # Project dependencies
```

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/LucaColamarino/Proj11CV
cd Proj11CV
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

*(Make sure you have Python ≥ 3.8 and a working PyTorch installation with GPU support if available.)*

---

## Dataset Setup

The project expects the **Cityscapes/Lost and Found** structure:

```
dataset/
├── leftImg8bit/train/<city>/*.png
└── gtFine(gtCoarse)/train/<city>/*_labelTrainIds.png

In order to set it up, you need to :
    Download Cityscapes dataset from https://www.cityscapes-dataset.com/downloads/ (download gtFine_trainvaltest.zip (241MB) [md5] and leftImg8bit_trainvaltest.zip (11GB) [md5])
    Download LostandFound dataset from https://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html (gtCoarse.zip and leftImg8bit.zip).

Create the environment variable CITYSCAPES_DATASET (absolute path to cityscapes dataset).
Run:
python .\OLD\cityscapesscripts\preparation\createTrainIdLabelImgs.py

Eventually adjust the CITYSCAPES_ROOT and LOSTANDFOUND_ROOT in the **Globals** section of the notebook *main.ipynb*

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

## Quick Start – How to run it

Run the notebook:

```bash
jupyter notebook main.ipynb
```