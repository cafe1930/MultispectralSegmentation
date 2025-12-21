# 📌 MultispectralSegmentation

This repository contains code for training neural networks for **Empirical Evaluation of UNet for Segmentation of Applicable Surfaces for Seismic Sensors Installation** article.  
The main entry point is `train_nn.py`, which supports two training modes: single-model training and experiments with different spectral band combinations.

---

## 📁 Repository Structure

Key files and directories:
```
MultispectralSegmentation/
├── models/ # Neural network architectures
├── training_configs/ # YAML training configurations
│ ├── models/ # Model configs (UNet, etc.)
│ └── encoders/ # Encoder configs (EfficientNet, ResNet, ...)
├── train_nn.py # Main training script
├── data.py # Dataset loaders
├── lightning_wrapper.py # PyTorch Lightning module for training
└── ... # Notebooks and auxiliary files
```

## Installation
Clone the repository:

```bash
git clone https://github.com/cafe1930/MultispectralSegmentation.git
cd MultispectralSegmentation
```
Create and activate a virtual environment:
```bash
conda create -n sattelite_segmentation python=3.12 -y
conda activate sattelite_segmentation
pip install -r requirements.txt
```

## 🧠 Dataset Preparation

Before training, the dataset directory must contain:

* `data_info_table.csv` — metadata describing images and labels

* `surface_classes.json` — list of class names

* `dataset_partition.json` — train/validation/test split

Dataset loading is handled inside train_nn.py via:
```python
train_loader, test_loader, class_name2idx_dict, classes_weights = \
    create_seismic_sensors_dataset(config_dict)
```

(or a dataset-specific loader depending on the selected task).

## 📊 Metrics and Monitoring

The project uses **torchmetrics** to compute metrics such as:

- Intersection over Union (IoU / Jaccard Index)
- Precision
- Recall
- Confusion Matrix

Metrics are logged separately for training and validation (`train_*`, `val_*`).

## 📚 Citations

If you use this code or ideas from this repository in your research, please cite the corresponding paper:

```bibtex
@ARTICLE{tag,
  author = {M. Uzdiaev, M. Astapova, A. Ronzhin, A. Figurek},
  title = {Empirical Evaluation of UNet for Segmentation of Applicable Surfaces for Seismic Sensors Installation},
  year = {2025},
  journal = {...}
}
```