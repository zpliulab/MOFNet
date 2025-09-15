# MOFNet: Multi-Omics Fusion Network for Cancer Subtype Classification

MOFNet is a supervised deep learning framework for integrating multi-omics data to improve cancer subtype classification. 
It combines graph convolutional networks with Similarity Graph Pooling (SGO) and cross-omics fusion in the label space (VCDN).

## Features
- Multi-omics integration (mRNA, DNA methylation, microRNA)
- Graph-based feature selection with SGO
- Graph Structure Learning (GSL) with Sparsemax normalization
- Cross-omics fusion in label space via VCDN
- Supports reproducible experiments with preprocessed TCGA datasets

## Data Availability
- Raw datasets are available from TCGA via the R package TCGAbiolinks.
- We provide preprocessing scripts, train/test splits, and small sample matrices
  under `data/` for reproducibility.

## Code Availability
- Source code is available here on GitHub.

## Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/YourUsername/MOFNet.git
cd MOFNet
pip install -r requirements.txt
python main.py --my_dataset BRCA_1 BRCA_2 BRCA_3 --my_dataset_test BRCA_test_1 BRCA_test_2 BRCA_test_3
