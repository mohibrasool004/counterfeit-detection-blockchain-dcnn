# Counterfeit Detection POC

## Overview
This proof-of-concept uses a toy Python blockchain to log SHA-256 hashes of images, paired with a Tiny DCNN that classifies “genuine” vs. “counterfeit.”

## Folder Structure
counterfeit_detection_project/
├── blockchain.py
├── dcnn_model.py
├── main.py
├── requirements.txt
├── README.md
└── data/
├── train/
│ ├── genuine/
│ └── counterfeit/
├── val/
│ ├── genuine/
│ └── counterfeit/
└── test_images/


## How to Prepare Data
1. Download a handful of “genuine” images and “counterfeit” images (see below for suggested Google search queries).
2. Place them under:
   - `data/train/genuine/`
   - `data/train/counterfeit/`
   - `data/val/genuine/`
   - `data/val/counterfeit/`
3. Reserve some mixed images in `data/test_images/` for inference.

## Steps to Run
1. `pip install -r requirements.txt`
2. `python dcnn_model.py`  
   → Trains a small CNN for 3–5 epochs and produces `counterfeit_detector.h5`.
3. Copy some images into `data/test_images/`.
4. `python main.py`  
   → Runs inference on `data/test_images/`, logs each image’s hash and predicted label into a blockchain, and prints the chain.

## Notes
- This is not production-grade. It’s a minimal demo to show how a blockchain ledger can “freeze” classification results.
- You can adjust the CNN architecture or PoW difficulty in `blockchain.py` as needed.