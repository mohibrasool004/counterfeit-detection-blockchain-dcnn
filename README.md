# Counterfeit Detection using Blockchain and DCNN

## Overview
This research project demonstrates a novel approach to counterfeit detection by combining Deep Convolutional Neural Networks (DCNN) with blockchain technology. The system classifies currency images as "genuine" or "counterfeit" and creates an immutable audit trail using a proof-of-work blockchain.

## Project Structure
```
counterfeit_detection_project/
├── blockchain.py              # Blockchain implementation with proof-of-work
├── dcnn_model.py             # CNN model training script
├── main.py                   # Main inference and blockchain logging pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── counterfeit_detector.h5   # Trained CNN model
├── chain_record.json         # Blockchain execution results
└── data/
    ├── genuine/
    │   ├── Train/            # Training data for genuine currency
    │   │   ├── 1Hundrednote/
    │   │   ├── 2Hundrednote/
    │   │   ├── 2Thousandnote/
    │   │   ├── 5Hundrednote/
    │   │   ├── Fiftynote/
    │   │   ├── Tennote/
    │   │   └── Twentynote/
    │   └── Test/             # Test data for genuine currency
    │       ├── 1Hundrednote/
    │       ├── 2Hundrednote/
    │       ├── 2Thousandnote/
    │       ├── 5Hundrednote/
    │       ├── Fiftynote/
    │       ├── Tennote/
    │       └── Twentynote/
    └── counterfeit/
        ├── 2000_Features/    # Counterfeit 2000 rupee notes
        ├── 2000_dataset/
        ├── 500_Features/     # Counterfeit 500 rupee notes
        └── 500_dataset/
```

## Key Features
- **Deep CNN Architecture**: 3-layer convolutional network with dropout regularization
- **Blockchain Integration**: Proof-of-work blockchain for immutable record keeping
- **SHA-256 Hashing**: Cryptographic integrity verification of images
- **End-to-End Pipeline**: From image input to blockchain logging
- **Real Dataset**: 300+ genuine and counterfeit Indian currency images

## Quick Start
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (Optional - pre-trained model included)
   ```bash
   python dcnn_model.py
   ```

3. **Run Detection Pipeline**
   ```bash
   python main.py
   ```

## How It Works
1. **Image Processing**: Load and preprocess currency images (128x128 RGB)
2. **CNN Classification**: Predict "genuine" or "counterfeit" with confidence scores
3. **Hash Generation**: Compute SHA-256 hash of original image for integrity
4. **Blockchain Logging**: Mine new block with image hash, prediction, and timestamp
5. **Immutable Record**: Create tamper-proof audit trail of all decisions

## Technical Details
- **CNN Architecture**: Conv2D(32) → Conv2D(64) → Conv2D(128) → Dense(128) → Dense(2)
- **Blockchain**: SHA-256 hashing with configurable proof-of-work difficulty
- **Dataset**: Indian currency notes (10, 20, 50, 100, 200, 500, 2000 rupees)
- **Image Size**: 128x128 pixels, RGB channels
- **Training**: 5 epochs with 20% validation split

## Research Applications
This framework demonstrates trustworthy AI systems for:
- Financial fraud detection
- Supply chain authentication  
- Medical diagnosis verification
- Legal evidence documentation
- Regulatory compliance auditing

## Sample Results
The system has processed 42 test images with results logged in `chain_record.json`:
- 41 images correctly classified as genuine
- 1 image flagged as potential counterfeit
- Each decision cryptographically secured in blockchain

## Notes
- This is a research proof-of-concept demonstrating blockchain-AI integration
- Production deployment would require larger datasets and enhanced security measures
- Blockchain difficulty can be adjusted in `blockchain.py` for performance tuning

## Author
Mohib Rasool
