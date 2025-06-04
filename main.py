# File: main.py

import os
import hashlib
import json
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from blockchain import Blockchain

# ===========================
# USER‐CONFIGURABLE PARAMETERS
# ===========================
MODEL_PATH = "counterfeit_detector.h5"
DIFFICULTY = 2

# Paths (relative to project root)
GENUINE_TEST_DIR = os.path.join("data", "genuine", "Test")
# If you later have a counterfeit/Test folder, you can add it similarly:
# COUNTERFEIT_TEST_DIR = os.path.join("data", "counterfeit", "Test")

def compute_image_sha256(image_path):
    """
    Read raw bytes from image file → compute SHA‐256 → return hex digest string.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return hashlib.sha256(img_bytes).hexdigest()

def preprocess_and_predict(model, image_path):
    """
    1. Load image from image_path, resize to (128,128), normalize to [0,1].
    2. Predict with model.
    3. Return predicted label: "genuine" or "counterfeit".
    """
    img = load_img(image_path, target_size=(128, 128))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1, 128, 128, 3)
    preds = model.predict(arr)         # shape: (1, 2)
    class_idx = np.argmax(preds, axis=1)[0]
    # Keras’s categorical encoding: index 0 → first alphabetical class (likely "counterfeit" or "genuine").
    # To be safe, inspect model.class_names if using flow_from_dataframe. Here we assume:
    #   class_indices = {"counterfeit": 0, "genuine": 1}  OR  vice versa.
    # Our training script used flow_from_dataframe, which orders classes alphabetically:
    #   "counterfeit" < "genuine", so index=0 → "counterfeit", index=1 → "genuine".
    return "genuine" if class_idx == 1 else "counterfeit"

def main():
    # 1. Load or instantiate blockchain
    bc = Blockchain(difficulty=DIFFICULTY)

    # 2. Load the CNN model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found. Please train first with 'python dcnn_model.py'.")
        return

    model = load_model(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")

    # 3. Gather all test images under data/genuine/Test/…
    if not os.path.isdir(GENUINE_TEST_DIR):
        print(f"ERROR: Test directory '{GENUINE_TEST_DIR}' not found.")
        return

    test_filepaths = []
    for root_dir, _, files in os.walk(GENUINE_TEST_DIR):
        for fname in files:
            ext = fname.lower().split('.')[-1]
            if ext in ("jpg", "jpeg", "png", "bmp"):
                test_filepaths.append(os.path.join(root_dir, fname))

    if not test_filepaths:
        print(f"ERROR: No image files found in '{GENUINE_TEST_DIR}'.")
        return

    # 4. Iterate over each test image, predict, hash, and add to blockchain
    for image_path in test_filepaths:
        fname = os.path.basename(image_path)
        sha256_hash = compute_image_sha256(image_path)
        predicted_label = preprocess_and_predict(model, image_path)

        data_dict = {
            "image_hash": sha256_hash,
            "filename": fname,
            "label": predicted_label,
            "timestamp": time.time()
        }
        bc.add_block(data_dict)

        print(f"[+] {fname} → {predicted_label}   (hash: {sha256_hash[:10]}...)")

    # 5. Print full chain
    print("\n=== Full Blockchain ===")
    for block in bc.chain:
        print(
            f"Index: {block.index}  | "
            f"Hash: {block.hash[:12]}...  | "
            f"Prev: {block.previous_hash[:12]}...  | "
            f"Data: {block.data}"
        )

    # 6. Save chain to JSON
    json_chain = []
    for blk in bc.chain:
        json_chain.append({
            "index": blk.index,
            "timestamp": blk.timestamp,
            "data": blk.data,
            "previous_hash": blk.previous_hash,
            "nonce": blk.nonce,
            "hash": blk.hash
        })
    with open("chain_record.json", "w") as f:
        json.dump(json_chain, f, indent=2)
    print("\nBlockchain written to 'chain_record.json'")

if __name__ == "__main__":
    main()
