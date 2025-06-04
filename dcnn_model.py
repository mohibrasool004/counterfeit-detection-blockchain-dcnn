# File: dcnn_model.py

import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ===========================
# USER‐CONFIGURABLE PARAMETERS
# ===========================
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
NUM_CLASSES = 2          # genuine vs. counterfeit
EPOCHS = 5               # you can reduce to 3 for a quick demo
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2   # 20% of combined data for validation

# Paths (relative to project root)
GENUINE_TRAIN_DIR = os.path.join("data", "genuine", "Train")
COUNTERFEIT_DIR = os.path.join("data", "counterfeit")

# Output model filename
OUTPUT_MODEL_PATH = "counterfeit_detector.h5"


def gather_filepaths_and_labels():
    """
    Walk through the Kaggle folders and build a DataFrame of:
      - 'filepath': full path to each image file under genuine/train or counterfeit
      - 'label': 'genuine' or 'counterfeit'
    Returns:
      df (pd.DataFrame) with columns ['filepath', 'label']
    """
    filepaths = []
    labels = []

    # 1. Genuine: under data/genuine/Train/, subfolders like 1Hundrednote/, 2Hundrednote/, etc.
    for root_dir, _, files in os.walk(GENUINE_TRAIN_DIR):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                full_path = os.path.join(root_dir, fname)
                filepaths.append(full_path)
                labels.append("genuine")

    # 2. Counterfeit: under data/counterfeit/, possibly nested subfolders
    for root_dir, _, files in os.walk(COUNTERFEIT_DIR):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                full_path = os.path.join(root_dir, fname)
                filepaths.append(full_path)
                labels.append("counterfeit")

    df = pd.DataFrame({
        "filepath": filepaths,
        "label": labels
    })
    return df


def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """
    Build a simple CNN:
      - 3 conv blocks (32→64→128 filters)
      - Flatten → Dense(128) → Dropout(0.5) → Dense(NUM_CLASSES, softmax)
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    return model


def train_model():
    """
    1. Gather filepaths & labels into DataFrame.
    2. Split into train_df / val_df, stratified by 'label'.
    3. Create ImageDataGenerators & train the CNN.
    4. Save model to OUTPUT_MODEL_PATH.
    """
    # Step 1: gather all filepaths
    df = gather_filepaths_and_labels()
    if df.empty:
        print("ERROR: No images found under 'data/genuine/Train' or 'data/counterfeit'.")
        return

    # Step 2: train/validation split (stratify by 'label')
    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SPLIT,
        stratify=df["label"],
        random_state=42
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Sanity check
    print(f"Total images: {len(df)}")
    print(f"→ Training: {len(train_df)}")
    print(f"→ Validation: {len(val_df)}")
    print(train_df["label"].value_counts(), val_df["label"].value_counts(), sep="\n")

    # Step 3: Data generators
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="filepath",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    # Step 4: Build and compile model
    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Step 5: Train
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator)
    )

    # Step 6: Save model
    model.save(OUTPUT_MODEL_PATH)
    print(f"Training complete. Model saved at '{OUTPUT_MODEL_PATH}'.")


if __name__ == "__main__":
    train_model()
