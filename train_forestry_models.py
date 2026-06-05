# ── CELL 1: Environment Setup ─────────────────────────────────────────────
# Detects whether running on Google Colab or locally.
# On Colab: clones the dataset from GitHub automatically.
# On local: adjust DATASET_ROOT to point to your local path.

import os
import sys

IS_COLAB = 'COLAB_GPU' in os.environ or os.path.exists('/content')

if IS_COLAB:
    print("Running on Google Colab")
    if not os.path.exists('/content/forestry-image-dataset'):
        print("Cloning dataset from GitHub (this may take a few minutes)...")
        os.system('git clone --depth=1 https://github.com/daus-Onn/forestry-image-dataset.git /content/forestry-image-dataset')
    else:
        print("Dataset already available.")
    DATASET_ROOT = "/content/forestry-image-dataset/forestry_dataset_splitted"

else:
    print("Running locally")
    # ── CHANGE THIS PATH to your local forestry_dataset_splitted folder ──────
    DATASET_ROOT = r".\forestry-image-dataset\forestry_dataset_splitted"
    # Example: r"C:\Users\Zariff\Downloads\forestry-image-dataset\forestry_dataset_splitted"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR   = os.path.join(DATASET_ROOT, "val")
TEST_DIR  = os.path.join(DATASET_ROOT, "test")

print(f"\nTRAIN : {TRAIN_DIR}")
print(f"VAL   : {VAL_DIR}")
print(f"TEST  : {TEST_DIR}")# ── CELL 2: Imports ───────────────────────────────────────────────────────

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNetV3Large

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow : {tf.__version__}")
print(f"GPU        : {gpus[0].name if gpus else 'None - training will be slow on CPU!'}")
print(f"NumPy      : {np.__version__}")# ── CELL 3: Global Configuration ─────────────────────────────────────────

# Image settings
IMG_SIZE   = (224, 224)   # Standard input for ResNet50 / DenseNet121 / MobileNetV3
BATCH_SIZE = 16

# Dataset
N_CLASSES   = 4
CLASS_NAMES = ['bamboo_forest', 'coniferous_pine_forest',
               'mangrove_forest', 'tropical_rainforest']

# Training schedule: 50 epochs total (Phase 1 + Phase 2 = 50)
TOTAL_EPOCHS   = 50    # Assignment requirement
PHASE1_EPOCHS  = 20    # Feature extraction: head only, base frozen
PHASE2_EPOCHS  = 30    # Fine-tuning: last N layers unfrozen
UNFREEZE_LAST  = 30    # Number of base-model layers to unfreeze in Phase 2

# Learning rates
LR_P1 = 1e-3   # Phase 1 - higher LR fine for training new head
LR_P2 = 1e-5   # Phase 2 - low LR prevents destroying pretrained weights

# Only load these extensions (SVG is a vector format - TF cannot decode it)
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

# Save directory for best model checkpoints
SAVE_DIR = "./model_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Configuration ready:")
print(f"  IMG_SIZE       : {IMG_SIZE}")
print(f"  BATCH_SIZE     : {BATCH_SIZE}")
print(f"  N_CLASSES      : {N_CLASSES} -> {CLASS_NAMES}")
print(f"  TOTAL_EPOCHS   : {TOTAL_EPOCHS}  (Phase1={PHASE1_EPOCHS} frozen, Phase2={PHASE2_EPOCHS} fine-tune)")
print(f"  LR Phase1/2    : {LR_P1} / {LR_P2}")# ── CELL 4: Dataset Statistics ────────────────────────────────────────────
# Counts valid (non-SVG) images per class per split.

def count_split(split_dir):
    result = {}
    skipped = 0
    for cls in sorted(os.listdir(split_dir)):
        cls_path = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        files = os.listdir(cls_path)
        v = sum(1 for f in files if Path(f).suffix.lower() in VALID_EXTS)
        s = len(files) - v
        result[cls] = v
        skipped += s
    return result, skipped

train_counts, skip_tr = count_split(TRAIN_DIR)
val_counts,   skip_va = count_split(VAL_DIR)
test_counts,  skip_te = count_split(TEST_DIR)
total_skipped = skip_tr + skip_va + skip_te

print(f"{'Class':<28} {'Train':>6}  {'Val':>5}  {'Test':>5}  {'Total':>6}")
print("-" * 58)
grand = 0
for cls in CLASS_NAMES:
    t  = train_counts.get(cls, 0)
    v  = val_counts.get(cls, 0)
    te = test_counts.get(cls, 0)
    grand += t + v + te
    print(f"{cls:<28} {t:>6}  {v:>5}  {te:>5}  {t+v+te:>6}")
print("-" * 58)
tr_t = sum(train_counts.values())
va_t = sum(val_counts.values())
te_t = sum(test_counts.values())
print(f"{'TOTAL':<28} {tr_t:>6}  {va_t:>5}  {te_t:>5}  {grand:>6}")
print(f"\nSVG files skipped (unsupported): {total_skipped}")
print(f"Usable images                  : {grand}")# ── CELL 5: Custom Dataset Loader ─────────────────────────────────────────
# Builds a tf.data.Dataset from a directory.
# Skips SVG files. Supports jpg/png/webp.
# Optionally applies data augmentation (for training set only).

def build_dataset(split_dir, augment=False, shuffle=True):
    """
    Scans split_dir (which has sub-folders per class), collects valid image
    paths and integer labels, then returns a batched tf.data.Dataset.

    Args:
        split_dir : path to train / val / test directory
        augment   : apply random augmentation (use True only for training)
        shuffle   : shuffle before batching

    Returns:
        dataset   : tf.data.Dataset of (image, label) batches
        n_images  : number of images in this split
    """
    cls_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    file_paths, labels = [], []

    for cls in CLASS_NAMES:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if Path(fname).suffix.lower() in VALID_EXTS:
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_to_idx[cls])

    n_images = len(file_paths)

    # ── Build pipeline ────────────────────────────────────────────────────
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=n_images, seed=SEED)

    # ── Load and resize ───────────────────────────────────────────────────
    @tf.function
    def load_image(path, label):
        raw  = tf.io.read_file(path)
        img  = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img  = tf.image.resize(img, IMG_SIZE)
        img  = tf.cast(img, tf.float32) / 255.0   # Normalise to [0, 1]
        return img, label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Silently drop any remaining problematic files (e.g. corrupt webp)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    # ── Data augmentation (training only) ─────────────────────────────────
    if augment:
        augment_pipeline = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.12),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ], name="augmentation")

        @tf.function
        def apply_augment(img, label):
            img = augment_pipeline(img, training=True)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, label

        ds = ds.map(apply_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, n_images


print("Building datasets...")
train_ds, n_train = build_dataset(TRAIN_DIR, augment=True,  shuffle=True)
val_ds,   n_val   = build_dataset(VAL_DIR,   augment=False, shuffle=False)
test_ds,  n_test  = build_dataset(TEST_DIR,  augment=False, shuffle=False)

print(f"\nTrain : {n_train} images  ({n_train // BATCH_SIZE + 1} batches)")
print(f"Val   : {n_val}   images  ({n_val   // BATCH_SIZE + 1} batches)")
print(f"Test  : {n_test}  images  ({n_test  // BATCH_SIZE + 1} batches)")
print("\nDatasets ready!")# ── CELL 6: Visualise Sample Images ──────────────────────────────────────
# Shows 3 random samples from each class in the training set.

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
fig.suptitle("Sample Images from Training Set", fontsize=16, fontweight='bold', y=1.01)

cls_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

for row, cls in enumerate(CLASS_NAMES):
    cls_dir = os.path.join(TRAIN_DIR, cls)
    files   = [f for f in os.listdir(cls_dir) if Path(f).suffix.lower() in VALID_EXTS]
    sampled = np.random.choice(files, min(4, len(files)), replace=False)

    for col, fname in enumerate(sampled):
        img_path = os.path.join(cls_dir, fname)
        try:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, (200, 200)).numpy().astype('uint8')
        except Exception:
            img = np.zeros((200, 200, 3), dtype='uint8')

        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(cls.replace('_', '\n'), fontsize=10,
                                       rotation=0, ha='right', va='center',
                                       labelpad=60)

plt.tight_layout()
plt.savefig("sample_images.png", dpi=120, bbox_inches='tight')
# plt.show()
print("Sample images saved as sample_images.png")# ── CELL 7: Model Builder Function ────────────────────────────────────────

def build_transfer_model(backbone_fn, model_name,
                         include_preprocessing=False):
    """
    Builds a transfer-learning model with a custom classification head.

    Args:
        backbone_fn            : Keras application function (e.g. ResNet50)
        model_name             : string name for the model
        include_preprocessing  : True only for MobileNetV3 (disables built-in preprocess)

    Returns:
        model : full tf.keras.Model
        base  : the backbone layer (needed to unfreeze in Phase 2)
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input")

    # ── Backbone (ImageNet weights, no top classifier) ────────────────────
    kwargs = dict(include_top=False, weights='imagenet',
                  input_shape=(*IMG_SIZE, 3))
    if include_preprocessing:
        kwargs['include_preprocessing'] = False   # We handle normalisation

    base = backbone_fn(**kwargs)
    base.trainable = False   # Phase 1: freeze all base layers

    # ── Forward pass through backbone ────────────────────────────────────
    x = base(inputs, training=False)   # training=False keeps BN frozen in Phase 1

    # ── Custom classification head ────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation='relu', name="fc1")(x)
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(N_CLASSES, activation='softmax', name="predictions")(x)

    model = keras.Model(inputs, outputs, name=model_name)
    return model, base


print("build_transfer_model() defined.")
print("Architecture: BaseModel -> GAP -> Dense(256) -> BN -> Dropout -> Dense(4, softmax)")# ── CELL 8: mAP Callback ──────────────────────────────────────────────────
# Computes mean Average Precision on the validation set at each epoch.
# mAP = macro-average of per-class Average Precision (one-vs-rest PR-AUC).
# This is computed using sklearn.metrics.average_precision_score.

class MAPCallback(keras.callbacks.Callback):
    """
    Custom Keras callback that computes mAP on the validation set
    at the end of each epoch and logs it as 'val_mAP'.

    mAP for multi-class classification:
      For each class c: AP_c = area under Precision-Recall curve (one-vs-rest)
      mAP = mean(AP_0, AP_1, ..., AP_{n-1})
    """

    def __init__(self, val_dataset, n_classes, compute_every=1):
        super().__init__()
        self.val_dataset   = val_dataset
        self.n_classes     = n_classes
        self.compute_every = compute_every   # Compute every N epochs
        self.history       = []              # Stores mAP per epoch
        self._last_map     = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.compute_every != 0:
            self.history.append(self._last_map)
            if logs: logs['val_mAP'] = self._last_map
            return

        y_true_all, y_pred_all = [], []

        for x_batch, y_batch in self.val_dataset:
            preds = self.model.predict(x_batch, verbose=0)
            y_true_all.extend(y_batch.numpy().tolist())
            y_pred_all.extend(preds.tolist())

        # One-hot encode ground truth for sklearn
        y_true_oh = keras.utils.to_categorical(
            np.array(y_true_all), self.n_classes)
        y_pred_np  = np.array(y_pred_all)

        map_score = average_precision_score(
            y_true_oh, y_pred_np, average='macro')

        self._last_map = float(map_score)
        self.history.append(self._last_map)
        if logs:
            logs['val_mAP'] = self._last_map

        print(f"  val_mAP: {self._last_map:.4f}", end='')


print("MAPCallback defined.")
print("Usage: tracks mAP (macro avg. precision) on validation set each epoch.")# ── CELL 9: Two-Phase Training Function ───────────────────────────────────

def train_model(model, base, model_name):
    """
    Trains a transfer-learning model in two phases for exactly TOTAL_EPOCHS.

    Phase 1 (epochs 1-PHASE1_EPOCHS):
      - Base frozen, only custom head trains.
      - Adam(lr=LR_P1)

    Phase 2 (epochs PHASE1_EPOCHS+1 to TOTAL_EPOCHS):
      - Last UNFREEZE_LAST layers of base unfrozen.
      - Adam(lr=LR_P2) - very small LR to fine-tune without destroying weights.

    Returns:
      combined_history : dict with keys: loss, accuracy, val_loss, val_accuracy, val_mAP
      training_time    : total seconds for both phases
    """

    ckpt_path  = os.path.join(SAVE_DIR, f"{model_name}_best.h5")
    map_cb     = MAPCallback(val_ds, N_CLASSES, compute_every=1)

    # Common callbacks
    shared_cbs = [
        map_cb,
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_accuracy',
            save_best_only=True, save_weights_only=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-8, verbose=1),
    ]

    t0 = time.time()

    # ── Phase 1: Feature Extraction (frozen base) ─────────────────────────
    print(f"\n{'='*65}")
    print(f"  {model_name}  |  Phase 1: Feature Extraction  (epochs 1-{PHASE1_EPOCHS})")
    print(f"  Base frozen. Training classification head only.")
    print(f"{'='*65}")

    model.compile(
        optimizer=keras.optimizers.Adam(LR_P1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    hist1 = model.fit(
        train_ds,
        epochs=PHASE1_EPOCHS,
        validation_data=val_ds,
        callbacks=shared_cbs,
        verbose=1
    )

    # ── Phase 2: Fine-tuning (partial unfreeze) ───────────────────────────
    print(f"\n{'='*65}")
    print(f"  {model_name}  |  Phase 2: Fine-Tuning  (epochs {PHASE1_EPOCHS+1}-{TOTAL_EPOCHS})")
    print(f"  Unfreezing last {UNFREEZE_LAST} layers of base. LR reduced to {LR_P2}.")
    print(f"{'='*65}")

    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAST]:
        layer.trainable = False

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"  Trainable layers: {trainable_count}/{len(model.layers)} total")

    model.compile(
        optimizer=keras.optimizers.Adam(LR_P2),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    hist2 = model.fit(
        train_ds,
        initial_epoch=PHASE1_EPOCHS,         # Continue from epoch 20
        epochs=TOTAL_EPOCHS,                  # End at epoch 50
        validation_data=val_ds,
        callbacks=shared_cbs,
        verbose=1
    )

    training_time = time.time() - t0

    # ── Merge Phase 1 and Phase 2 histories ───────────────────────────────
    combined = {}
    for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
        combined[key] = hist1.history.get(key, []) + hist2.history.get(key, [])
    combined['val_mAP'] = map_cb.history

    # Pad mAP if shorter than epoch count (due to compute_every)
    while len(combined['val_mAP']) < TOTAL_EPOCHS:
        combined['val_mAP'].append(combined['val_mAP'][-1])

    mins = int(training_time // 60)
    secs = int(training_time %  60)
    print(f"\n  Training complete!  Total time: {mins}m {secs}s")
    print(f"  Best model saved to: {ckpt_path}")

    return combined, training_time


print("train_model() defined.")# ── CELL 10: Build ResNet50 ───────────────────────────────────────────────

resnet_model, resnet_base = build_transfer_model(
    backbone_fn=ResNet50,
    model_name="ResNet50"
)

# Print model summary
print(f"ResNet50 total layers     : {len(resnet_model.layers)}")
print(f"Total parameters          : {resnet_model.count_params():,}")
print(f"Trainable parameters      : {sum(v.numpy().size for v in resnet_model.trainable_variables):,}")
print(f"Non-trainable parameters  : {sum(v.numpy().size for v in resnet_model.non_trainable_variables):,}")
print("\nModel head:")
for layer in resnet_model.layers[-6:]:
    print(f"  Layer: {layer.name}  |  Trainable: {layer.trainable}")# ── CELL 11: Train ResNet50 ───────────────────────────────────────────────
# 50 epochs total: 20 frozen + 30 fine-tuned

resnet_history, resnet_time = train_model(
    model=resnet_model,
    base=resnet_base,
    model_name="ResNet50"
)

print(f"\nResNet50 Final Results:")
print(f"  val_accuracy (last epoch) : {resnet_history['val_accuracy'][-1]:.4f}")
print(f"  val_mAP      (last epoch) : {resnet_history['val_mAP'][-1]:.4f}")
print(f"  Training time             : {resnet_time/60:.1f} minutes")# ── CELL 12: Build DenseNet121 ────────────────────────────────────────────

densenet_model, densenet_base = build_transfer_model(
    backbone_fn=DenseNet121,
    model_name="DenseNet121"
)

print(f"DenseNet121 total layers  : {len(densenet_model.layers)}")
print(f"Total parameters          : {densenet_model.count_params():,}")
print(f"Trainable parameters      : {sum(v.numpy().size for v in densenet_model.trainable_variables):,}")
print(f"Non-trainable parameters  : {sum(v.numpy().size for v in densenet_model.non_trainable_variables):,}")
print("\nModel head:")
for layer in densenet_model.layers[-6:]:
    print(f"  Layer: {layer.name}  |  Trainable: {layer.trainable}")# ── CELL 13: Train DenseNet121 ────────────────────────────────────────────

densenet_history, densenet_time = train_model(
    model=densenet_model,
    base=densenet_base,
    model_name="DenseNet121"
)

print(f"\nDenseNet121 Final Results:")
print(f"  val_accuracy (last epoch) : {densenet_history['val_accuracy'][-1]:.4f}")
print(f"  val_mAP      (last epoch) : {densenet_history['val_mAP'][-1]:.4f}")
print(f"  Training time             : {densenet_time/60:.1f} minutes")# ── CELL 14: Build MobileNetV3Large ──────────────────────────────────────
# Note: include_preprocessing=False because we already normalise to [0,1].
# MobileNetV3's built-in preprocessing normalises to [-1, 1], which would
# conflict with our normalisation if not disabled.

mobilenet_model, mobilenet_base = build_transfer_model(
    backbone_fn=MobileNetV3Large,
    model_name="MobileNetV3Large",
    include_preprocessing=True      # Pass True so we set include_preprocessing=False
)

print(f"MobileNetV3Large total layers : {len(mobilenet_model.layers)}")
print(f"Total parameters              : {mobilenet_model.count_params():,}")
print(f"Trainable parameters          : {sum(v.numpy().size for v in mobilenet_model.trainable_variables):,}")
print(f"Non-trainable parameters      : {sum(v.numpy().size for v in mobilenet_model.non_trainable_variables):,}")
print("\nModel head:")
for layer in mobilenet_model.layers[-6:]:
    print(f"  Layer: {layer.name}  |  Trainable: {layer.trainable}")# ── CELL 15: Train MobileNetV3Large ──────────────────────────────────────

mobilenet_history, mobilenet_time = train_model(
    model=mobilenet_model,
    base=mobilenet_base,
    model_name="MobileNetV3Large"
)

print(f"\nMobileNetV3Large Final Results:")
print(f"  val_accuracy (last epoch) : {mobilenet_history['val_accuracy'][-1]:.4f}")
print(f"  val_mAP      (last epoch) : {mobilenet_history['val_mAP'][-1]:.4f}")
print(f"  Training time             : {mobilenet_time/60:.1f} minutes")# ── CELL 16: Summary Table ────────────────────────────────────────────────

models_info = {
    'ResNet50':        (resnet_model,     resnet_history,    resnet_time),
    'DenseNet121':     (densenet_model,   densenet_history,  densenet_time),
    'MobileNetV3Large':(mobilenet_model,  mobilenet_history, mobilenet_time),
}

rows = []
for name, (mdl, hist, t) in models_info.items():
    rows.append({
        'Model'             : name,
        'Parameters'        : f"{mdl.count_params():,}",
        'Best Val Accuracy' : f"{max(hist['val_accuracy']):.4f}",
        'Final Val Accuracy': f"{hist['val_accuracy'][-1]:.4f}",
        'Best val_mAP'      : f"{max(hist['val_mAP']):.4f}",
        'Final val_mAP'     : f"{hist['val_mAP'][-1]:.4f}",
        'Training Time'     : f"{t/60:.1f} min",
    })

summary_df = pd.DataFrame(rows).set_index('Model')
print("Model Comparison Summary:")
print(summary_df)# ── CELL 17: Training Curves ──────────────────────────────────────────────
# 3 rows (one per model) × 3 columns (accuracy, loss, mAP)

model_names  = ['ResNet50', 'DenseNet121', 'MobileNetV3Large']
histories    = [resnet_history, densenet_history, mobilenet_history]
colors       = [('#1976D2', '#F57C00'), ('#388E3C', '#D32F2F'), ('#7B1FA2', '#FFA000')]
epochs_range = range(1, TOTAL_EPOCHS + 1)

fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.suptitle('Training History - All Models', fontsize=16, fontweight='bold')

for row, (name, hist, (c1, c2)) in enumerate(zip(model_names, histories, colors)):

    # Accuracy
    ax = axes[row, 0]
    ax.plot(epochs_range, hist['accuracy'],     color=c1, label='Train', linewidth=1.5)
    ax.plot(epochs_range, hist['val_accuracy'], color=c2, label='Val',   linewidth=1.5, linestyle='--')
    ax.axvline(x=PHASE1_EPOCHS, color='grey', linestyle=':', alpha=0.7, label='Phase 1/2 boundary')
    ax.set_title(f'{name} - Accuracy', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Loss
    ax = axes[row, 1]
    ax.plot(epochs_range, hist['loss'],     color=c1, label='Train', linewidth=1.5)
    ax.plot(epochs_range, hist['val_loss'], color=c2, label='Val',   linewidth=1.5, linestyle='--')
    ax.axvline(x=PHASE1_EPOCHS, color='grey', linestyle=':', alpha=0.7)
    ax.set_title(f'{name} - Loss', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # mAP
    ax = axes[row, 2]
    ax.plot(epochs_range, hist['val_mAP'], color=c2, label='val_mAP', linewidth=2)
    ax.axvline(x=PHASE1_EPOCHS, color='grey', linestyle=':', alpha=0.7, label='Phase 1/2 boundary')
    ax.set_title(f'{name} - mAP', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('mAP')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
# plt.show()
print("Training curves saved as training_curves.png")# ── CELL 18: Final Evaluation on Test Set ────────────────────────────────
# Load best saved checkpoints and evaluate on the held-out test set.

def evaluate_on_test(model_name, model_obj):
    """
    Loads the best checkpoint for model_name, runs inference on
    the test set, and returns y_true, y_pred_probs, y_pred_labels.
    """
    ckpt = os.path.join(SAVE_DIR, f"{model_name}_best.h5")
    if os.path.exists(ckpt):
        model_obj.load_weights(ckpt)
        print(f"  Loaded best weights: {ckpt}")
    else:
        print(f"  No checkpoint found for {model_name}, using current weights.")

    y_true_all, y_pred_probs_all = [], []

    for x_batch, y_batch in test_ds:
        preds = model_obj.predict(x_batch, verbose=0)
        y_true_all.extend(y_batch.numpy().tolist())
        y_pred_probs_all.extend(preds.tolist())

    y_true  = np.array(y_true_all)
    y_probs = np.array(y_pred_probs_all)
    y_pred  = np.argmax(y_probs, axis=1)

    # Metrics
    acc  = np.mean(y_true == y_pred)
    y_oh = keras.utils.to_categorical(y_true, N_CLASSES)
    mAP  = average_precision_score(y_oh, y_probs, average='macro')

    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test mAP      : {mAP:.4f}")
    print()
    print("  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    return y_true, y_probs, y_pred, acc, mAP


print("\n" + "="*65)
print("  TEST SET EVALUATION")
print("="*65)

print("\nResNet50:")
r_yt, r_yp, r_yl, r_acc, r_map = evaluate_on_test("ResNet50",        resnet_model)

print("\nDenseNet121:")
d_yt, d_yp, d_yl, d_acc, d_map = evaluate_on_test("DenseNet121",     densenet_model)

print("\nMobileNetV3Large:")
m_yt, m_yp, m_yl, m_acc, m_map = evaluate_on_test("MobileNetV3Large",mobilenet_model)# ── CELL 19: Confusion Matrices ───────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Confusion Matrices - Test Set', fontsize=15, fontweight='bold')

test_results = [
    ('ResNet50',         r_yt, r_yl),
    ('DenseNet121',      d_yt, d_yl),
    ('MobileNetV3Large', m_yt, m_yl),
]

short_names = ['Bamboo', 'Conifer', 'Mangrove', 'Tropical']

for ax, (name, y_true, y_pred) in zip(axes, test_results):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # as %

    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=short_names, yticklabels=short_names,
        ax=ax, cbar=True, linewidths=0.5
    )
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
# plt.show()
print("Confusion matrices saved as confusion_matrices.png")
print("Values shown as percentage of true class (row-normalised).")# ── CELL 20: Final Comparison Bar Charts ─────────────────────────────────

model_labels = ['ResNet50', 'DenseNet121', 'MobileNetV3']
test_accs   = [r_acc, d_acc, m_acc]
test_maps   = [r_map, d_map, m_map]
train_times = [resnet_time/60, densenet_time/60, mobilenet_time/60]
params_m    = [resnet_model.count_params()/1e6,
               densenet_model.count_params()/1e6,
               mobilenet_model.count_params()/1e6]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Final Model Comparison (Test Set)', fontsize=15, fontweight='bold')
palette = ['#1976D2', '#388E3C', '#7B1FA2']
x = np.arange(len(model_labels))

# Accuracy
ax = axes[0, 0]
bars = ax.bar(x, test_accs, color=palette, edgecolor='grey', width=0.5)
ax.set_title('Test Accuracy', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(model_labels)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Accuracy')
for bar, v in zip(bars, test_accs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# mAP
ax = axes[0, 1]
bars = ax.bar(x, test_maps, color=palette, edgecolor='grey', width=0.5)
ax.set_title('Test mAP (mean Average Precision)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(model_labels)
ax.set_ylim(0, 1.1)
ax.set_ylabel('mAP')
for bar, v in zip(bars, test_maps):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Training Time
ax = axes[1, 0]
bars = ax.bar(x, train_times, color=palette, edgecolor='grey', width=0.5)
ax.set_title('Training Time (minutes)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(model_labels)
ax.set_ylabel('Minutes')
for bar, v in zip(bars, train_times):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, f'{v:.1f}m',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Parameter Count
ax = axes[1, 1]
bars = ax.bar(x, params_m, color=palette, edgecolor='grey', width=0.5)
ax.set_title('Model Parameters (Millions)', fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(model_labels)
ax.set_ylabel('Parameters (M)')
for bar, v in zip(bars, params_m):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, f'{v:.1f}M',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
# plt.show()
print("Comparison chart saved as model_comparison.png")