import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
N_CLASSES = 4
CLASS_NAMES = ['bamboo_forest', 'coniferous_pine_forest', 'mangrove_forest', 'tropical_rainforest']
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
TEST_DIR = r'.\forestry-image-dataset\forestry_dataset_splitted\test'
SAVE_DIR = './model_checkpoints'

def build_test_dataset(split_dir):
    cls_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    file_paths, labels = [], []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(split_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(tuple(VALID_EXTS)):
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(cls_to_idx[cls])
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    def load_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = build_test_dataset(TEST_DIR)

def evaluate_on_test(model_name):
    ckpt = os.path.join(SAVE_DIR, f'{model_name}_best.h5')
    print(f'Evaluating {model_name} from {ckpt}...')
    try:
        model = keras.models.load_model(ckpt, compile=False)
    except Exception as e:
        print(f'Load failed for {model_name}: {e}')
        return None, None, None, 0.0, 0.0

    y_true_all, y_pred_probs_all = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true_all.extend(y_batch.numpy().tolist())
        y_pred_probs_all.extend(preds.tolist())
    y_true = np.array(y_true_all)
    y_probs = np.array(y_pred_probs_all)
    y_pred = np.argmax(y_probs, axis=1)
    acc = np.mean(y_true == y_pred)
    y_oh = keras.utils.to_categorical(y_true, N_CLASSES)
    mAP = average_precision_score(y_oh, y_probs, average='macro')
    print(f'  Test Acc: {acc:.4f}, mAP: {mAP:.4f}')
    return y_true, y_probs, y_pred, acc, mAP

r_yt, r_yp, r_yl, r_acc, r_map = evaluate_on_test('ResNet50')
d_yt, d_yp, d_yl, d_acc, d_map = evaluate_on_test('DenseNet121')
m_yt, m_yp, m_yl, m_acc, m_map = evaluate_on_test('MobileNetV3Large')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Confusion Matrices', fontsize=15, fontweight='bold')
results = [('ResNet50', r_yt, r_yl), ('DenseNet121', d_yt, d_yl), ('MobileNetV3Large', m_yt, m_yl)]
for ax, (name, y_t, y_p) in zip(axes, results):
    if y_t is None: continue
    cm = confusion_matrix(y_t, y_p)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=['Bamboo', 'Conifer', 'Mangrove', 'Tropical'],
                yticklabels=['Bamboo', 'Conifer', 'Mangrove', 'Tropical'])
    ax.set_title(name)
plt.savefig('confusion_matrices.png')
print('Done!')
