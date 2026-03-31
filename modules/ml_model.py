"""
=============================================================================
MODUL MACHINE LEARNING
=============================================================================
Dua model terpisah:
  1. Model Manipulasi (ELA-based) — deteksi editing/forgery
  2. Model AI Detector — deteksi foto AI-generated vs foto asli
=============================================================================
"""

import os, logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, applications, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow tidak tersedia. Instal: pip install tensorflow")


def _check_tf():
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow diperlukan. Instal: pip install tensorflow")


# ---------------------------------------------------------------------------
# Model 1: Deteksi Manipulasi (input = ELA features)
# ---------------------------------------------------------------------------

def build_manipulation_model(input_shape=(128, 128, 3)):
    """
    CNN untuk deteksi manipulasi berbasis fitur ELA.
    Label: 0 = AUTHENTIC, 1 = MANIPULATED
    """
    _check_tf()
    base = applications.MobileNetV2(weights="imagenet", include_top=False,
                                    input_shape=input_shape)
    base.trainable = False

    inp  = tf.keras.Input(shape=input_shape)
    x    = applications.mobilenet_v2.preprocess_input(inp)
    x    = base(x, training=False)
    x    = layers.GlobalAveragePooling2D()(x)
    x    = layers.Dense(128, activation="relu")(x)
    x    = layers.Dropout(0.5)(x)
    out  = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out, name="ManipulationDetector")
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    logger.info(f"Model Manipulasi: {model.count_params():,} parameter")
    return model


# ---------------------------------------------------------------------------
# Model 2: Deteksi Foto AI (input = raw pixel + ELA gabungan)
# ---------------------------------------------------------------------------

def build_ai_detector_model(input_shape=(128, 128, 3)):
    """
    CNN untuk deteksi foto AI-generated.
    Label: 0 = REAL_PHOTO, 1 = AI_GENERATED

    Arsitektur sedikit berbeda karena karakteristik AI-generated image
    lebih bergantung pada pola frekuensi dan tekstur halus.
    """
    _check_tf()

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Blok 1 — tangkap frekuensi rendah
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Blok 2 — tangkap tekstur menengah
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Blok 3 — tangkap pola tingkat tinggi
        layers.Conv2D(128, (5, 5), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),  # 0=real, 1=AI
    ], name="AIDetector")

    model.compile(optimizer=optimizers.Adam(5e-5),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    logger.info(f"Model AI Detector: {model.count_params():,} parameter")
    return model


# ---------------------------------------------------------------------------
# Training (generik untuk kedua model)
# ---------------------------------------------------------------------------

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=30, batch_size=16, save_path=None):
    _check_tf()
    cb = [
        callbacks.EarlyStopping(monitor="val_auc", patience=7,
                                restore_best_weights=True, mode="max"),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                   patience=3, min_lr=1e-7, verbose=1),
    ]
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cb.append(callbacks.ModelCheckpoint(save_path, monitor="val_auc",
                                            save_best_only=True, mode="max"))
    return model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=batch_size, callbacks=cb)


def evaluate_model(model, X_test, y_test):
    _check_tf()
    from sklearn.metrics import classification_report, confusion_matrix
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).flatten()
    return {
        "loss": loss, "accuracy": acc, "auc": auc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def save_model(model, path):
    _check_tf()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)


def load_model(path):
    _check_tf()
    return tf.keras.models.load_model(path)


# ---------------------------------------------------------------------------
# Dataset builder untuk AI Detector
# ---------------------------------------------------------------------------

def build_ai_dataset(real_dir: str, ai_dir: str,
                     target_size=(128, 128)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bangun dataset untuk training model AI Detector.
    Label: 0 = REAL_PHOTO, 1 = AI_GENERATED

    Args:
        real_dir  : folder berisi foto asli dari kamera
        ai_dir    : folder berisi foto AI-generated
    """
    from ai_detector import extract_ai_features
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    X, y = [], []

    for folder, label in [(real_dir, 0), (ai_dir, 1)]:
        if not os.path.isdir(folder):
            logger.warning(f"Folder tidak ditemukan: {folder}")
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in supported]
        logger.info(f"Memproses {len(files)} citra dari {folder} (label={label})...")
        for path in files:
            try:
                feat = extract_ai_features(path, target_size)
                X.append(feat)
                y.append(label)
            except Exception as e:
                logger.warning(f"Lewati {path}: {e}")

    if not X:
        raise ValueError("Dataset kosong.")
    return np.stack(X), np.array(y, dtype=np.int32)
