# Genre-based Music Classifier (GTZAN) — CNN on Spectrograms

Academic project (EE798Z) under Prof. Sandip Tiwari.  
Goal: build a classical audio pipeline that **extracts features → creates spectrograms → trains a CNN** to classify songs by **genre**.

---

## ✨ Summary

- Processed raw audio and extracted features to train a **categorization model** for music **genre classification**.
- Used **Fast Fourier Transform (FFT)** via short-time windows to generate **spectrograms** (Librosa).
- Built a **TensorFlow CNN** that consumes spectrogram images and outputs **10-way** genre predictions.
- Ran an **ablation** to compare performance **with vs. without Conv2D layers** (convolutional stack vs. a shallow non-conv baseline).
- Reached **~83% accuracy** on GTZAN using **cross-entropy loss**, validating the effectiveness of a CNN over spectrograms.

---

## 📚 Dataset

- **GTZAN**: 10 genres, 100 tracks each (30s clips).  
- Typical split: train/valid/test with artist/track shuffling to reduce leakage (kept consistent across runs).

> This project treats each 30s clip as one example after converting it to a fixed-size log-spectrogram.

---

## 🔄 Pipeline

1. **Load & standardize**: mono, `sr=22,050`, duration trimmed/padded to a fixed window.  
2. **Spectrograms (FFT/STFT)**: compute magnitude spectrograms and convert to **log-scaled (dB)** for stability.  
3. **Normalization** and resizing to a consistent `H×W` before batching.  
4. **Modeling**:  
   - **CNN** on spectrograms (Conv → BN → ReLU → Pool stacks → Dense → Softmax).  
   - **Baseline**: non-conv model (e.g., MLP on flattened spectrogram or shallow global pooling head).  
5. **Training**: categorical cross-entropy, Adam, early stopping.  
6. **Evaluation**: accuracy (primary), confusion matrix for per-genre insights.

---

## 🔧 Minimal code references

### Feature extraction (Librosa → log-spectrogram)
```python
import numpy as np, librosa

def load_log_spectrogram(path, sr=22050, n_fft=2048, hop=512):
    y, sr = librosa.load(path, sr=sr, mono=True)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    return S_db  # (freq_bins, time_frames)
````

### TensorFlow CNN (spectrogram → probabilities)

```python
import tensorflow as tf
from tensorflow.keras import layers as L, models as M

def make_cnn(input_shape=(224, 224, 1), num_classes=10):
    x_in = L.Input(shape=input_shape)
    x = L.Conv2D(32, 3, padding="same")(x_in); x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.MaxPool2D()(x)
    x = L.Conv2D(64, 3, padding="same")(x); x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.MaxPool2D()(x)
    x = L.Conv2D(128, 3, padding="same")(x); x = L.BatchNormalization()(x); x = L.ReLU()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    x = L.Dense(128, activation="relu")(x)
    out = L.Dense(num_classes, activation="softmax")(x)
    model = M.Model(x_in, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
```

### Non-conv baseline (for the ablation)

```python
def make_nonconv_baseline(input_shape=(224, 224, 1), num_classes=10):
    x_in = L.Input(shape=input_shape)
    x = L.Flatten()(x_in)
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(0.4)(x)
    out = L.Dense(num_classes, activation="softmax")(x)
    model = M.Model(x_in, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
```

---

## 📈 Results (snapshot)

* **CNN on log-spectrograms** (TensorFlow): **\~83% accuracy** (cross-entropy).
* **Without Conv2D** (non-conv baseline): notably lower accuracy, confirming the benefit of convolution on time-frequency images.
* Qualitatively, confusion tends to occur between closely related genres (e.g., rock/metal, pop/disco), which is consistent with literature.

> Exact numbers vary slightly by split/seed; the project emphasizes the **relative gain** from convolution and spectrogram inputs.

---

## 🧪 Notes & Choices

* **FFT/STFT** chosen for transparency; Mel-spectrograms also tested with similar trends.
* **Input size** normalized (e.g., 224×224) to balance resolution vs. compute.
* **Regularization**: BatchNorm + Dropout (0.3–0.4).
* **Deterministic decoding** is not relevant; classification uses softmax over 10 genres.

---

## ⚠️ Limitations

* GTZAN is small and can be sensitive to split strategies; results should be viewed as **course-scale** evidence, not SOTA claims.
* Limited data augmentation (pitch/time shifts) — a promising direction for gains.
* No domain adaptation to real-world streaming or multi-label tagging.

---

## 📂 Repository notes

Typical assets you’ll see here:

* **Feature extraction notebooks** (Librosa, FFT/STFT, optional melody/tempo features).
* **CNN training notebook** that consumes spectrogram tensors.
* Small utilities to standardize I/O, label encoding, and spectrogram shaping.

---

## 🙏 Acknowledgments

* **Librosa** for audio feature extraction.
* **TensorFlow/Keras** for modeling.
* **GTZAN** dataset for academic use.

```
