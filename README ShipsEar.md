# DWSTr: Hybrid CNN-Transformer for ShipsEar Acoustic Classification

## Overview

This repository contains a complete, end-to-end implementation of an audio classification model designed to identify 12 distinct types of vessels and ambient ocean noises using their underwater acoustic signatures.

The core of this project is the **DWSTr** (Depthwise Separable Convolution and Transformer) model. This hybrid deep learning architecture processes raw underwater signals by combining the lightweight spatial feature extraction of Depthwise Separable Convolutions (DWS) with the powerful long-range temporal modeling of Vision Transformer (ViT) encoders.

---

## Architecture Details

The DWSTr model contains **1,339,866 parameters** and processes inputs through a carefully designed sequence of feature-learning phases:

1. **Data Augmentation (`SpecAugmentLayer`):** Applies dynamic time and frequency masking during training (up to 10 frequency bins and 1 time step) to heavily penalize overfitting and force the model to learn robust acoustic features.


2. **DWS Block (Spatial Features):**
* A 3x3 Depthwise Convolutional Layer captures localized frequency-time relationships.


* A 1x1 Pointwise Convolutional Layer maps cross-channel interactions, outputting 64 distinct feature maps.




3. **Patch Embedding:** The resulting 128x4 feature map is flattened into a sequence of 32 patches, followed by a linear projection into a 64-dimensional space. A learnable Class Token and standard Positional Embeddings are concatenated.


4. **Transformer Encoder (Temporal Features):**
* 6 sequential Transformer blocks using a **Pre-Layer Normalization** architecture.


* Multi-Head Attention (4 heads).


* MLP block (1024 dimension) utilizing GELU activation and a **30%** dropout rate.




5. **Classification Head:** Extracts the updated class token and processes it through a dense network to output a softmax probability distribution across the 12 acoustic classes.



---

## Dataset & Preprocessing Pipeline

The model utilizes the **ShipsEar** database. Processing the raw `.wav` files requires a strict pipeline to extract optimal features before they reach the network.

* **Resampling:** Audio is loaded and standardized to **22,050 Hz**.


* **Segmentation:** The audio is sliced into uniform **75ms** non-overlapping segments (yielding 1,654 samples per segment). This resulted in a massive dataset expansion to **151,142 total segments**.


* **Pre-emphasis:** A high-pass filter is applied to amplify higher frequencies: $y[n] = x[n] - 0.97 \cdot x[n-1]$.


* **Mel-Spectrograms:** Extracted using an FFT window of 2048, a hop length of 512, and 128 Mel bins, yielding a final tensor shape of **(128, 4, 1)** per segment.



---

## Training Configuration

The dataset was split using a stratified approach to maintain class distribution: **70% Training** (105,799 samples), **10% Validation** (15,115 samples), and **20% Testing** (30,228 samples).

* **Optimizer:** Adam (Initial LR = 0.001, Weight Decay = 0.0001)


* **Loss Function:** Sparse Categorical Crossentropy


* **Batch Size:** 256


* **Class Weighting:** Applied directly via `sklearn`'s balanced compute function to handle the severe dataset imbalance (where majority classes dominate minority ones like Pilot Ships and Dredgers).



**Callbacks & Convergence:**

* `EarlyStopping` monitored validation loss with a patience of 5 epochs.


* `ReduceLROnPlateau` decayed the learning rate by a factor of 0.5 upon a 3-epoch plateau.


* Training automatically halted at **Epoch 73**, restoring the optimal weights from **Epoch 68**, having dropped the learning rate to **0.0000625**.



---

## Final Results & Performance

Evaluated against the completely held-out test set, the DWSTr architecture demonstrated excellent discriminative learning across complex underwater acoustic signatures.

* **Overall Test Accuracy:** **94.89%**

* **Test Loss:** **0.1518**


### Per-Class Evaluation

| Class | Precision | Recall | F1-Score | Support | Accuracy |
| --- | --- | --- | --- | --- | --- |
| Dredger | 99.01% | 100.00% | 99.50% | 703 | 100.00% |
| Fishboat | 94.40% | 99.13% | 96.71% | 1,378 | 99.13% |
| Motorboat | 89.94% | 94.63% | 92.23% | 2,720 | 94.63% |
| Mussel Boat | 89.92% | 97.54% | 93.57% | 1,948 | 97.54% |
| Natural Ambient Noise | 95.92% | 97.74% | 96.82% | 3,057 | 97.74% |
| Ocean Liner | 93.42% | 97.86% | 95.59% | 2,524 | 97.86% |
| Passengers | 98.86% | 90.54% | 94.52% | 11,410 | 90.54% |
| Pilot Ship | 95.31% | 99.19% | 97.21% | 369 | 99.19% |
| Roro | 94.65% | 98.12% | 96.35% | 4,040 | 98.12% |
| Sailboat | 83.14% | 96.70% | 89.41% | 1,091 | 96.70% |
| Trawler | 96.84% | 98.17% | 97.50% | 437 | 98.17% |
| Tugboat | 88.89% | 97.28% | 92.89% | 551 | 97.28% |

**Key Takeaways:**

* The balanced class weighting strategy is highly successful; the model detects minority classes almost flawlessly (e.g., **100.00%** on Dredgers, **99.19%** on Pilot Ships).


* Confusion patterns indicate slight acoustic overlap in specific vessel categories, leaving room for targeted tuning to improve the precision of Sailboats and the recall of Passenger ships.
