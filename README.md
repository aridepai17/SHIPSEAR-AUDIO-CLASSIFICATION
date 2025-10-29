# ShipsEar Audio Classification with DWSTr (Depthwise Separable Convolution and Transformer)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Architecture Overview](#architecture-overview)
- [Detailed Notebook Walkthrough](#detailed-notebook-walkthrough)
  - [Cell 1: Setup and Environment Configuration](#cell-1-setup-and-environment-configuration)
  - [Cell 2: DWSTr Preprocessor Class](#cell-2-dwstr-preprocessor-class)
  - [Cell 3: Metadata and Data Loading Functions](#cell-3-metadata-and-data-loading-functions)
  - [Cell 4: Execute Preprocessing and Save Data](#cell-4-execute-preprocessing-and-save-data)
  - [Cell 5: Data Splitting and Final Save](#cell-5-data-splitting-and-final-save)
  - [Cell 6: Load Final Data and Sanity Check](#cell-6-load-final-data-and-sanity-check)
  - [Cell 7: Import TensorFlow/Keras and Configure Environment](#cell-7-import-tensorflowkeras-and-configure-environment)
  - [Cell 8: Depthwise Separable Convolution (DWS) Block](#cell-8-depthwise-separable-convolution-dws-block)
  - [Cell 9: Transformer Encoder Block](#cell-9-transformer-encoder-block)
  - [Cell 9.5: Spec-Augmentation Layer](#cell-95-spec-augmentation-layer)
  - [Cell 10: Complete DWSTr Architecture](#cell-10-complete-dwstr-architecture)
  - [Cell 11: Build and Compile Model](#cell-11-build-and-compile-model)
  - [Cell 12: Setup Training Callbacks](#cell-12-setup-training-callbacks)
  - [Cell 13: Train the Model](#cell-13-train-the-model)
  - [Cell 14: Evaluate on Test Set](#cell-14-evaluate-on-test-set)
  - [Cell 15: Plot Training History](#cell-15-plot-training-history)
  - [Cell 16: Detailed Per-Class Evaluation](#cell-16-detailed-per-class-evaluation)
- [Results and Performance](#results-and-performance)
- [Best Practices and Design Decisions](#best-practices-and-design-decisions)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This project provides a complete, end-to-end implementation of an audio classification model designed to identify different types of ships based on their underwater acoustic signatures. The implementation is contained within a single Jupyter Notebook (`ShipsEar.ipynb`) and is optimized for execution in a Google Colab environment with GPU acceleration.

The core innovation of this project is the **DWSTr model**—a hybrid deep learning architecture that synergistically combines:

- **Depthwise Separable Convolutions (DWS)**: For efficient spatial feature extraction from Mel-spectrograms with significantly reduced computational overhead compared to standard convolutions.
- **Transformer Encoders**: For capturing complex temporal dependencies and long-range relationships within audio sequences using self-attention mechanisms.

This architecture represents a state-of-the-art approach to audio classification, achieving **94.89% accuracy** on the ShipsEar dataset, which contains 12 distinct ship type classes.

---

## Key Features

- ✅ **End-to-End Pipeline**: Seamless workflow from raw audio files to fully trained and evaluated deep learning model
- ✅ **Efficient Preprocessing**: Optimized audio-to-Mel-spectrogram conversion pipeline with 75ms segmentation strategy
- ✅ **Advanced Data Augmentation**: SpecAugment implementation for frequency and time masking to enhance model robustness
- ✅ **Hybrid Architecture**: Novel combination of CNN and Transformer components for spatial and temporal feature learning
- ✅ **Production-Ready Training**: Incorporates best practices including class weighting, learning rate scheduling, early stopping, and model checkpointing
- ✅ **Comprehensive Evaluation**: Detailed metrics including per-class performance, confusion matrices, and training visualization

---

## Dependencies

The project requires the following Python libraries. All dependencies are automatically installed in the first cell of the notebook.

| Library | Purpose | Version |
|---------|---------|---------|
| `tensorflow` / `keras` | Deep learning framework | ≥2.10.0 |
| `librosa` | Audio signal processing and feature extraction | ≥0.9.0 |
| `soundfile` | Audio file I/O operations | ≥0.10.0 |
| `pandas` | Metadata handling and data manipulation | ≥1.3.0 |
| `scikit-learn` | Data splitting, evaluation metrics, and class weighting | ≥1.0.0 |
| `tqdm` | Progress bars for long-running operations | ≥4.60.0 |
| `matplotlib` | Plotting and visualization | ≥3.5.0 |
| `seaborn` | Statistical visualization (confusion matrices) | ≥0.11.0 |
| `numpy` | Numerical computing | ≥1.21.0 |

---

## Architecture Overview

The DWSTr model follows this architectural flow:

```
Input Mel-Spectrogram (128×4×1)
         ↓
[SpecAugment Layer] ← (Training only: frequency & time masking)
         ↓
[Depthwise Separable Conv Block] ← Spatial feature extraction
         ↓
[Patch Embedding] ← Reshape 2D feature map → 1D sequence
         ↓
[Class Token + Positional Embedding] ← Vision Transformer prep
         ↓
[Transformer Encoder × 6] ← Temporal relationship learning
         ↓
[Classification Head] ← Final prediction (12 classes)
```

**Key Design Principles:**
1. **Efficiency**: DWS reduces parameters while maintaining feature extraction capability
2. **Temporal Modeling**: Transformers capture long-range dependencies in audio sequences
3. **Regularization**: SpecAugment + Dropout prevent overfitting
4. **Class Balancing**: Weighted loss handles imbalanced dataset

---

## Detailed Notebook Walkthrough

This section provides an in-depth, technical explanation of each cell in the `ShipsEar.ipynb` notebook. Each explanation includes:
- **What happens**: Step-by-step code execution
- **Why it happens**: The motivation and theory behind each component
- **How it works**: Implementation details and mathematical underpinnings
- **Technical considerations**: Design decisions, edge cases, and best practices

---

### Cell 1: Setup and Environment Configuration

#### **What Happens**

This cell initializes the Google Colab environment and prepares all necessary tools for the project.

**Step-by-Step Execution:**

1. **Google Drive Mounting**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   - Establishes a connection between the Colab runtime and your Google Drive
   - Creates a mount point at `/content/drive` for persistent file access
   - Requires user authentication on first execution

2. **Package Installation**
   ```python
   !pip install librosa soundfile pandas tqdm scikit-learn -q
   ```
   - Installs required libraries directly into the Colab environment
   - The `-q` flag suppresses verbose output for cleaner execution
   - Uses `!` (shell command) instead of `%pip` for compatibility

3. **Library Imports**
   ```python
   import numpy as np
   import pandas as pd
   import librosa
   # ... etc
   ```
   - Loads essential modules for the entire pipeline
   - `warnings.filterwarnings('ignore')` suppresses librosa deprecation notices

4. **Path Configuration**
   ```python
   DATASET_ROOT_PATH = '/content/drive/MyDrive/...'
   METADATA_CSV_PATH = '/content/drive/MyDrive/...'
   SAVE_DIR = '/content/drive/MyDrive/...'
   ```
   - Defines absolute paths to dataset and output directories
   - Creates `SAVE_DIR` if it doesn't exist using `os.makedirs(SAVE_DIR, exist_ok=True)`

#### **Why This Matters**

- **Persistent Storage**: Colab's file system is ephemeral. Mounting Drive ensures your data survives session disconnections and allows you to reuse processed data across sessions.
- **Reproducibility**: Centralizing path definitions makes it easy to adapt the code to different environments or dataset locations.
- **Memory Management**: By saving intermediate results to Drive, you avoid keeping all data in RAM simultaneously, which is critical for large audio datasets.

#### **Technical Considerations**

- **Path Format**: Always use absolute paths in Colab. Relative paths can fail after directory changes.
- **Drive Mount Timeout**: Google Drive mounts can timeout after periods of inactivity. Re-run this cell if you get "file not found" errors.
- **Permission Handling**: First-time mounting requires Google account authentication—follow the browser prompts.

---

### Cell 2: DWSTr Preprocessor Class

#### **What Happens**

This cell defines a comprehensive audio preprocessing class that transforms raw audio waveforms into Mel-spectrogram tensors suitable for deep learning.

**Class Structure:**

```python
class DWSTrPreprocessor:
    def __init__(self)          # Initialize hyperparameters
    def preemphasis(self)       # High-frequency boost filter
    def segment_audio(self)     # 75ms windowing
    def extract_mel_spectrogram(self)  # Frequency domain conversion
    def process_file(self)      # Complete pipeline orchestrator
```

#### **Deep Dive into Each Method**

**1. Initialization (`__init__`)**
```python
self.sr = 22050                    # Sampling rate (Hz)
self.segment_duration = 0.075      # 75 milliseconds
self.segment_samples = 1654        # Samples per segment
self.n_fft = 2048                  # FFT window size
self.hop_length = 512              # Frame shift (overlap control)
self.n_mels = 128                  # Mel filterbank bins
self.fmax = 11025                  # Maximum frequency (Nyquist)
```

**Why These Values?**
- **22050 Hz Sampling Rate**: Standard for audio ML (reduces computational load vs 44.1 kHz while preserving most acoustic information)
- **75ms Segments**: Optimal trade-off between temporal resolution and feature richness. Too short = insufficient context; too long = reduced sample count
- **2048-point FFT**: Provides good frequency resolution (2048/22050 ≈ 93 ms window)
- **512 hop_length**: ~50% overlap ensures smooth temporal transitions
- **128 Mel bins**: Captures perceptual frequency scales (more bins than 64, fewer than 256)

**2. Pre-emphasis Filter (`preemphasis`)**
```python
def preemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return np.append(audio[0], audio[1:] - coef * audio[:-1])
```

**Mathematical Formulation:**
\[
y[n] = x[n] - \alpha \cdot x[n-1]
\]
where \(\alpha = 0.97\)

**Why Pre-emphasis?**
- **High-Frequency Boost**: Audio signals naturally decay at higher frequencies. Pre-emphasis compensates for this, making high-frequency features (like ship engine harmonics) more prominent.
- **Noise Reduction**: High-frequency noise reduction is easier when signal-to-noise ratio is improved.
- **Standard Practice**: Common preprocessing step in speech recognition and audio classification pipelines.

**3. Audio Segmentation (`segment_audio`)**
```python
def segment_audio(self, audio: np.ndarray) -> list[np.ndarray]:
    segments = []
    for start in range(0, len(audio) - self.segment_samples + 1, self.segment_samples):
        segment = audio[start:start + self.segment_samples]
        if len(segment) == self.segment_samples:
            segments.append(segment)
    return segments
```

**Key Details:**
- **Non-overlapping Windows**: Each segment starts exactly where the previous one ended (`step = segment_samples`)
- **Edge Handling**: The condition `len(audio) - self.segment_samples + 1` prevents partial segments
- **Data Augmentation Effect**: One audio file (e.g., 30 seconds) produces hundreds of training samples, dramatically increasing dataset size

**Example:**
- Input: 30-second audio @ 22050 Hz = 661,500 samples
- 75ms segments = 1,654 samples each
- Number of segments = 661,500 / 1,654 ≈ **400 segments**

**4. Mel-Spectrogram Extraction (`extract_mel_spectrogram`)**

**Step-by-Step Process:**

```python
# Step 1: Apply pre-emphasis
emphasized = self.preemphasis(segment)

# Step 2: Compute Mel-spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=emphasized,
    sr=self.sr,
    n_fft=self.n_fft,           # 2048-point FFT
    hop_length=self.hop_length,  # 512-sample hop
    n_mels=self.n_mels,          # 128 mel bins
    power=2.0,                   # Power spectrum (magnitude squared)
    window='hann'                # Hann windowing function
)

# Step 3: Convert to decibel scale
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
```

**Mel-Scale Transformation:**
The Mel-scale approximates human auditory perception, where low frequencies have higher resolution than high frequencies:
\[
m = 2595 \log_{10}(1 + f/700)
\]

**Power-to-dB Conversion:**
\[
\text{Mel\_spec\_db} = 10 \log_{10}(\text{Mel\_spec}) - \max(10 \log_{10}(\text{Mel\_spec}))
\]
Logarithmic scaling ensures that quiet and loud components are both represented in the visualizable range (typically -80 to 0 dB).

**Shape Guarantee:**
```python
expected_frames = 4
if mel_spec_db.shape[1] != expected_frames:
    # Padding or trimming to ensure consistent (128, 4) shape
```

**Why 128×4?**
- **128 frequency bins**: Captures full audible spectrum (0-11,025 Hz) with good resolution
- **4 time frames**: For a 75ms segment with 512 hop_length: (1654 - 2048)//512 + 1 ≈ 4 frames
- **Consistent Input Size**: Fixed dimensions enable batched processing and transformer tokenization

**5. Complete Pipeline (`process_file`)**

This method orchestrates the entire preprocessing for a single audio file:
1. Load audio file (with resampling to 22050 Hz and mono conversion)
2. Segment into 75ms clips
3. Convert each segment to Mel-spectrogram
4. Return array of shape `(num_segments, 128, 4, 1)`

**Output Format:**
- Each file produces `N` segments (varies with file length)
- Each segment is `(128, 4, 1)` - ready for neural network input
- The trailing `1` dimension represents a single channel (grayscale spectrogram)

---

### Cell 3: Metadata and Data Loading Functions

#### **What Happens**

This cell implements three functions that handle metadata parsing, data cleaning, and the complete dataset loading pipeline.

**Function Breakdown:**

**1. `clean_class_names(df, class_column)`**

**Purpose:** Standardizes class labels to prevent duplicate categories from case mismatches.

```python
df[class_column] = df[class_column].str.title()
```

**Why This Matters:**
- Metadata files often contain inconsistent casing: "Fishboat", "fishboat", "FISHBOAT"
- Without normalization, the model treats these as **three separate classes**
- `str.title()` converts everything to Title Case: "Fishboat"

**Real-World Impact:**
```
Before: ['Dredger', 'dredger', 'DREDGER'] → 3 classes (wrong!)
After:  ['Dredger', 'Dredger', 'Dredger'] → 1 class (correct!)
```

**2. `create_id_to_class_map(metadata_path)`**

**Purpose:** Creates lookup dictionaries mapping audio file IDs to their corresponding ship type labels.

**Detailed Execution:**

```python
# Step 1: Read CSV file
df = pd.read_csv(metadata_path)

# Step 2: Clean class names
df = clean_class_names(df, CLASS_COLUMN)

# Step 3: Ensure ID is string (for filename matching)
df[ID_COLUMN] = df[ID_COLUMN].astype(str)

# Step 4: Create ID → Class Name dictionary
ID_TO_CLASS_MAP = df.drop_duplicates(...).set_index(ID_COLUMN)[CLASS_COLUMN].to_dict()

# Step 5: Create Class Name → Index mapping
class_names = sorted(df[CLASS_COLUMN].unique().tolist())
CLASS_TO_IDX = {name: i for i, name in enumerate(class_names)}
```

**Output Structure:**
```python
ID_TO_CLASS_MAP = {
    'ID001': 'Dredger',
    'ID002': 'Fishboat',
    # ... etc
}

CLASS_TO_IDX = {
    'Dredger': 0,
    'Fishboat': 1,
    'Motorboat': 2,
    # ... 12 total classes
}
```

**Why Two Mappings?**
- **ID_TO_CLASS_MAP**: Links filenames (which contain IDs) to their true labels during preprocessing
- **CLASS_TO_IDX**: Converts human-readable class names to integer indices (required by neural networks)

**3. `load_and_process_all_data(dataset_path, metadata_path)`**

**Purpose:** The main orchestrator that processes all audio files and generates the complete feature-label dataset.

**Detailed Flow:**

**Step 1: Initialize Preprocessor**
```python
preprocessor = DWSTrPreprocessor()
```
- Creates a single preprocessor instance (more efficient than recreating it per file)

**Step 2: Find All Audio Files**
```python
audio_files = glob.glob(os.path.join(dataset_path, '**', '*.wav'), recursive=True)
```
- Uses recursive glob to find `.wav` files in all subdirectories
- Handles nested folder structures automatically

**Step 3: Process Each File**
```python
for audio_file in tqdm(audio_files, desc="Preprocessing Audio Files"):
    filename = os.path.basename(audio_file)  # Extract filename
    file_id = filename.split('__')[0]       # Parse ID (format: ID__xx_yy_zz_name.wav)
    
    if file_id in ID_TO_CLASS_MAP:
        class_name = ID_TO_CLASS_MAP[file_id]
        class_idx = CLASS_TO_IDX[class_name]
        
        # Process audio → multiple Mel-spectrograms
        mel_specs = preprocessor.process_file(audio_file)
        
        # Extend lists (one label per segment)
        X_list.extend(mel_specs)
        y_list.extend([class_idx] * len(mel_specs))
```

**Key Implementation Details:**

- **Filename Parsing**: ShipsEar files follow format `ID__timestamp_name.wav`. We extract the `ID` part before `__`.
- **One-to-Many Mapping**: Each audio file produces multiple segments, all labeled with the same class. This is why we use `extend` with a list of repeated indices: `[class_idx] * len(mel_specs)`.
- **Error Handling**: The `try-except` block gracefully skips corrupted or non-standard files instead of crashing the entire pipeline.

**Step 4: Convert to NumPy Arrays**
```python
X = np.expand_dims(np.array(X_list), axis=-1)  # Add channel dimension: (N, 128, 4) → (N, 128, 4, 1)
y = np.array(y_list)                            # Integer labels: (N,)
```

**Why `expand_dims(axis=-1)`?**
- TensorFlow/Keras expects a channel dimension for 2D inputs
- `(128, 4)` becomes `(128, 4, 1)`, consistent with image processing conventions
- The `-1` index means "add at the end"

**Memory Management:**
```python
del X_list, y_list  # Explicitly free memory after NumPy conversion
```
- Python lists consume more memory than NumPy arrays
- Deleting the lists immediately after conversion prevents memory bloat during large dataset processing

**Output Summary:**
```
Preprocessing Complete: 151,142 total segments generated
Initial Feature Shape: (151142, 128, 4, 1)
```
- From ~90 original audio files, we generated **151,142 training samples**
- Each sample is a 75ms Mel-spectrogram ready for neural network input

---

### Cell 4: Execute Preprocessing and Save Data

#### **What Happens**

This cell executes the preprocessing pipeline and immediately persists the results to disk.

**Execution Flow:**

```python
# 1. Run preprocessing (this can take 30+ minutes)
X_all, y_all, class_names = load_and_process_all_data(DATASET_ROOT_PATH, METADATA_CSV_PATH)

# 2. Save to compressed NumPy archive
np.savez_compressed(temp_save_path, X=X_all, y=y_all)

# 3. Free memory
del X_all, y_all

# 4. Save class names separately (for model loading later)
pickle.dump(class_names, open(class_names_path, 'wb'))
```

#### **Why This Cell Exists**

**1. Time Savings**
- Preprocessing is computationally expensive (FFT, Mel-transform, file I/O)
- Saving results means you only run preprocessing **once**
- Subsequent notebook runs can skip directly to model training

**2. Memory Efficiency**
- Large datasets can exceed Colab's RAM limits
- Saving and deleting allows the next cell (data splitting) to operate without memory pressure

**3. Version Control**
- `.npz` files can be versioned or backed up
- Enables reproducible experiments with consistent train/test splits

#### **Technical Details**

**NumPy Compressed Format:**
```python
np.savez_compressed('file.npz', X=X_all, y=y_all)
```
- `savez_compressed` uses `.npz` format with gzip compression
- Typically achieves 2-3x size reduction vs uncompressed `.npy`
- Load time is only slightly slower than uncompressed format

**File Structure:**
```
processed_data_DWSTr/
├── full_data_preprocessed.npz  # Complete dataset (X, y)
└── class_names.pkl              # Class name list
```

**Loading Saved Data:**
```python
data = np.load('full_data_preprocessed.npz')
X = data['X']
y = data['y']
```

**Why Separate Class Names?**
- Class names are Python lists (not NumPy arrays)
- Pickle is the standard Python serialization for complex objects
- Separate file makes it easy to inspect class order without loading full dataset

---

### Cell 5: Data Splitting and Final Save

#### **What Happens**

This cell performs a critical machine learning step: splitting the dataset into training, validation, and test sets.

**Two-Stage Splitting Strategy:**

```python
# Stage 1: Separate training from (test + validation)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,              # 30% for test+val combined
    stratify=y,                 # Maintain class distribution
    random_state=42              # Reproducibility
)

# Stage 2: Split temp into test and validation
val_ratio_adjusted = 0.1 / 0.3  # ≈ 0.333
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_ratio_adjusted,
    stratify=y_temp,
    random_state=42
)
```

**Final Distribution:**
- **Training**: 70% (105,799 samples)
- **Testing**: 20% (30,228 samples)
- **Validation**: 10% (15,115 samples)

#### **Why Three Splits?**

**Training Set (70%)**
- Used for gradient descent optimization
- Largest portion ensures model sees sufficient examples

**Validation Set (10%)**
- Monitors training progress during epochs
- Used by callbacks (early stopping, learning rate reduction)
- **Never used for final performance reporting** (would be biased)

**Test Set (20%)**
- **Completely held out** until final evaluation
- Provides unbiased estimate of real-world performance
- Only touched once—at the very end

#### **Stratified Splitting Explained**

**What is Stratification?**
```python
stratify=y  # Maintains class distribution in all splits
```

**Without Stratification (Bad):**
```
Training:   [70% Dredgers, 5% Pilot Ships]  ← Imbalanced
Validation: [10% Dredgers, 15% Pilot Ships]  ← Different distribution!
```
- Model might never see enough examples of rare classes during training
- Validation metrics become unreliable

**With Stratification (Good):**
```
Training:   [7% Dredgers, 0.24% Pilot Ships]  ← Matches original
Validation: [7% Dredgers, 0.24% Pilot Ships]    ← Same distribution
Test:       [7% Dredgers, 0.24% Pilot Ships]  ← Same distribution
```
- All splits have identical class proportions
- Ensures fair evaluation and training

#### **Random State for Reproducibility**

```python
random_state=42
```
- Same seed = same split every time
- Critical for comparing different model architectures
- Enables fair competition between experiments

#### **Memory-Aware Design**

```python
# Reload from disk (doesn't keep everything in RAM)
data = np.load(temp_save_path)
X = data['X']
y = data['y']

# ... perform splitting ...

# Save splits and immediately delete
del X, y, X_temp, y_temp
```

This pattern (load → process → save → delete) is essential for large datasets in memory-constrained environments.

---

### Cell 6: Load Final Data and Sanity Check

#### **What Happens**

This cell prepares the final dataset splits for model training by loading them from disk and performing integrity checks.

**Loading Function:**
```python
def load_final_data(data_dir: str):
    with np.load('train_data.npz') as data:
        X_train, y_train = data['X'], data['y']
    # ... load test and validation similarly
    return X_train, X_test, X_val, y_train, y_test, y_val, class_names
```

**Sanity Checks:**
```python
print(f"X_train shape: {X_train.shape}")  # Expected: (105799, 128, 4, 1)
print(f"Expected feature shape (128, 4, 1): {X_train.shape[1:] == (128, 4, 1)}")
print(f"Train label count: {len(y_train)}")
```

#### **Why These Checks Matter**

**1. Shape Verification**
- Ensures no data corruption during save/load
- Confirms feature dimensions match model input requirements
- Catches common errors (wrong reshape, missing channel dimension)

**2. Label Count Validation**
- Verifies that `len(y_train) == len(X_train)` (one label per sample)
- Detects indexing bugs or misalignment

**3. Class Name Inspection**
- Prints all 12 class names for manual verification
- Helps catch metadata parsing errors (e.g., missing classes)

#### **Using Context Managers**

```python
with np.load('file.npz') as data:
    X = data['X']
```
- `np.load` returns a context manager (in NumPy ≥1.15)
- Automatically closes file handles after use
- Prevents memory leaks from open file descriptors

#### **Efficiency Note**

Loading happens **once** at the start of training. All subsequent cells reference these loaded variables, avoiding repeated disk I/O.

---

### Cell 7: Import TensorFlow/Keras and Configure Environment

#### **What Happens**

This cell initializes the deep learning framework and configures the computational environment for reproducible training.

**Step-by-Step:**

**1. Import TensorFlow Ecosystem**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
```

**2. Set Random Seeds**
```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
```

**3. GPU Configuration**
```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

#### **Why Reproducibility Matters**

**Random Seeds Control:**
- **NumPy**: Random data shuffling, train/test splits
- **TensorFlow**: Weight initialization, dropout masks, data augmentation randomness

**Without Seeds:**
- Each run produces different results (different initial weights, different data order)
- Cannot compare experiments or reproduce published results

**With Seeds:**
- Identical initialization every time
- Enables fair comparison of hyperparameter changes
- Makes debugging easier (consistent error reproduction)

#### **GPU Memory Growth Explained**

**The Problem:**
By default, TensorFlow allocates **all** GPU memory at startup, even if not needed immediately. This can cause:
- Out-of-memory errors when multiple processes share GPU
- Inability to use GPU for other tasks

**The Solution:**
```python
tf.config.experimental.set_memory_growth(gpu, True)
```
- Allocates GPU memory **on-demand**
- Starts with minimal allocation, grows as needed
- Prevents memory conflicts

**Alternative Approaches:**
```python
# Limit to specific amount (e.g., 4GB)
tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(
    gpu[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
```

#### **Version Checking**

```python
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

**Why Check Versions?**
- Different TensorFlow versions have API differences
- GPU availability affects training speed (CPU is 10-100x slower)
- Helps diagnose "CUDA out of memory" vs "no GPU" errors

---

### Cell 8: Depthwise Separable Convolution (DWS) Block

#### **What Happens**

This cell implements the first major component of DWSTr: a Depthwise Separable Convolution block that efficiently extracts spatial features from Mel-spectrograms.

**Architecture:**
```
Input (128, 4, 1)
    ↓
[Depthwise Conv2D: 3×3, 64 filters]  ← Spatial feature learning
    ↓
[Batch Normalization]
    ↓
[ReLU Activation]
    ↓
[Pointwise Conv2D: 1×1, 64 filters]   ← Channel mixing
    ↓
[Batch Normalization]
    ↓
[ReLU Activation]
    ↓
Output (128, 4, 64)
```

**Implementation:**
```python
def create_dws_block(input_shape=(128, 4, 1), num_filters=64):
    inputs = layers.Input(shape=input_shape)
    
    # Step 1: Depthwise Convolution
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        depthwise_initializer='glorot_uniform'
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Step 2: Pointwise Convolution
    x = layers.Conv2D(
        filters=num_filters,
        kernel_size=(1, 1),
        padding='valid'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    return models.Model(inputs=inputs, outputs=x)
```

#### **Deep Dive: Depthwise Separable Convolution**

**Mathematical Comparison:**

**Standard Convolution:**
- For input `(H, W, C_in)` → output `(H, W, C_out)`
- Parameters: `K × K × C_in × C_out` (where K = kernel size)
- For 3×3, 1→64: **576 parameters**

**Depthwise Separable:**
- **Step 1 (Depthwise)**: Apply `C_in` separate 3×3 filters (one per channel)
  - Parameters: `K × K × C_in = 3 × 3 × 1 = 9`
- **Step 2 (Pointwise)**: Mix channels with 1×1 convolution
  - Parameters: `1 × 1 × C_in × C_out = 1 × 1 × 1 × 64 = 64`
- **Total: 9 + 64 = 73 parameters** (8x reduction!)

**Why This Works:**
- Separates **spatial learning** (depthwise) from **channel learning** (pointwise)
- Most image/spectrogram patterns are separable (spatial structure is independent of channel mixing)
- Provides near-identical feature extraction with dramatically fewer parameters

#### **Component Details**

**1. Depthwise Convolution**
```python
layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')
```
- Applies a 3×3 filter **independently** to each input channel
- `padding='same'` maintains spatial dimensions (128×4)
- Learns spatial patterns (edges, textures) in frequency-time domain

**2. Batch Normalization**
```python
layers.BatchNormalization()
```
- Normalizes activations across the batch dimension
- Stabilizes training, allows higher learning rates
- Formula: `(x - μ) / √(σ² + ε)` where μ, σ² are batch statistics

**3. Pointwise Convolution**
```python
layers.Conv2D(filters=64, kernel_size=(1, 1))
```
- 1×1 convolution mixes information across channels
- Expands from 1 channel → 64 feature maps
- Enables rich feature representation without spatial processing

**4. Activation Function: ReLU**
```python
layers.ReLU()
```
- `f(x) = max(0, x)` introduces non-linearity
- Sparsifies activations (negative values become 0)
- Helps gradient flow during backpropagation

#### **Why DWS in Audio Processing?**

1. **Efficiency**: Audio datasets are large; reduced parameters = faster training
2. **Effectiveness**: Mel-spectrograms have spatial structure (frequency patterns, temporal evolution) that DWS captures well
3. **Scalability**: Can stack multiple DWS blocks without parameter explosion

---

### Cell 9: Transformer Encoder Block

#### **What Happens**

This cell implements the core temporal learning component: a Transformer Encoder block with Pre-Layer Normalization architecture.

**Architecture (Pre-LN Transformer):**
```
Input Sequence (N_patches, embed_dim)
    ↓
[Layer Normalization]                    ← Pre-LN (before attention)
    ↓
[Multi-Head Self-Attention]              ← Temporal relationships
    ↓
[Residual Connection: input + attention]
    ↓
[Layer Normalization]                    ← Pre-LN (before MLP)
    ↓
[MLP: Dense → GELU → Dropout → Dense]
    ↓
[Residual Connection: previous + MLP]
    ↓
Output Sequence (N_patches, embed_dim)
```

**Implementation Highlights:**
```python
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.3):
        # Multi-Head Attention mechanism
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate
        )
        
        # Feed-forward network
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),
        ])
        
        # Normalization layers
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
```

#### **Deep Dive: Multi-Head Self-Attention**

**Attention Mechanism Formula:**
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Where:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What information do I have?"
- **V (Value)**: "What is the actual content?"
- **d_k**: Dimension of keys (for scaling)

**Self-Attention in Our Context:**
- All patches in the Mel-spectrogram sequence attend to each other
- Each patch learns which other patches are relevant for classification
- Example: An "engine noise" patch might attend strongly to "harmonics" patches

**Multi-Head Mechanism:**
```python
num_heads=4  # 4 parallel attention computations
```
- Instead of one attention computation, we run **4 in parallel**
- Each head learns different relationships:
  - Head 1: Short-term temporal patterns
  - Head 2: Long-range dependencies
  - Head 3: Frequency correlations
  - Head 4: Cross-patch interactions
- Concatenated together for richer representations

**Why Self-Attention for Audio?**
- Audio signals have **long-range dependencies** (a sound at time T=0 can relate to T=100ms later)
- Traditional CNNs struggle with long distances (receptive field grows slowly)
- Transformers capture **any** distance in a single layer

#### **Pre-Layer Normalization vs Post-Layer Normalization**

**Our Implementation (Pre-LN):**
```python
# Normalize BEFORE attention
inputs_norm = self.layernorm1(inputs)
attn_output = self.att(inputs_norm, ...)
x = inputs + attn_output  # Residual after attention
```

**Alternative (Post-LN):**
```python
# Normalize AFTER attention
attn_output = self.att(inputs, ...)
x = self.layernorm1(inputs + attn_output)
```

**Why Pre-LN?**
- **Training Stability**: Pre-LN gradients flow more smoothly
- **Faster Convergence**: Empirically shows better performance
- **Modern Standard**: Used in Vision Transformers, GPT-3, etc.

#### **Residual Connections**

```python
x = inputs + attn_output  # Skip connection
```

**Purpose:**
- Allows gradients to bypass layers (identity mapping)
- Prevents "vanishing gradients" in deep networks
- Enables training of very deep architectures (we use 6 transformer blocks)

**Mathematical Intuition:**
- Without residual: `output = f(x)` (harder to learn)
- With residual: `output = x + f(x)` (easier to learn, starts as identity)

#### **MLP (Feed-Forward Network)**

```python
self.mlp = keras.Sequential([
    layers.Dense(mlp_dim, activation='gelu'),      # Expand: 64 → 1024
    layers.Dropout(dropout_rate),
    layers.Dense(embed_dim),                        # Compress: 1024 → 64
    layers.Dropout(dropout_rate),
])
```

**Architecture:**
- **Expansion**: 64 → 1024 (16x increase) adds representational capacity
- **Compression**: 1024 → 64 returns to embedding dimension
- **GELU Activation**: `GELU(x) = x · Φ(x)` (smoother than ReLU, used in modern transformers)

**Why Dropout?**
- Randomly sets 30% of activations to 0 during training
- Prevents overfitting by forcing model to not rely on specific neurons
- Disabled during inference (`training=False`)

---

### Cell 9.5: Spec-Augmentation Layer

#### **What Happens**

This cell implements a custom Keras layer for SpecAugment—a data augmentation technique that masks frequency and time regions in Mel-spectrograms.

**SpecAugment Process:**
```
Original Mel-Spectrogram (128×4)
    ↓
[Frequency Masking]  ← Randomly mask 0-10 frequency bins
    ↓
[Time Masking]       ← Randomly mask 0-1 time frames
    ↓
Augmented Spectrogram (128×4)
```

**Implementation:**
```python
class SpecAugmentLayer(layers.Layer):
    def call(self, inputs, training=None):
        if not training:
            return inputs  # No augmentation during inference
        
        # Frequency masking: mask 0-10 bins
        f = tf.random.uniform(..., maxval=self.freq_mask_param)
        f0 = tf.random.uniform(..., maxval=n_freqs - f)
        freq_mask = create_mask(f0, f)
        
        # Time masking: mask 0-1 frames
        t = tf.random.uniform(..., maxval=self.time_mask_param + 1)
        t0 = tf.random.uniform(..., maxval=n_time - t)
        time_mask = create_mask(t0, t)
        
        return inputs * freq_mask * time_mask
```

#### **Why SpecAugment?**

**Problem: Overfitting**
- Neural networks memorize training data
- Performance drops on unseen audio (different noise, recording conditions)

**Solution: Data Augmentation**
- Artificially increases dataset size and diversity
- Forces model to learn **robust** features (not specific frequency patterns)

**Real-World Analogies:**
- **Frequency Masking**: Simulates missing frequency bands (e.g., low-pass filtering)
- **Time Masking**: Simulates audio dropouts or temporal gaps

#### **Frequency Masking Details**

```python
f = tf.random.uniform(shape=(), minval=0, maxval=10)  # Mask 0-10 bins
f0 = tf.random.uniform(..., minval=0, maxval=128-f)   # Starting position
```

**Example:**
- Randomly choose: mask 5 bins starting at bin 40
- Result: Bins 40-44 set to 0, rest unchanged
- Model must classify using remaining frequency information

**Why 0-10 bins?**
- Too aggressive (mask 50+ bins) = destroys too much information
- Too conservative (mask 0-2 bins) = minimal regularization effect
- 0-10 bins (out of 128) = ~8% masking = good balance

#### **Time Masking Details**

```python
t = tf.random.uniform(shape=(), minval=0, maxval=1+1)  # Mask 0-1 frames
t0 = tf.random.uniform(..., minval=0, maxval=4-t)       # Starting position
```

**Constraint:**
- Only 4 time frames total (75ms segment is short!)
- Masking more than 1 frame would remove 25%+ of temporal information
- 0-1 frame masking = safe regularization

#### **Training vs Inference Behavior**

```python
if not training:
    return inputs  # Critical: no augmentation on validation/test
```

**Why This Matters:**
- **Training**: Augmentation improves generalization
- **Validation/Test**: We want **real** performance metrics, not augmented data
- Using augmentation on test set would inflate accuracy (cheating)

#### **Broadcasting Magic**

```python
freq_mask = tf.reshape(freq_mask, (1, 128, 1, 1))  # [1, 128, 1, 1]
time_mask = tf.reshape(time_mask, (1, 1, 4, 1))     # [1, 1, 4, 1]
return inputs * freq_mask * time_mask  # Broadcasting applies masks
```

**Shape Explanation:**
- Input batch: `(batch_size, 128, 4, 1)`
- Frequency mask: `(1, 128, 1, 1)` → broadcasts across batch and time
- Time mask: `(1, 1, 4, 1)` → broadcasts across batch and frequency
- Element-wise multiplication applies both masks simultaneously

---

### Cell 10: Complete DWSTr Architecture

#### **What Happens**

This cell assembles all previously defined components into the complete DWSTr model architecture.

**Complete Model Flow:**

```
Input: Mel-Spectrogram (128, 4, 1)
    ↓
[SpecAugment Layer]                    ← Data augmentation (training only)
    ↓
[Depthwise Separable Conv Block]       ← Spatial features → (128, 4, 64)
    ↓
[Reshape to Patches]                   ← (128, 4, 64) → (32, 256)
    ↓
[Patch Projection]                     ← (32, 256) → (32, 64)
    ↓
[Class Token Addition]                 ← (32, 64) → (33, 64)
    ↓
[Positional Embedding]                 ← Add position info
    ↓
[Dropout]                               ← Regularization
    ↓
[Transformer Encoder × 6]              ← Temporal learning
    ↓
[Extract Class Token]                  ← (33, 64) → (64,)
    ↓
[Layer Normalization]
    ↓
[Classification Head: Dense → GELU → Dense]
    ↓
Output: Class Probabilities (12,)
```

#### **Detailed Component Breakdown**

**1. Patch Embedding**

**Why Patches?**
- Transformers operate on **sequences**, not 2D images
- We convert the 2D feature map into a sequence of "patches"

```python
num_patches_h = 128 // 4 = 32  # Vertical patches
num_patches_w = 4 // 4 = 1      # Horizontal patches
num_patches = 32 * 1 = 32        # Total patches
```

**Reshape Operation:**
```python
patches = layers.Reshape((num_patches, patch_size * patch_size * dws_filters))
# (128, 4, 64) → (32, 1024)  # Each patch is 4×4×64 = 1024 values
```

**Patch Projection:**
```python
patch_embeddings = layers.Dense(projection_dim)(patches)
# (32, 1024) → (32, 64)  # Reduce to embedding dimension
```

**2. Class Token**

```python
patch_embeddings = ClassTokenLayer(projection_dim)(patch_embeddings)
# (32, 64) → (33, 64)
```

**Purpose:**
- Special token prepended to the sequence
- Aggregates information from all patches via self-attention
- Used for final classification (similar to BERT's [CLS] token)

**Implementation:**
```python
class ClassTokenLayer(layers.Layer):
    def build(self, input_shape):
        # Learnable token (initialized randomly)
        self.class_token = self.add_weight(
            shape=(1, 1, projection_dim),
            initializer='random_normal',
            trainable=True
        )
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Broadcast to batch size
        class_tokens = tf.broadcast_to(self.class_token, [batch_size, 1, projection_dim])
        # Concatenate: [class_token, patch1, patch2, ..., patch32]
        return tf.concat([class_tokens, inputs], axis=1)
```

**Why Learnable?**
- The class token learns to aggregate the most relevant information for classification
- It's not fixed—it adapts during training

**3. Positional Embedding**

```python
encoded_patches = PositionalEmbedding(num_positions, projection_dim)(patch_embeddings)
```

**Purpose:**
- Transformers have no inherent notion of position (self-attention is permutation-invariant)
- We need to encode **where** each patch comes from in the spectrogram

**Implementation:**
```python
class PositionalEmbedding(layers.Layer):
    def __init__(self, num_positions, projection_dim):
        # Learnable position embeddings (one per position)
        self.position_embedding = layers.Embedding(
            input_dim=num_positions,  # 33 positions (1 class token + 32 patches)
            output_dim=projection_dim
        )
    
    def call(self, inputs):
        positions = tf.range(0, num_positions)  # [0, 1, 2, ..., 32]
        position_embeddings = self.position_embedding(positions)
        return inputs + position_embeddings  # Add (not concatenate)
```

**Why Addition, Not Concatenation?**
- Keeps embedding dimension constant (64)
- Easier for the model to learn positional relationships

**4. Transformer Encoder Stack**

```python
for i in range(num_transformer_blocks):  # 6 blocks
    encoded_patches = TransformerBlock(...)(encoded_patches)
```

**Progressive Feature Refinement:**
- **Block 1**: Learns local temporal relationships
- **Block 2-3**: Captures medium-range dependencies
- **Block 4-6**: Establishes global context across entire sequence

**Why 6 Blocks?**
- Balance between capacity and overfitting
- Too few (1-2): Insufficient temporal modeling
- Too many (12+): Overfitting risk, slower training

**5. Classification Head**

```python
# Extract class token (first element of sequence)
representation = layers.Lambda(lambda x: x[:, 0])(encoded_patches)  # (33, 64) → (64,)

# Final layers
representation = layers.LayerNormalization()(representation)
representation = layers.Dropout(0.3)(representation)
representation = layers.Dense(1024, activation='gelu')(representation)  # Expand
representation = layers.Dropout(0.3)(representation)
outputs = layers.Dense(12, activation='softmax')(representation)  # Class probabilities
```

**Why Extract Class Token?**
- The class token has aggregated information from all patches via attention
- It's a compact representation of the entire audio segment

**Softmax Activation:**
```python
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```
- Converts raw logits into probability distribution
- Ensures probabilities sum to 1.0
- Suitable for multi-class classification

---

### Cell 11: Build and Compile Model

#### **What Happens**

This cell instantiates the DWSTr model with specific hyperparameters and compiles it for training.

**Model Configuration:**
```python
MODEL_CONFIG = {
    'input_shape': (128, 4, 1),
    'num_classes': 12,
    'dws_filters': 64,
    'patch_size': 4,
    'projection_dim': 64,
    'num_transformer_blocks': 6,
    'num_heads': 4,
    'mlp_dim': 1024,
    'dropout_rate': 0.3
}
```

**Compilation:**
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

#### **Hyperparameter Rationale**

**`dws_filters=64`**
- Number of output channels from DWS block
- Balanced: enough capacity without excessive parameters

**`patch_size=4`**
- Divides 128×4 spectrogram into 32 patches (each 4×4)
- Smaller patches (2×2) = more patches, more computation
- Larger patches (8×8) = fewer patches, less fine-grained features

**`projection_dim=64`**
- Embedding dimension for transformer
- Standard size for this model scale (larger models use 768+)

**`num_transformer_blocks=6`**
- Depth of transformer stack
- Empirically found to work well for this task

**`num_heads=4`**
- Multi-head attention parallelism
- `projection_dim / num_heads = 64 / 4 = 16` (head dimension)
- Standard ratio (head dim typically 32-128)

**`mlp_dim=1024`**
- Feed-forward expansion (64 → 1024 → 64)
- 16x expansion is standard in transformers

**`dropout_rate=0.3`**
- 30% neurons randomly disabled during training
- Strong regularization for preventing overfitting

#### **Optimizer: Adam with Weight Decay**

**Adam Optimizer:**
- Adaptive learning rate per parameter
- Combines benefits of momentum and RMSprop
- Well-suited for transformer training

**Learning Rate: 0.001**
- Starting learning rate
- Will be reduced by `ReduceLROnPlateau` callback if needed
- Standard for Adam (can go higher/lower depending on task)

**Weight Decay: 0.0001**
- L2 regularization on weights
- Prevents weights from growing too large
- Helps generalization

**Loss Function: `sparse_categorical_crossentropy`**

**Mathematical Formulation:**
\[
L = -\log(p_{y_{true}})
\]
Where `p_y_true` is the predicted probability of the true class.

**Why "Sparse"?**
- `y_train` contains integer labels (0, 1, 2, ..., 11)
- Non-sparse version expects one-hot encoded labels
- Sparse version is more memory-efficient

**Example:**
```python
# True label: 3 (Motorboat)
# Model prediction: [0.1, 0.05, 0.02, 0.8, 0.01, ...]
# Loss = -log(0.8) ≈ 0.223
```

#### **Model Summary Output**

```python
model.summary()
```

**Output Analysis:**
- **Total Parameters: 1,339,866**
  - DWS Block: ~9,600 params
  - Transformer Blocks: ~1,300,000 params (majority)
  - Classification Head: ~30,000 params

**Parameter Distribution:**
- Most parameters are in transformer MLPs (Dense layers)
- Attention mechanisms are relatively parameter-efficient
- Model is moderately sized (good for Colab GPU training)

---

### Cell 12: Setup Training Callbacks

#### **What Happens**

This cell configures Keras callbacks that monitor and control the training process.

**Three Callbacks:**

**1. ModelCheckpoint**
```python
ModelCheckpoint(
    filepath='dwstr_best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
```

**Functionality:**
- Saves model weights after each epoch **if** validation accuracy improves
- `save_best_only=True`: Only keeps the best model (saves disk space)
- `mode='max'`: Maximize validation accuracy (vs 'min' for loss)

**Why This Matters:**
- Training can take hours—saving checkpoints prevents data loss from crashes
- Best model might occur mid-training (before overfitting starts)
- Can resume training from checkpoint if interrupted

**2. EarlyStopping**
```python
EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

**Functionality:**
- Stops training if validation loss doesn't improve for 5 consecutive epochs
- `restore_best_weights=True`: Reverts to best weights (not last epoch)

**Why Patience=5?**
- Too low (1-2): Might stop during temporary fluctuations
- Too high (10+): Wastes computation on overfitting
- 5 epochs: Good balance for this task

**Example Scenario:**
```
Epoch 10: val_loss = 0.15 (best so far)
Epoch 11: val_loss = 0.16 (no improvement, patience=1)
Epoch 12: val_loss = 0.17 (patience=2)
...
Epoch 15: val_loss = 0.18 (patience=5) → STOP TRAINING
```

**3. ReduceLROnPlateau**
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)
```

**Functionality:**
- Reduces learning rate by 50% if validation loss plateaus for 3 epochs
- Continues until learning rate drops below `min_lr`

**Why Reduce Learning Rate?**
- High learning rate (0.001) is good for initial learning
- As model converges, smaller steps (lower LR) help fine-tune
- Prevents overshooting optimal weights

**Example:**
```
Epoch 5:  LR = 0.001, val_loss = 0.20
Epoch 6:  LR = 0.001, val_loss = 0.20 (no improvement, patience=1)
Epoch 7:  LR = 0.001, val_loss = 0.20 (patience=2)
Epoch 8:  LR = 0.001, val_loss = 0.20 (patience=3) → REDUCE LR
Epoch 9:  LR = 0.0005, val_loss = 0.19 (improvement, reset patience)
```

#### **Callback Execution Order**

Callbacks are called in this order during each epoch:
1. **Before Training**: Callback setup
2. **After Each Batch**: Batch-level callbacks (if any)
3. **After Each Epoch**: 
   - Checkpoint saves (if improved)
   - Learning rate reduction (if plateau)
   - Early stopping check (if patience exceeded)

#### **Best Practices**

**Monitor Validation Metrics, Not Training:**
- Training metrics can be misleading (model memorizes)
- Validation metrics indicate real generalization

**Use Different Metrics for Different Callbacks:**
- EarlyStopping: `val_loss` (smoother, more stable)
- ModelCheckpoint: `val_accuracy` (direct performance measure)
- ReduceLROnPlateau: `val_loss` (indicates convergence)

---

### Cell 13: Train the Model

#### **What Happens**

This cell executes the core training loop, optimizing the model's weights to minimize classification error.

**Training Configuration:**
```python
BATCH_SIZE = 256
EPOCHS = 100
class_weight=class_weight_dict
```

**Training Execution:**
```python
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)
```

#### **Class Weighting Deep Dive**

**The Problem: Class Imbalance**

ShipsEar dataset has severe imbalance:
```
Passengers:      11,410 samples (37.7%)
Motorboat:         2,720 samples (9.0%)
Ocean Liner:       2,524 samples (8.3%)
...
Pilot Ship:          369 samples (1.2%)  ← Rare class!
Dredger:             703 samples (2.3%)
```

**Without Class Weights:**
- Model optimizes for majority class (Passengers)
- Achieves high accuracy by always predicting common classes
- Fails on rare classes (Pilot Ship, Dredger)

**With Class Weights:**
```python
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
```

**Formula (Balanced Weighting):**
\[
w_i = \frac{n_{samples}}{n_{classes} \times n_{samples_i}}
\]

**Example Calculation:**
- Total samples: 105,799
- Number of classes: 12
- Passengers: 11,410 samples → weight = 105799/(12×11410) ≈ **0.77**
- Pilot Ship: 369 samples → weight = 105799/(12×369) ≈ **23.9**

**Effect:**
- Misclassifying a Pilot Ship is penalized **31x more** than misclassifying a Passenger
- Forces model to learn rare class patterns

**Implementation:**
```python
class_weight_dict = {
    0: 1.23,   # Dredger
    1: 0.85,   # Fishboat
    ...
    7: 23.9,   # Pilot Ship (highest weight)
    ...
}
```

#### **Batch Size: 256**

**Why This Value?**

**Too Small (32-64):**
- More gradient updates per epoch (slower convergence)
- Higher variance in gradients (noisier training)
- Better GPU utilization needed

**Too Large (512-1024):**
- Smooth gradients, faster epochs
- May exceed GPU memory
- Less frequent weight updates can slow convergence

**256:**
- Good balance for this model size
- Fits comfortably in Colab GPU memory (typically 12-16GB)
- Standard for transformer training

**Steps Per Epoch:**
```
105,799 samples / 256 batch_size ≈ 414 steps/epoch
```

#### **Training Process**

**Each Epoch:**
1. **Shuffle** training data (random order each epoch)
2. **Batch** data into groups of 256
3. **Forward Pass**: Compute predictions
4. **Loss Calculation**: Compare predictions to true labels (weighted by class weights)
5. **Backward Pass**: Compute gradients
6. **Weight Update**: Adjust model weights via Adam optimizer
7. **Validation**: Evaluate on validation set (no weight updates)
8. **Callbacks**: Checkpoint, early stopping, LR reduction

**Training Output:**
```
Epoch 1/100
414/414 [==============================] - 95s 135ms/step
- loss: 2.3960
- accuracy: 0.0948
- val_loss: 1.9313
- val_accuracy: 0.2289
```

**Interpretation:**
- **Loss**: Cross-entropy error (lower is better)
- **Accuracy**: Fraction of correct predictions
- **Validation metrics**: Performance on unseen data

**Training Progression:**
- **Early epochs (1-5)**: Rapid improvement (learning basic patterns)
- **Mid epochs (6-14)**: Gradual refinement (paper mentions convergence around epoch 14)
- **Late epochs (15+)**: Plateaus or overfitting (early stopping intervenes)

---

### Cell 14: Evaluate on Test Set

#### **What Happens**

This cell loads the best saved model and evaluates it on the **held-out test set** for final performance reporting.

**Execution:**
```python
# Load model with custom layers
custom_objects = {
    'TransformerBlock': TransformerBlock,
    'ClassTokenLayer': ClassTokenLayer,
    'PositionalEmbedding': PositionalEmbedding,
    'SpecAugmentLayer': SpecAugmentLayer
}

best_model = keras.models.load_model(
    'dwstr_best_model.keras',
    custom_objects=custom_objects
)

# Evaluate
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
```

#### **Why Custom Objects?**

**The Problem:**
- TensorFlow saves models in HDF5 format
- Custom layers (TransformerBlock, SpecAugmentLayer) aren't in TensorFlow's registry
- Without `custom_objects`, loading fails with "Unknown layer" error

**The Solution:**
- Provide dictionary mapping layer names to class definitions
- TensorFlow uses these to reconstruct custom layers

**Why `safe_mode=False`?**
- TensorFlow 2.10+ added safe mode for security
- Our custom layers are safe, so we disable the restriction

#### **Test Set Evaluation**

**Why Test Set is Critical:**

- **Training Set**: Model has seen this—biased metric
- **Validation Set**: Used for hyperparameter tuning—still biased
- **Test Set**: **Never used** during training—unbiased performance estimate

**Test Results:**
```
Test Loss: 0.1518
Test Accuracy: 94.89%
```

**Interpretation:**
- **94.89% accuracy**: Model correctly classifies ~95 out of 100 audio segments
- **0.1518 loss**: Average cross-entropy error (low = high confidence predictions)
- **Target: 95.14%**: Achieved result is very close to paper's reported performance

#### **Model Loading Best Practices**

1. **Always Load Best Model**: Not the final epoch (might be overfitted)
2. **Verify Custom Objects**: Missing custom layer = crash
3. **Disable SpecAugment**: Inference should use real data, not augmented
4. **Batch Evaluation**: Use same batch size as training for consistency

---

### Cell 15: Plot Training History

#### **What Happens**

This cell visualizes the training progression by plotting accuracy and loss curves over epochs.

**Visualization:**
```python
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
```

**Output: Two side-by-side plots**

1. **Accuracy Curves**:
   - X-axis: Epoch number
   - Y-axis: Accuracy (0-1)
   - Two lines: Training (blue) vs Validation (orange)

2. **Loss Curves**:
   - X-axis: Epoch number
   - Y-axis: Cross-entropy loss
   - Two lines: Training vs Validation

#### **Interpreting Training Curves**

**Healthy Training:**
```
Training Accuracy:    Gradually increases, plateaus near 1.0
Validation Accuracy: Follows training closely, plateaus similarly
Gap:                 Small (< 5%) = Good generalization
```

**Overfitting (Bad):**
```
Training Accuracy:    Continues increasing → 1.0
Validation Accuracy:  Plateaus early, lower than training
Gap:                 Large (> 10%) = Model memorizing training data
```

**Underfitting (Bad):**
```
Training Accuracy:    Low, never improves much
Validation Accuracy:  Similar low performance
Gap:                 Small but both are low = Model too simple
```

**Example from Our Training:**
- Training accuracy reaches ~95%
- Validation accuracy reaches ~94%
- Gap is ~1% → **Healthy generalization!**

#### **Loss Curve Insights**

**Ideal Loss Behavior:**
- Starts high (random predictions: ~2.5 for 12 classes)
- Decreases rapidly in early epochs
- Gradually plateaus at low value (~0.15)

**Red Flags:**
- **Validation loss increasing while training loss decreases** → Overfitting
- **Both losses plateauing high** → Underfitting or learning rate too high
- **Erratic oscillations** → Batch size too small or learning rate unstable

#### **Saving Plots**

```python
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
```

**Why Save?**
- Documentation: Shows training progression for papers/reports
- Debugging: Helps identify training issues
- Comparison: Compare different model runs side-by-side

**High DPI (300):**
- Publication-quality resolution
- Sharp when printed or embedded in documents

---

### Cell 16: Detailed Per-Class Evaluation

#### **What Happens**

This cell provides a comprehensive analysis of model performance for each individual ship class.

**Components:**

**1. Classification Report**
```python
from sklearn.metrics import classification_report

print(classification_report(
    y_test, y_pred_classes,
    target_names=class_names,
    digits=4
))
```

**Metrics Explained:**

**Precision** (for each class):
\[
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
\]
- "Of all samples predicted as Dredger, how many were actually Dredgers?"
- High precision = Model is confident and correct when it predicts this class

**Recall** (for each class):
\[
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
\]
- "Of all actual Dredgers, how many did we correctly identify?"
- High recall = Model finds most instances of this class

**F1-Score**:
\[
\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]
- Harmonic mean of precision and recall
- Balanced metric (considers both false positives and false negatives)

**Support:**
- Number of test samples for each class
- Shows class distribution in test set

**Example Output:**
```
              Dredger     0.9901    1.0000    0.9950       703
             Fishboat     0.9440    0.9913    0.9671      1378
```
- **Dredger**: Perfect recall (100%), very high precision (99%) → Model excellent at finding Dredgers
- **Fishboat**: High recall (99%), good precision (94%) → Occasional false positives

**2. Confusion Matrix**

```python
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ...)
```

**What It Shows:**
- **Rows**: True labels (what the audio actually is)
- **Columns**: Predicted labels (what the model thinks it is)
- **Diagonal (bright blue)**: Correct predictions
- **Off-diagonal**: Confusion patterns

**Reading the Matrix:**
```
              Predicted →
           D  F  M  O  P  ...
Actual D   [700  0  3  0  0  ...]  ← 700 Dredgers correctly predicted
       F   [  5 1365  8  0  0  ...]  ← 5 Fishboats misclassified as Dredger
       M   [ 10  20 2570 120  0  ...]  ← 20 Motorboats confused with Fishboat
       ...
```

**Key Insights:**
- **Dredger**: Perfect predictions (700/700 on diagonal, zeros elsewhere)
- **Motorboat vs Fishboat**: Some confusion (similar acoustic signatures?)
- **Passengers (large class)**: Good accuracy but some misclassifications

**3. Per-Class Accuracy**

```python
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
```

**Calculation:**
- For each class, divide correct predictions by total samples
- Equivalent to recall (true positive rate)

**Output Example:**
```
Per-Class Accuracy:
  Dredger             : 100.00%  ← Perfect!
  Pilot Ship          :  99.19%  ← Excellent (rare class!)
  Passengers          :  90.54%  ← Lowest (but still good)
```

#### **Why Per-Class Analysis Matters**

**Overall Accuracy Can Mislead:**

Example scenario:
- Model always predicts "Passengers" (largest class)
- Overall accuracy: 37.7% (just from majority class)
- Per-class: 0% for all other classes!

**Per-Class Reveals:**
- Which classes are easy/hard to classify
- Which pairs of classes are commonly confused
- Whether rare classes are being learned (critical for imbalanced datasets)

**Actionable Insights:**
- **Low precision**: Model predicts this class too often (reduce threshold or add training data)
- **Low recall**: Model misses many instances (increase class weight or augment data)
- **Confusion patterns**: Similar classes (e.g., Motorboat ↔ Fishboat) might need more discriminative features

---

## Results and Performance

### Final Model Performance

| Metric | Value |
|-------|-------|
| **Test Accuracy** | 94.89% |
| **Test Loss** | 0.1518 |
| **Total Parameters** | 1,339,866 |
| **Training Time** | ~14 epochs (before early stopping) |
| **Best Validation Accuracy** | ~94.5% |

### Per-Class Performance Summary

| Class | Precision | Recall | F1-Score | Support | Accuracy |
|------|-----------|--------|----------|---------|----------|
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

**Key Observations:**
- **Best Performance**: Dredger (100% accuracy) and Pilot Ship (99.19% accuracy despite being rare)
- **Worst Performance**: Sailboat (83.14% precision) and Passengers (90.54% recall)
- **Class Weighting Success**: Rare classes (Pilot Ship, Dredger) achieve excellent performance, proving the effectiveness of balanced class weights

---

## Best Practices and Design Decisions

### 1. **Data Preprocessing**
- **75ms Segmentation**: Optimal balance between temporal resolution and sample count
- **Mel-Spectrogram**: Captures perceptual frequency scales better than raw FFT
- **Pre-emphasis**: Enhances high-frequency features critical for ship classification

### 2. **Model Architecture**
- **Hybrid CNN-Transformer**: Combines spatial (CNN) and temporal (Transformer) modeling
- **Depthwise Separable Convolution**: Reduces parameters by 8x while maintaining feature extraction
- **6 Transformer Blocks**: Sufficient depth without overfitting risk

### 3. **Training Strategy**
- **Class Weighting**: Critical for handling imbalanced dataset (37.7% vs 1.2% class sizes)
- **SpecAugment**: Regularization via frequency/time masking reduces overfitting
- **Learning Rate Scheduling**: Adaptive reduction prevents overshooting optimal weights

### 4. **Evaluation Methodology**
- **Stratified Splitting**: Maintains class distribution across all splits
- **Separate Test Set**: Unbiased final evaluation (never used during training)
- **Comprehensive Metrics**: Per-class analysis reveals model strengths/weaknesses

### 5. **Reproducibility**
- **Random Seeds**: Ensures identical results across runs
- **Fixed Splits**: Same train/val/test distribution every time
- **Versioned Checkpoints**: Model weights saved for future reference

---

## Troubleshooting

### Common Issues and Solutions

**1. Out of Memory (OOM) Errors**
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `BATCH_SIZE` (try 128 or 64)
- Enable GPU memory growth (already done in Cell 7)
- Process data in smaller chunks
- Use Colab Pro (higher RAM/GPU)

**2. Slow Preprocessing**
```
Preprocessing takes 2+ hours
```
**Solutions:**
- Use high-RAM Colab runtime (Runtime → Change runtime type)
- Save preprocessed data (already implemented)
- Skip reprocessing if data already saved

**3. Model Not Improving**
```
Validation accuracy stuck at ~30%
```
**Check:**
- Verify class weights are applied (`class_weight_dict` printed in Cell 13)
- Ensure SpecAugment is only active during training
- Check that data shapes are correct `(N, 128, 4, 1)`
- Verify train/val split is stratified

**4. Custom Layer Loading Error**
```
Unknown layer: 'TransformerBlock'
```
**Solution:**
- Ensure all custom classes are defined before loading
- Include all custom layers in `custom_objects` dictionary (Cell 14)

**5. Low Accuracy on Rare Classes**
```
Pilot Ship accuracy: 50%
```
**Solutions:**
- Increase class weight for rare classes (modify `class_weight.compute_class_weight`)
- Add more data augmentation for rare classes
- Consider oversampling rare classes during training

**6. Drive Mount Timeout**
```
FileNotFoundError: /content/drive/...
```
**Solution:**
- Re-run Cell 1 to remount Google Drive
- Check that paths in `DATASET_ROOT_PATH` are correct
- Verify files exist in your Drive

---

## Future Improvements

### Potential Enhancements

1. **Advanced Augmentation**
   - Mixup/CutMix for audio
   - Time-stretching and pitch-shifting
   - Background noise injection

2. **Architecture Modifications**
   - Increase transformer depth (8-12 blocks)
   - Experiment with different attention mechanisms (Linformer, Performer)
   - Add squeeze-and-excitation blocks to DWS

3. **Training Optimizations**
   - Learning rate warmup schedule
   - Gradient clipping for stability
   - Mixed precision training (FP16)

4. **Evaluation Enhancements**
   - ROC curves per class
   - T-SNE visualization of embeddings
   - Attention weight visualization

---

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{shipsEar_dwstr,
  title={DWSTr: Depthwise Separable Convolution and Transformer for Ship Audio Classification},
  author={Your Name},
  year={2024},
  note={Implementation of ShipsEar audio classification using hybrid CNN-Transformer architecture}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- ShipsEar dataset creators for providing the underwater acoustic dataset
- Original DWSTr paper authors for the architecture inspiration
- TensorFlow/Keras team for the excellent deep learning framework
- Google Colab for providing free GPU resources

---

**Last Updated**: 2024  
**Maintainer**: Advaith R Pai
**Status**: Production Ready ✅