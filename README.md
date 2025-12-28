# 🚦 Autonomous Traffic Sign Recognition System (v3.0 - Stable)

### **Achieving 94.26% Accuracy via Stabilized MobileNetV2 & YOLO-Style Optimization**

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-success) ![Accuracy](https://img.shields.io/badge/Accuracy-94.26%25-brightgreen) ![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange)

## 📋 Project Overview
This project implements a robust **Traffic Sign Recognition (TSR)** system designed for autonomous vehicle subsystems.
**Update (v3.0):** The latest release addresses the stability and overfitting issues observed in v2.0. By implementing a **Stabilized MobileNetV2** architecture with a "Frozen Spine" strategy and integrating **YOLO-style training optimizations**, the model achieves a consistent **94.26% validation accuracy** on the German Traffic Sign Recognition Benchmark (GTSRB), making it highly reliable for real-time deployment.

## 🏆 Key Achievements
* **Accuracy:** Improved from **91.56%** (v2) to **94.26%** (v3).
* **Architecture:** MobileNetV2 (Frozen Spine Strategy) optimized with YOLO training techniques.
* **Engineering Fixes:** * **Solved Overfitting:** In v2, the model "memorized" training data. v3 fixes this by freezing feature extraction layers.
    * **Stabilized Loss:** Validation loss is now consistent with training loss, eliminating "hallucinations" on unseen data.
    * **Edge Case Robustness:** Verified against tricky inputs like "No Vehicles" vs "Speed Limits".

## 🛠️ Tech Stack
* **Core:** Python, TensorFlow/Keras, OpenCV
* **Data Source:** [GTSRB Kaggle Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
* **Techniques:** * Transfer Learning (ImageNet weights)
    * **YOLO-Style Optimization Pipeline** (Cosine Decay, Mosaic-style Augmentation)
    * Frozen Layers Strategy
    * Aggressive Dropout (0.5)

## 📊 Evolution of Performance

| Version | Architecture | Strategy | Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0** | Custom CNN | Raw Training | 84.00% | ❌ **Deprecated** (Struggled with shadows/blur) |
| **v2.0** | MobileNetV2 | Full Unfreeze | 91.56% | ⚠️ **Unstable** (High variance/Overfitting risk) |
| **v3.0** | **MobileNetV2 + YOLO Opt** | **Frozen Spine** | **94.26%** | ✅ **Stable & Production Ready** |

## 🧠 Key Engineering Decisions (v3.0)

To break the 91% ceiling and fix stability issues, three major architectural changes were implemented:

### 1. The "Frozen Spine" Strategy
In v2.0, unfreezing the entire network caused "Catastrophic Forgetting." In v3.0, we **froze the first 100 layers** of MobileNetV2.
* **Bottom Layers (Frozen):** Retain the pre-trained "ImageNet" knowledge (edge detection, basic shapes).
* **Top Layers (Trainable):** Only the high-level interpretation layers are retrained to recognize specific traffic symbols.

### 2. YOLO-Style Training Optimization
Inspired by YOLO object detection training pipelines, we implemented:
* **Cosine Decay Learning Rate:** Smoothly reduces the learning rate to find the global minima.
* **Label Smoothing:** Prevents the model from becoming "overconfident" on noisy data.

### 3. Aggressive Dropout Injection
We increased the Dropout rate to **0.5**. This randomly disables 50% of the neurons during training, forcing the network to learn redundant features.

## 💻 Code Snippet: The Stabilized Architecture

```python
# The "Frozen Spine" - Preventing Overfitting
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

# Freeze the bottom 100 layers to preserve basic vision features
for layer in base_model.layers[:100]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)  # Stabilize weights
x = Dropout(0.5)(x)          # Aggressive Dropout to prevent memorization
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)          # Second Dropout layer for redundancy
predictions = Dense(43, activation='softmax')(x)
