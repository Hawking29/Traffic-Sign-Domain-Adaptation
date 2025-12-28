# 🚦 Traffic Sign Recognition System (v2.0)

### **From 84% to 91%: Solving the "Domain Gap" in AI**

## 📋 Project Overview
This project builds a robust Traffic Sign Recognition system capable of identifying signs in challenging environments. 
**Update (v2.0):** The model has been upgraded from a simple custom CNN to a **MobileNetV2** architecture using Transfer Learning, trained on the official **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.

## 🏆 Key Achievements
* **Accuracy:** Improved from **84%** (v1) to **91.56%** (v2).
* **Architecture:** MobileNetV2 (Pre-trained on ImageNet, Fine-Tuned for 43 classes).
* **Engineering Fixes:** * Solved "Red Circle Bias" where the model confused *No Vehicles* with *Speed Limits*.
    * Implemented "Smart Cropping" to fix aspect-ratio distortion on digital inputs.
    * Used Kaggle API for high-speed cloud data pipeline.

## 🛠️ Tech Stack
* **Core:** Python, TensorFlow/Keras, OpenCV
* **Data Source:** [GTSRB Kaggle Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
* **Techniques:** Transfer Learning, Data Augmentation, Active Learning (Patching).

## 📊 Results & Analysis

| Version | Model | Training Data | Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0** | Custom CNN | Custom Pickle | 84.0% | Failed on "No Vehicles" sign due to low resolution. |
| **v2.0** | **MobileNetV2** | **Full GTSRB** | **91.6%** | Robust detection. Successfully identifies edge cases. |

### **The "No Vehicles" Bug Fix**
In v1, the model consistently misidentified the "No Vehicles" sign as a "Speed Limit" (Class 1) because it prioritized the red circle shape over the inner symbol. 
By upgrading to high-res (75x75) inputs and unfreezing the MobileNet layers, the v2 model correctly identifies the car symbol (Class 15).

![Latest Results](Results_v2.png)

## 🧠 Lessons Learned
* **Data Quality is King:** Switching from compressed pickle files to the raw Kaggle dataset gave a 7% accuracy boost immediately.
* **Aspect Ratio Matters:** Real-world cameras see squashed images. Pre-processing must include "Smart Cropping" to preserve geometry.

## 👤 Author
**[Abhirup Chattopadhyay]**
*Electrical Engineering Student, Jalpaiguri Government Engineering College*
