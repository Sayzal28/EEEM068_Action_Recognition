# Action Recognition Using Vision Transformers

This project explores the use of the **Timesformer** model, a Vision Transformer (ViT) architecture, for human action recognition. The model is fine-tuned on a subset of the HMDB51 dataset to classify 25 different human actions.

The study includes a comprehensive analysis of:
* Hyperparameter tuning (Optimizers, Learning Rate, Batch Size)
* The impact of different pre-trained datasets (Kinetics-400, Kinetics-600, SSV2)
* Various frame sampling techniques (Uniform, Equidistant, Random)
* A performance comparison with other ViT models like ViViT

The best-performing model achieved a **Top-1 accuracy of 93.6%** and a **Top-5 accuracy of 99.2%**.

This repository also includes a **Streamlit web interface** for real-time video classification.



## 📋 Project Overview

### Model: Timesformer
The core of this project is the **Timesformer**, a transformer-based architecture designed for video. Unlike standard ViTs that process images, Timesformer efficiently captures spatio-temporal information by using **divided space-time attention**. This approach separates attention into two parts:
1.  **Spatial Attention:** Applied within each frame to model relationships between patches.
2.  **Temporal Attention:** Applied across frames to model motion and temporal dependencies.

### Dataset: HMDB_simp
The model was fine-tuned on `HMDB_simp`, a custom subset of the **HMDB51** dataset.
* **Classes:** 25
* **Videos:** 50 per class, totaling 1250 videos
* **Data Split:** 900 training, 225 validation, and 125 test clips

The 25 supported action classes include:
> `brush_hair`, `cartwheel`, `chew`, `climb_stairs`, `draw_sword`, `fencing`, `flic_flac`, `golf`, `handstand`, `pick`, `pour`, `pushup`, `ride_bike`, `shoot_bow`, `shoot_gun`, `situp`, `smile`, `smoke`, `throw`, `wave`, `clap`, `climb`, `eat`, `kiss`, `punch`

## 🛠️ Methodology & Training

### Preprocessing and Augmentation
The Timesformer model expects 8 sampled frames at a 224x224 resolution.
* For videos with an insufficient number of frames, a sequential augmentation pipeline was applied **before** sampling.
* **Augmentation Techniques:**
    * Frame interpolation (pixel averaging)
    * Temporal reversal (appending reversed frames)
    * Brightness variation (factors of 0.6, 0.8, 1.2, 1.4)

### Training Configuration
The model was trained using the Hugging Face `Trainer`, with logging managed by Tensorboard. The optimal configuration was found to be:
* **Pre-trained Model:** `facebook/timesformer-base-finetuned-k400`
* **Optimizer:** SGD
* **Learning Rate:** 0.0035
* **Batch Size:** 8
* **Epochs:** 8
* **Momentum:** 0.9
* **Weight Decay:** 0.003

## 📊 Key Results & Findings

### 1. Final Performance
The best configuration achieved outstanding results on the test set:
* **Top-1 Accuracy:** 93.6%
* **Top-5 Accuracy:** 99.2%

### 2. Hyperparameter Tuning
* The **SGD optimizer** (93.6% Top-1) significantly outperformed ADAM (17.6% Top-1 with the same LR) for this fine-tuning task. This suggests SGD's tendency to find flatter minima is beneficial for generalizing temporal patterns.

### 3. Impact of Pre-training Dataset
* The choice of pre-training dataset is **crucial**.
* Models trained on **Kinetics-400** (93.6%) and **Kinetics-600** (93.6%) performed best, as their domain (general human actions) matches HMDB_simp.
* The **SSV2**-trained model performed worst (84.8%) due to a domain mismatch. SSV2 focuses on human-object interactions, whereas HMDB_simp focuses on human actions regardless of objects.

### 4. Impact of Sampling Technique
* **Uniform sampling** (1/32 rate) performed the best (93.6% Top-1). This method preserves the video's natural temporal structure.
* **Equidistant sampling** was a close second (92.8%).
* **Random sampling** performed the worst (91.2% Top-1), likely because the fragmented motion information makes it difficult for the model to learn action progression.

### 5. Impact of Augmentation Order
* Applying augmentations **before sampling** (93.6% Top-1) was critical. This creates a more diverse pool of frames for the model to sample from.
* Augmenting **after sampling** (85.83% Top-1) severely limited the augmentation's effectiveness and could create unrealistic motion from interpolating a small set of frames.

### 6. Model Comparison
* **Timesformer** (93.6% Top-1) outperformed **ViViT** (88.8% Top-1) when both were fine-tuned on the same dataset.

## 🚀 Web Interface & Demo

A web interface was developed using **Streamlit** to demonstrate the model's practical application.

**Features:**
* Allows a user to upload any video file.
* Displays the sampled frames used for prediction.
* Predicts the action and displays the **class prediction** and **confidence score**.
* Note: The model will perform best on videos that belong to one of the 25 trained classes.

## 💡 Future Work

Based on the study's conclusion, future work could include:
* Fine-tuning the model on a larger, more diverse dataset to improve generalization.
* Improving computational efficiency for deployment using techniques like model distillation or quantization.

## 📚 References
* [1] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." 2020.
* [2] Bertasius, Gedas, Heng Wang, and Lorenzo Torresani. "Is space-time attention all you need for video understanding?." 22021.
* [3] Arnab, Anurag, et al. "Vivit: A video vision transformer." 2021.
* [4] Hugging Face: facebook/timesformer-base-finetuned-k400
