# **DA Project – Kaggle Contest (2025)**

**Name:** Shashank Satish Adsule  
**Roll:** DA25M005  
**Date:** 18 November 2025  

This repository contains the complete pipeline—data loading, preprocessing, embedding generation, augmentation, feature engineering, model training, and evaluation—developed for solving the Kaggle Data Analysis Contest (2025). The project brings together modern NLP embedding techniques, classical ML models, and custom augmentation strategies to predict quality scores for text–metric pairs.

## 🚀 **Project Structure**
```bash
├── dataset/
│ ├── train_data.json
│ ├── test_data.json
│ ├── metric_name_embeddings.npy
│ └── defination_embedding.pkl
│
├── code/
│ ├── create_syntetic_data.py
│ └── metric_embeding_mapping.py
│
├── genrate_emb/
│ ├── gen_embedding_v1.ipynb
│ └── gen_embedding_v2.ipynb
│
├── final_code/
│ ├── version 1/
│ └── version 2/
│ ├── train_combine.ipynb
│ ├── test_combine.ipynb
│ ├── train_split.ipynb
│ └── test_split.ipynb
│
├── assests/models/ (Saved .pkl models)
├── outputs/ (Generated CSVs & logs)
└── Report.pdf (Detailed Project Report)
```

---

## **1. Introduction**
This project explores a complete workflow for predicting discrete quality scores (0–10) based on a combination of metric definitions and text responses. Multiple sentence-transformer architectures, structured data augmentation, and feature engineering techniques were used to build a robust training dataset. The final models are trained and evaluated using a suite of machine learning algorithms—including linear models, ensemble methods, and neural networks.

---

## **2. Dataset Overview**
The dataset contains the following key fields:

- **metric_name**
- **user_prompt**
- **system_prompt**
- **response**
- **score** (label)

The score distribution is **highly skewed**, with many samples clustered around 9–10. Metric names are also mapped to embedding vectors via a predefined embedding dictionary.

---

## **3. Data Loading & Preprocessing**
Preprocessing includes:

- converting text to lowercase  
- removing punctuation, digits, and extra whitespace  
- filling missing text fields with empty strings  
- mapping metric_name → metric_embeddings using a prebuilt dictionary  
- converting score to `float32` for efficiency  

This results in a clean, standardized dataset ready for embedding and model training.

---

## **4. Embedding Generation**
Embeddings were generated using three HuggingFace models:

- `google/embedding-gemma-300m`
- `all-mpnet-base-v2`
- `intfloat/e5-base-v2`  ← **best performing**

Two pipelines were implemented:

- **v1** → concatenate all text components → single embedding  
- **v2** → generate separate embeddings for each text component  

Embeddings were saved as `.npy` or `.parquet` for reuse.

---

## **5. Data Augmentation**
To enrich the dataset and reduce score imbalance, multiple synthetic samples were generated:

### Negative Samples
- **Shuffle-based negatives** (misaligned text)
- **Noise-corrupted negatives** (Gaussian perturbation)
- **Metric-swap negatives** (wrong metric–text pairing)

### Mid-range Samples (3–6)
Generated using noise-corrupted embeddings to fill sparse score regions.

### High-score Samples (9–10)
Generated using positive mismatched pairs.

All augmented samples were concatenated with original data, resulting in a larger and more balanced dataset.

---

## **6. Feature Engineering**
For each metric–text embedding pair:

- **Cosine similarity**
- **Absolute difference**
- **Element-wise product**
- **Concatenated raw embeddings**

Final feature vector dimension: **3073**

These features capture linear and non-linear relationships, improving model robustness.

---

## **7. Model Training**
Multiple algorithms were tested:

- **LinearRegression**
- **Ridge Regression**
- **XGBRegressor**
- **XGBRFRegressor**
- **RandomForestRegressor**
- **MLPRegressor**

Each model was trained using the engineered features. The training pipeline fits the model to `x_train`, generates predictions for `x_test`, and calculates:

- **MSE**
- **R² score**
- **RMSE (train/test)**

Predictions are rounded and clipped to remain within the valid score range (0–10).  
All final models are saved as `.pkl` in `assests/models/`.

---

## **8. Evaluation Results**

| Model | MSE (train) | R² (train) | RMSE (train) | RMSE (test) |
|-------|-------------|------------|--------------|-------------|
| LinearRegression | 11.99 | 0.07 | 3.47 | 4.35 |
| Ridge Regression | 11.96 | 0.08 | 3.47 | 4.33 |
| XGBRegressor | **0.74** | 0.10 | **0.90** | 3.34 |
| XGBRFRegressor | 0.77 | 0.06 | 0.91 | 4.05 |
| RandomForestRegressor | 0.78 | 0.04 | 0.92 | 4.62 |
| MLPRegressor | 7.40 | **0.43** | 2.71 | **2.95** |

---

## **Key Findings**
- **XGBRegressor** achieved the lowest training error.  
- **MLPRegressor** showed the strongest generalization on the test set.  
- Data augmentation meaningfully improved mid-range score predictions.  

---

## **How to Run**
1. Install dependencies  
2. Generate embeddings using the notebooks in `genrate_emb/`  
3. Run augmentation scripts in `code/`  
4. Train models using notebooks in `final_code/`  
5. Evaluate and generate predictions via saved models  

---

## **License**
This repository is intended for academic use as part of the 2025 Data Analysis Project.
