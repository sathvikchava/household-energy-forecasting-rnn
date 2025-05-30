# Forecasting Household Energy Consumption with RNN-based Time Series Models

**Course:** ADTA 5560 – Recurrent Neural Networks for Sequence Data  
**Team Name:** Dynamic Neurons  
**Date:** May 2025  
**Team Members:** Sathvik Chava, Sumanjali Banjara, David Adeyemi, Andrew Gunason Paul  
**Tech Stack:** Python, TensorFlow/Keras, Jupyter Notebook  

---

##  Project Overview

This project focuses on building deep learning models to forecast short-term household energy consumption using time series data. We used a real-world dataset containing 10-minute interval readings from a single residential home, applying various Recurrent Neural Network (RNN) architectures to model and predict energy usage patterns.

The goal was to explore and compare different neural sequence models to support smart energy management, reduce electricity costs, and promote sustainability.

---

##  Objectives

- Predict appliance energy usage using environmental and temporal features.
- Evaluate and compare five different RNN architectures.
- Generate actionable insights for energy optimization in residential settings.

---

##  Dataset Description

- **Source:** UCI Machine Learning Repository  
- **Observations:** ~19,735  
- **Granularity:** 10-minute intervals  
- **Target Variable:** `Appliances` – energy consumed (Wh)  
- **Features:** Indoor temperatures and humidities, outdoor weather, time-based features (hour, day of week, sinusoidal encodings)

---

##  Models Implemented

| Model             | Description                             | Avg MSE   | Avg MAE   |
|------------------|-----------------------------------------|-----------|-----------|
| Simple RNN        | Vanilla recurrent architecture          | 0.007563  | 0.042867  |
| Bi-directional RNN| Context from past and future            | 0.007748  | 0.061030  |
| LSTM              | Memory-augmented RNN                    | 0.007586  | 0.058501  |
| GRU               | Efficient memory structure              | 0.007902  | 0.060999  |
| Encoder-Decoder   | Context vector with LSTM-based decoding | **0.007469** | **0.056027** |

**Best Model:** Encoder-Decoder LSTM – lowest error, stable convergence

---

##  Data Preprocessing Steps

- Parsed and indexed timestamps
- Engineered features: `hour`, `dayofweek`, `month`, `is_weekend`, sinusoidal encodings
- Excluded irrelevant/noisy features (`lights`, `rv1`, `rv2`)
- Scaled inputs using Min-Max scaling
- Used sliding windows for RNN sequence generation
- Sequential 70/15/15 train-validation-test split

---

##  Evaluation Metrics

Models were evaluated using:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Visual comparison** of predicted vs. actual values
- **Training behavior** via loss curves

---

##  Key Takeaways

- Encoder-Decoder architecture achieved the best overall performance.
- All models struggled with predicting sharp energy spikes (likely due to user behavior not captured in data).
- Feature engineering significantly improved learning of daily and weekly cycles.
- GRU offered the best balance of simplicity and performance for faster training in production scenarios.

---

##  Skills Used / Gained

###  Technical Skills
- Time Series Forecasting
- RNN Architectures (Simple RNN, LSTM, GRU, Encoder-Decoder)
- TensorFlow / Keras model building
- Feature Engineering for temporal data
- Data Scaling, Sequence Generation
- Model tuning and early stopping
- Evaluation using MSE, MAE, and visual plots

###  Analytical & Applied ML Skills
- Comparative model analysis
- Trend interpretation and cyclic behavior analysis
- Sequence data modeling principles
- Optimization of energy forecasting use cases

###  Tools & Practices
- Jupyter Notebook for experimentation
- Python (NumPy, pandas, matplotlib, seaborn)
- GitHub for version control and project sharing

---

##  Files in This Repo

| File Name                     | Description                                      |
|------------------------------|--------------------------------------------------|
| `Dynamic_Neurons_Project_Report.pdf` | Full written report with detailed methodology |
| `DynamicNeurons-PPT.pdf`     | Project presentation slide deck                 |
| `Dynamic_Neurons_PythonCode.ipynb` | Final notebook used to train and evaluate models |
| `energydata_complete.csv`    | Dataset from UCI repository                     |
| `README.md`                  | Project overview and documentation              |

---

##  Notes

This was a collaborative academic project. This repository contains my version and acknowledges the full team. The dataset is publicly available and used for educational purposes.

