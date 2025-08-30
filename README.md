# Network Intrusion Detection Using Machine Learning

This project implements a **Network Intrusion Detection System (NIDS)** using machine learning techniques to identify malicious network traffic.  
We train and evaluate multiple models on the **NSL-KDD dataset**, an improved version of the classic KDD’99 dataset, to achieve high accuracy and low false positive rates.  
A **Streamlit-based GUI** is also provided for real-time predictions and visualization.

---

## 🚀 Features
- Preprocessing of network traffic data (cleaning, encoding, normalization).
- Feature selection using correlation analysis and tree-based importance.
- Multiple ML models implemented:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
- Hyperparameter tuning with Grid/Random Search.
- Model evaluation with Accuracy, Precision, Recall, F1-score, and False Positive Rate.
- User-friendly **Streamlit GUI** for intrusion detection.

---

## 📊 Dataset

- **NSL-KDD dataset**  
  - 41 features + 1 label  
  - Attack categories: DoS, Probe, R2L, U2R  
  - Classes: Normal vs Attack  

Small **sample dataset files** are included in `Dataset/` for quick testing:
- `KDDTraining_sample.csv`  
- `KDDTesting_sample.csv`  

🔗 Full dataset can be downloaded from: [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)  
Place the files in the `Dataset/` folder before running training.
