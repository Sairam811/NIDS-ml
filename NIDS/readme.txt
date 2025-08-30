Intrusion Detection System using Machine Learning (NSL-KDD)

This project implements a machine learning-based Intrusion Detection System (IDS)
 using the NSL-KDD dataset. It includes training models in Google Colab and a Streamlit-based GUI 
 to predict and visualize network attack types.



Project Structure

project-folder/
│
├── dataset/ # Contains NSL-KDD dataset files
├── all the models/ # Contains saved .pkl model and scaler files
├── Web_GUI.py # Streamlit GUI application
├── DM_Project_code.ipynb # Colab notebook for model training
└── README.txt # Project documentation

 How to Run the Project

### 1. Train the Models (Google Colab)
- we have to Open `train_models.ipynb` in Google Colab.
- Upload `KDDTraining_dataset.txt` and `KDDTesting_dataset.txt` which was there in the `dataset/` folder.
- Run the cells to train models: Logistic Regression, XGBoost, Random Forest, and SVM.
- Save the models using `joblib.dump()`:
  - `logistic_regression_model.pkl`
  - `xgboost_model.pkl`
  - `random_forest_model.pkl`
  - `svm_model.pkl`
  - `minmax_scaler.pkl`
- Download the `.pkl` files and place them in the `models/` folder.


 2. Run the Streamlit GUI (VS Code or Terminal)

#### Setup:

cd project-folder/
pip install -r requirements.txt
Launch GUI:
streamlit run app.py


GUI Features
Input network traffic features from the sidebar

Select a model (Logistic Regression, XGBoost, Random Forest, SVM)

View predicted attack type and visual highlight

Includes example values for known neptune attack