# 🧪 Trajectory Analysis for Particle Position Detection  

## 📖 Overview  
This project aims to develop a predictive model for estimating the (x, y) position of a particle passing through a **Resistive Silicon Detector (RSD) sensor** with 12 pads. Using **signal processing and multiple regression techniques**, we analyze sensor data to accurately determine particle trajectories.

## 🛠️ Methodology  

### 🔹 **1. Data Preprocessing**  
- The dataset includes **514,000 records** (training) and **385,500 records** (evaluation).  
- Each record consists of **signal features from 18 sensor pads**.  
- **Noise reduction & feature selection** performed using:
  - Outlier detection (distribution plots, correlation analysis).
  - Feature importance analysis.

### 🔹 **2. Model Selection & Training**  
- Multiple regression models were evaluated, including:
  - **Extra Trees Regression** 🌲
  - **Random Forest Regression** 🌳
  - Decision Trees, Bagging, and Linear Regression for comparison.
- The best models were chosen based on **Adjusted R², MAE, and MSE** scores.

### 🔹 **3. Hyperparameter Tuning**  
- **Key parameters optimized**:  
  - `max_depth`, `n_estimators`, `max_features`, and `criterion` (using Grid Search).  
  - Models trained using **hold-out validation** (80% train, 20% validation).  

## 📊 Results  
- **Best-performing model**: **Extra Trees Regression**  
  - **Average Euclidean Distance**: **4.589**  
  - **Optimal Hyperparameters**:  
    - `max_depth = 48`
    - `n_estimators = 750`
    - `criterion = 'friedman_mse'`  
- **Random Forest achieved** a similar performance, but Extra Trees showed better generalization.  
- **Key Features Identified**: `pmax`, `negpmax` (strongest predictors).  

## 🚀 How to Run  
### **1. Install Dependencies**
Ensure you have Python and the required libraries installed:
```bash
pip install pandas numpy matplotlib scikit-learn
