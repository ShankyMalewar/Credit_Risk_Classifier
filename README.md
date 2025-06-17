# 💳 Credit Safety Checker

**Credit Safety Checker** is a machine learning-powered web application that helps users assess their credit safety level based on detailed financial inputs.  
This project showcases an end-to-end ML pipeline with data processing, feature engineering, model training, MLOps practices, deployment, and frontend integration.

---

## 🚀 Project Summary

- Aggregated data from **7 CSV files** (main application, bureau, credit card balance, installments, POS cash balance, previous application, bureau balance).
- Engineered **281 features** through sums, means, mins, maxes, ratios.
- Carefully removed features causing target leakage during exploratory analysis.
- Reduced to **20 most important features** via SHAP analysis for final modeling.
- Trained Random Forest, XGBoost, CatBoost, ANN, and ensemble models.
- Applied MLOps practices with MLflow for experiment tracking, hyperparameter logging, model registry.
- Designed and deployed a Flask web app with clean UI for predictions.
- Set up CI/CD with GitHub Actions for automated testing and deployment.
- Deployed on Render using GitHub integration.



---

## 🧠 Key Topics and Concepts

### Data Engineering + Feature Selection
- Aggregated and merged data from 7 different CSV files.
- Created 281 features using domain-driven aggregation logic.
- Identified and removed features causing target leakage through validation.
- Applied SHAP for feature importance → selected best 20 features.

### Model Training
- Trained Random Forest, XGBoost, CatBoost, ANN, ensemble models.
- Handled imbalance via stratified split, class weights.
- Evaluated using F1, recall, accuracy.

### MLOps (via Made With ML concepts)
- **Experiment Tracking:** MLflow for param, metric, artifact logging.
- **Model Registry:** Used MLflow model versioning.
- **Reproducibility:** Saved preprocessor, seeds, environment configs.
- **Testing:** Unit tests for API, data validation, model scoring.
- **CI/CD:** GitHub Actions workflow for testing + deployment.
- **Deployment:** Flask API and UI deployed to Render.

### SHAP Analysis
- SHAP values used to explain model predictions.
- Identified the most impactful features on credit risk.
- Improved model transparency and trustworthiness.

---

## 🎨 Frontend
- HTML + CSS + JS clean modern form.
- 20 inputs, buttons for RF, XGB, Ensemble predictions.
- Fully responsive design.

---

## 🛠️ Tech Stack
Python, Scikit-learn, XGBoost, CatBoost, PyTorch, MLflow, SHAP, Flask, GitHub Actions, Render

---

## ✨ Acknowledgements
- Inspired by the Made With ML MLOps course.
- SHAP insights from work by Scott Lundberg and contributors.


