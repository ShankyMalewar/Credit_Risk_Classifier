from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
MODEL_DIR = os.path.join(BASE_DIR, 'models')

rf_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))

# Common feature order
feature_order = [
    "num__AMT_CREDIT_SUM_sum", "num__AMT_CREDIT_SUM_mean",
    "num__INST_AMT_PAYMENT_MAX", "num__INST_AMT_PAYMENT_MEAN",
    "num__AMT_CREDIT_SUM_DEBT_max", "num__POS_MONTHS_BALANCE_MEAN",
    "num__INST_PAYMENT_RATIO_MIN", "num__SK_ID_BUREAU_sum",
    "num__INST_AMT_INSTALMENT_MAX", "num__INST_AMT_INSTALMENT_SUM",
    "num__AMT_CREDIT_SUM_DEBT_mean", "num__POS_MONTHS_BALANCE_MIN",
    "num__AMT_CREDIT_SUM_DEBT_sum", "cat__NAME_EDUCATION_TYPE_Higher education",
    "num__INST_AMT_PAYMENT_SUM", "num__POS_MONTHS_BALANCE_SUM",
    "num__DAYS_CREDIT_sum", "num__DAYS_CREDIT_min",
    "num__DAYS_CREDIT_max", "num__DAYS_CREDIT_mean"
]

@app.route('/')
def index():
    return render_template('index.html')

def process_input(data):
    X = [[float(data[feat]) for feat in feature_order]]
    return X

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    try:
        data = request.get_json()
        X = process_input(data)
        prob = rf_model.predict_proba(X)[0, 1]
        pred = int(prob > 0.5)
        return jsonify({"prediction": f"RF: {pred} (Risk probability: {prob:.2f})"})
    except Exception as e:
        return jsonify({"prediction": f"Error: {str(e)}"}), 500

@app.route('/predict_xgb', methods=['POST'])
def predict_xgb():
    try:
        data = request.get_json()
        X = process_input(data)
        prob = xgb_model.predict_proba(X)[0, 1]
        pred = int(prob > 0.5)
        return jsonify({"prediction": f"XGB: {pred} (Risk probability: {prob:.2f})"})
    except Exception as e:
        return jsonify({"prediction": f"Error: {str(e)}"}), 500

@app.route('/predict_ensemble', methods=['POST'])
def predict_ensemble():
    try:
        data = request.get_json()
        X = process_input(data)
        prob_rf = rf_model.predict_proba(X)[0, 1]
        prob_xgb = xgb_model.predict_proba(X)[0, 1]
        ensemble_prob = (0.5 * prob_rf) + (0.5 * prob_xgb)
        pred = int(ensemble_prob > 0.5)
        return jsonify({"prediction": f"Ensemble: {pred} (Risk probability: {ensemble_prob:.2f})"})
    except Exception as e:
        return jsonify({"prediction": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
