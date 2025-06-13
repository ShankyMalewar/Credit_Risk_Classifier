# src/config.py

#Paths
DATA_PATH = "data/"
MODEL_PATH = "models/"
RAW_DATA_FILES = [
    "application_train.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "previous_application.csv"
]

#Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2

#Class Balance
DOWNSAMPLE_RATIO = 2  # majority = 2x minority

#Numerical Pipeline
NUM_IMPUTE_STRATEGY = "median"
SCALE_NUMERICAL = True

#Categorical Pipeline
CAT_IMPUTE_STRATEGY = "most_frequent"
DROP_FIRST_CATEGORY = True
HANDLE_UNKNOWN = "ignore"

#ANN Hyperparameters
ANN_INPUT_DIM = None  # To be set dynamically
ANN_HIDDEN1 = 256
ANN_HIDDEN2 = 128
ANN_DROPOUT = 0.3
ANN_EPOCHS = 15
ANN_LR = 0.0001
BATCH_SIZE = 64

# Cross-validation
CV_SPLITS = 5


RF_MODEL_FILE = MODEL_PATH + "rf_model.pkl"
XGB_MODEL_FILE = MODEL_PATH + "xgb_model.pkl"
ANN_MODEL_FILE = MODEL_PATH + "ann_model.pth"
PREPROCESSOR_FILE = MODEL_PATH + "preprocessor.pkl"


REGISTER_RF_NAME = "CreditRFClassifier"
REGISTER_XGB_NAME = "CreditXGBClassifier"
