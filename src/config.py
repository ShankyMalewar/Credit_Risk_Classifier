# src/config.py

# ----------- PATHS -----------
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

# ----------- REPRODUCIBILITY -----------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ----------- CLASS BALANCING -----------
DOWNSAMPLE_RATIO = 2  # majority = 2x minority

# ----------- NUMERIC PIPELINE -----------
NUM_IMPUTE_STRATEGY = "median"
SCALE_NUMERICAL = True

# ----------- CATEGORICAL PIPELINE -----------
CAT_IMPUTE_STRATEGY = "most_frequent"
DROP_FIRST_CATEGORY = True
HANDLE_UNKNOWN = "ignore"

# ----------- ANN HYPERPARAMETERS -----------
ANN_INPUT_DIM = None  # To be set dynamically
ANN_HIDDEN1 = 256
ANN_HIDDEN2 = 128
ANN_DROPOUT = 0.3
ANN_EPOCHS = 15
ANN_LR = 0.0001
BATCH_SIZE = 64

# ----------- CROSS-VALIDATION -----------
CV_SPLITS = 5

# ----------- MODEL FILES -----------
RF_MODEL_FILE = MODEL_PATH + "rf_model.pkl"
XGB_MODEL_FILE = MODEL_PATH + "xgb_model.pkl"
ANN_MODEL_FILE = MODEL_PATH + "ann_model.pth"
PREPROCESSOR_FILE = MODEL_PATH + "preprocessor.pkl"

# ----------- FEATURE SELECTION FILES (Top 20) -----------
TOP_20_INDICES_FILE = MODEL_PATH + "top20_feature_indices.pkl"
TOP_20_NAMES_FILE = MODEL_PATH + "top20_feature_names.pkl"

# ----------- MODEL REGISTRY NAMES -----------
REGISTER_RF_NAME = "CreditRFClassifier"
REGISTER_XGB_NAME = "CreditXGBClassifier"
