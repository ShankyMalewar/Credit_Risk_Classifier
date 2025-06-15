import joblib
import os

COLAB_DATA_PATH = "colab_data"

def load_fixed_colab_split():
    X_train = joblib.load(os.path.join(COLAB_DATA_PATH, "X_train_top20.pkl"))
    X_test = joblib.load(os.path.join(COLAB_DATA_PATH, "X_test_top20.pkl"))
    y_train = joblib.load(os.path.join(COLAB_DATA_PATH, "y_train_top20.pkl"))
    y_test = joblib.load(os.path.join(COLAB_DATA_PATH, "y_test_top20.pkl"))
    return X_train, X_test, y_train, y_test
