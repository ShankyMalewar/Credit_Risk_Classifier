import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from credit_risk_classifier.data import get_merged_data
from credit_risk_classifier.preprocessing import preprocess_and_split_clean
from config import MODEL_PATH, TOP_20_INDICES_FILE, TOP_20_NAMES_FILE, PREPROCESSOR_FILE

#Loading and preprocess
df = get_merged_data()
X_train, _, y_train, _, preprocessor = preprocess_and_split_clean(df)

X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
feature_names = preprocessor.get_feature_names_out()

rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight='balanced')
rf.fit(X_train_dense, y_train)

#top 20 feature indices and names
importances = rf.feature_importances_
top_20_indices = np.argsort(importances)[-20:]
top_20_names = feature_names[top_20_indices]

#saving
joblib.dump(top_20_indices.tolist(), TOP_20_INDICES_FILE)
joblib.dump(top_20_names.tolist(), TOP_20_NAMES_FILE)

print("✅ Top-20 feature indices and names saved.")
