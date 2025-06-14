from setuptools import setup, find_packages

setup(
    name="credit_risk_classifier",
    version="0.1.0",
    description="A credit risk classification package with MLflow tracking.",
    author="Shanky",
    packages=find_packages(),
    install_requires=[
        "numpy", "pandas", "scikit-learn", "torch", "xgboost", "mlflow","joblib"
    ],
    include_package_data=True,
    python_requires=">=3.8",
)
