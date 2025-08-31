# scripts/ml_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_clean_data(path="data/labeled_features.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    print(f"Number of rows loaded: {len(df)}")

    # Clean extreme distance values
    for col in ["dist_to_road_m", "dist_to_factory_m", "dist_to_farmland_m"]:
        df[col] = df[col].apply(lambda x: x if x < 1e6 else np.nan)

    # Fill NaNs with median values
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    print("\nTraining Random Forest with hyperparameter tuning...")
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]  # valid values only
    }

    rf_search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=20, cv=3, random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)

    print(f"Best RF hyperparameters: {rf_search.best_params_}")
    y_pred = rf_search.predict(X_test)
    print("\n=== Random Forest Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("\n=== Random Forest Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Plot feature importance
    plot_feature_importance(rf_search.best_estimator_, feature_names, title="Random Forest Feature Importance")

    return rf_search.best_estimator_

def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    print("\nTraining XGBoost classifier...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    xgb_search = RandomizedSearchCV(
        xgb, param_distributions=param_dist,
        n_iter=20, cv=3, random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)

    print(f"Best XGBoost hyperparameters: {xgb_search.best_params_}")
    y_pred = xgb_search.predict(X_test)
    print("\n=== XGBoost Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("\n=== XGBoost Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Plot feature importance
    plot_feature_importance(xgb_search.best_estimator_, feature_names, title="XGBoost Feature Importance")

    return xgb_search.best_estimator_

def main():
    df = load_and_clean_data()

    features = ["pm2_5","pm10","no2","so2","co","o3","aqi",
                "temperature","humidity","wind_speed",
                "dist_to_road_m","dist_to_factory_m","dist_to_farmland_m"]
    X = df[features]
    y = df["pollution_source"]

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    best_rf = train_random_forest(X_train, y_train, X_test, y_test, features)
    best_xgb = train_xgboost(X_train, y_train, X_test, y_test, features)

    # Save label encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, "models/label_encoder.pkl")

    # Choose best model (example: RF)
    best_model = best_rf
    model_path = os.path.join("models", "pollution_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nBest-performing model saved to: {model_path}")

if __name__ == "__main__":
    main()
