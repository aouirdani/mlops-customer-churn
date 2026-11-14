
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/raw/Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models/churn_model.joblib"
TUNING_PATH = BASE_DIR / "reports/tuning_results.json"
REPORT_PATH = BASE_DIR / "reports/final_evaluation.json"


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }


def main():
    print("Chargement des données et du modèle...")

    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Modèle initial déjà entraîné et sauvegardé
    baseline = joblib.load(MODEL_PATH)

    # Paramètres sélectionnés en validation croisée
    with open(TUNING_PATH, encoding="utf-8") as f:
        tuning = json.load(f)

    best_params = tuning["best_params"]

    # Nouveau modèle avec les paramètres optimisés
    optimized = joblib.load(MODEL_PATH)
    optimized.set_params(**best_params)
    optimized.fit(X_train, y_train)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("customer-churn-final-evaluation")

    results = {}

    for name, model in {
        "baseline": baseline,
        "optimized": optimized,
    }.items():
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics

        with mlflow.start_run(run_name=name):
            mlflow.log_param(
                "C", model.get_params()["classifier__C"]
            )
            mlflow.log_metrics({
                key: value
                for key, value in metrics.items()
                if key != "confusion_matrix"
            })

        print(f"\n{name.upper()}")
        print(json.dumps(metrics, indent=4))

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nRapport enregistré : {REPORT_PATH}")


if __name__ == "__main__":
    main()
