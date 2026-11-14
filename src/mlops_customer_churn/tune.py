
import json
from pathlib import Path

import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/raw/Telco-Customer-Churn.csv"
REPORT_PATH = BASE_DIR / "reports/tuning_results.json"


def main():
    print("Chargement du dataset...")

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # On conserve le jeu de test pour l'évaluation finale.
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    # Hyperparamètres à comparer
    param_grid = {
        "classifier__C": [
            0.001, 0.01, 0.1, 1, 10, 100
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("customer-churn-hyperparameter-tuning")

    print("Optimisation en cours...")

    with mlflow.start_run(run_name="logistic-regression-grid-search"):
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "roc_auc")
        mlflow.log_param("train_random_state", 42)
        mlflow.log_param("class_weight", "balanced")

        search.fit(X_train, y_train)

        results = []

        for i, params in enumerate(search.cv_results_["params"]):
            c_value = params["classifier__C"]

            mean_auc = float(
                search.cv_results_["mean_test_score"][i]
            )
            std_auc = float(
                search.cv_results_["std_test_score"][i]
            )

            result = {
                "C": float(c_value),
                "roc_auc_mean": mean_auc,
                "roc_auc_std": std_auc,
            }
            results.append(result)

            # Un run MLflow pour chaque configuration
            with mlflow.start_run(
                run_name=f"C={c_value}",
                nested=True,
            ):
                mlflow.log_param("C", c_value)
                mlflow.log_metric("roc_auc_mean", mean_auc)
                mlflow.log_metric("roc_auc_std", std_auc)

            print(
                f"C={c_value} | "
                f"ROC-AUC={mean_auc:.4f} ± {std_auc:.4f}"
            )

        best_params = {
            key: float(value)
            for key, value in search.best_params_.items()
        }

        report = {
            "best_params": best_params,
            "best_roc_auc": float(search.best_score_),
            "results": results,
        }

        mlflow.log_params(best_params)
        mlflow.log_metric(
            "best_roc_auc", search.best_score_
        )

        REPORT_PATH.parent.mkdir(
            parents=True, exist_ok=True
        )

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        mlflow.log_artifact(str(REPORT_PATH))

    print("\n--- MEILLEURS PARAMÈTRES ---")
    print(json.dumps(best_params, indent=4))
    print(f"Meilleur ROC-AUC CV : {search.best_score_:.4f}")
    print(f"Rapport enregistré : {REPORT_PATH}")


if __name__ == "__main__":
    main()
