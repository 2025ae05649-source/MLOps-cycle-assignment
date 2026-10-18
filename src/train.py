"""
Step 3: Train models and track experiments with MLflow.

Trains Logistic Regression and Random Forest. For each model:
  - GridSearchCV (5-fold) to tune a few hyperparameters
  - cross-validation metrics + test set metrics
  - everything logged to MLflow (params, metrics, plots, model)
The best model (by test ROC-AUC) is saved to models/model.pkl

Run from project root:  python src/train.py
See results:            mlflow ui --port 5001
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,  # noqa: E402
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.preprocess import (build_preprocessor, load_data,  # noqa: E402
                            split_features_target)

DATA_PATH = os.path.join(BASE_DIR, "data", "heart_clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "reports", "figures")
RANDOM_STATE = 42


def get_models():
    # model name -> (estimator, small grid of params to try)
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000),
            {"clf__C": [0.01, 0.1, 1, 10],
             "clf__penalty": ["l2"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"clf__n_estimators": [100, 200],
             "clf__max_depth": [4, 6, None],
             "clf__min_samples_split": [2, 5]},
        ),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_prob),
    }


def save_plots(name, model, X_test, y_test):
    os.makedirs(PLOT_DIR, exist_ok=True)

    cm_path = os.path.join(PLOT_DIR, f"{name}_confusion_matrix.png")
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                          cmap="Blues")
    plt.title(f"Confusion matrix - {name}")
    plt.savefig(cm_path)
    plt.close()

    roc_path = os.path.join(PLOT_DIR, f"{name}_roc_curve.png")
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC curve - {name}")
    plt.savefig(roc_path)
    plt.close()
    return [cm_path, roc_path]


def main():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    mlflow.set_experiment("heart-disease")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_name, best_model, best_auc = None, None, -1
    all_results = {}

    for name, (estimator, grid) in get_models().items():
        print(f"\n===== {name} =====")
        pipe = Pipeline([("preprocess", build_preprocessor()),
                         ("clf", estimator)])

        with mlflow.start_run(run_name=name):
            search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc",
                                  n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print("Best params:", search.best_params_)

            # cross validation on training data with the best params
            cv_scores = cross_validate(
                model, X_train, y_train, cv=cv,
                scoring=["accuracy", "precision", "recall", "roc_auc"])
            cv_metrics = {f"cv_{k[5:]}": v.mean()
                          for k, v in cv_scores.items()
                          if k.startswith("test_")}

            test_metrics = evaluate(model, X_test, y_test)
            metrics = {**cv_metrics, **test_metrics}
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            mlflow.log_param("model_type", name)
            mlflow.log_params({k.replace("clf__", ""): v
                               for k, v in search.best_params_.items()})
            mlflow.log_metrics(metrics)
            for p in save_plots(name, model, X_test, y_test):
                mlflow.log_artifact(p, artifact_path="plots")
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     input_example=X_test.head(2))

            all_results[name] = {"best_params": search.best_params_,
                                 **metrics}
            if test_metrics["test_roc_auc"] > best_auc:
                best_name, best_model = name, model
                best_auc = test_metrics["test_roc_auc"]

    # save the best model as a pickle for the API
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "results": all_results}, f,
                  indent=2)
    print(f"\nBest model: {best_name} (test ROC-AUC = {best_auc:.4f})")
    print("Saved to models/model.pkl")


if __name__ == "__main__":
    main()
