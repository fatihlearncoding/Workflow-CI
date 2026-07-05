"""modelling.py (MLProject / Workflow-CI)
==========================================
Re-training model klasifikasi arah harga pangan Jatim di dalam MLflow Project.
Dijalankan otomatis oleh workflow CI (GitHub Actions) via `mlflow run`.

Tracking menggunakan file store lokal (mlruns/) milik runner CI.
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

EXPERIMENT_NAME = "Harga Pangan Jatim - CI Retraining"


def load_data(data_dir: str):
    with open(os.path.join(data_dir, "feature_info.json")) as f:
        info = json.load(f)
    features = info["numeric_features"] + info["categorical_features"]
    target = info["target"]
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train[features], train[target], test[features], test[target]


def main(data_dir, n_estimators, max_depth, min_samples_leaf):
    # Catatan: saat dijalankan via `mlflow run`, run aktif sudah dibuat oleh CLI
    # (MLFLOW_RUN_ID di environment) — cukup lanjutkan run tersebut.
    # Eksperimen ditentukan lewat flag --experiment-name pada perintah `mlflow run`.
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=False)

    X_train, y_train, X_test, y_test = load_data(data_dir)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        mlflow.log_metric("train_seconds", time.perf_counter() - t0)

        y_pred = model.predict(X_test)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_f1_macro", f1_score(y_test, y_pred, average="macro"))

        # Simpan model + artefak evaluasi
        mlflow.sklearn.log_model(model, "model", input_example=X_test.iloc[:5])

        labels = sorted(y_test.unique())
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred, labels=labels), display_labels=labels
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        fig.tight_layout()
        fig.savefig("training_confusion_matrix.png", dpi=120)
        mlflow.log_artifact("training_confusion_matrix.png")

        print(classification_report(y_test, y_pred))
        print("MLFLOW_RUN_ID=" + run.info.run_id)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="harga_pangan_jatim_preprocessing")
    p.add_argument("--n-estimators", type=float, default=200)
    p.add_argument("--max-depth", type=float, default=14)
    p.add_argument("--min-samples-leaf", type=float, default=20)
    a = p.parse_args()
    main(a.data, int(a.n_estimators), int(a.max_depth), int(a.min_samples_leaf))
