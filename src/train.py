from clearml import Task
import argparse
import json
import os

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_PARAMS = {
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
}


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training file not found: {path}")
    return pd.read_csv(path)


def prepare_features(df):
    if "energy_generated" not in df.columns:
        raise ValueError("Expected target column 'energy_generated' in training data")
    X = df.drop(columns=["energy_generated"])
    y = df["energy_generated"]
    return X, y


def train_model(params=None, task=None, data_path=None, artifact_dir=None):
    params = {**DEFAULT_PARAMS, **(params or {})}
    task = task or Task.current_task()
    data_path = data_path or os.path.join(os.getcwd(), "data", "train.csv")
    artifact_dir = artifact_dir or os.path.join(os.getcwd(), "artifacts")
    model_path = os.path.join(artifact_dir, "solar_energy_model.joblib")

    print("Starting solar energy regression training...")
    df = load_data(data_path)
    X, y = prepare_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    random_state=params["random_state"],
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    metrics = {
        "mae": mean_absolute_error(y_val, y_pred),
        "mse": mean_squared_error(y_val, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
        "r2": r2_score(y_val, y_pred),
    }

    logger = task.get_logger() if task else None
    for metric_name, metric_value in metrics.items():
        if logger:
            logger.report_scalar(
                title="metrics",
                series=metric_name,
                iteration=1,
                value=float(metric_value),
            )
        print(f"{metric_name}: {metric_value:.4f}")

    os.makedirs(artifact_dir, exist_ok=True)
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    names = list(metrics.keys())
    ax1.bar(names, [metrics[name] for name in names], color="tab:blue")
    ax1.set_title("Validation metrics")
    ax1.set_ylabel("Value")
    fig1.tight_layout()
    metrics_plot = os.path.join(artifact_dir, "metrics_plot.png")
    fig1.savefig(metrics_plot)
    plt.close(fig1)
    if task:
        task.upload_artifact(name="metrics_plot", artifact_object=metrics_plot)

    actual_vs_predicted_plot = os.path.join(artifact_dir, "actual_vs_predicted.png")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_val, y_pred, alpha=0.7)
    prediction_min = min(float(y_val.min()), float(np.min(y_pred)))
    prediction_max = max(float(y_val.max()), float(np.max(y_pred)))
    ax3.plot(
        [prediction_min, prediction_max],
        [prediction_min, prediction_max],
        linestyle="--",
        color="tab:red",
    )
    ax3.set_xlabel("Actual energy generation")
    ax3.set_ylabel("Predicted energy generation")
    ax3.set_title("Actual vs. predicted validation values")
    fig3.tight_layout()
    fig3.savefig(actual_vs_predicted_plot)
    plt.close(fig3)
    if task:
        task.upload_artifact(
            name="actual_vs_predicted",
            artifact_object=actual_vs_predicted_plot,
        )

    predictions_file = os.path.join(artifact_dir, "validation_predictions.csv")
    pd.DataFrame(
        {"actual": y_val.to_numpy(), "predicted": y_pred}
    ).to_csv(predictions_file, index=False)
    if task:
        task.upload_artifact(
            name="validation_predictions",
            artifact_object=predictions_file,
        )

    importances = pipeline.named_steps["model"].feature_importances_
    feature_names = X.columns.tolist()
    idx = np.argsort(importances)
    fig2, ax2 = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.3)))
    ax2.barh([feature_names[i] for i in idx], importances[idx], color="tab:green")
    ax2.set_title("Feature importances")
    fig2.tight_layout()
    importances_plot = os.path.join(artifact_dir, "feature_importances.png")
    fig2.savefig(importances_plot)
    plt.close(fig2)
    if task:
        task.upload_artifact(name="feature_importances", artifact_object=importances_plot)
        for name, importance in zip(feature_names, importances):
            logger.report_scalar(
                title="feature_importances",
                series=name,
                iteration=1,
                value=float(importance),
            )

    dump(pipeline, model_path)
    if task:
        task.upload_artifact(name="solar_energy_model", artifact_object=model_path)
    print(f"Model saved to: {model_path}")

    metrics_file = os.path.join(artifact_dir, "metrics.json")
    with open(metrics_file, "w", encoding="utf8") as file:
        json.dump({"metrics": metrics, "params": params}, file, indent=2)
    if task:
        task.upload_artifact(name="metrics_json", artifact_object=metrics_file)

    sentinel = os.path.join(artifact_dir, "_SUCCESS")
    with open(sentinel, "w", encoding="utf8") as file:
        file.write("success")
    print(f"Wrote success sentinel: {sentinel}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, help="Number of trees")
    parser.add_argument("--max_depth", type=int, help="Max tree depth")
    parser.add_argument("--min_samples_split", type=int, help="Min samples split")
    parser.add_argument("--task_name", type=str, help="ClearML task name override")
    args, _ = parser.parse_known_args()

    params = DEFAULT_PARAMS.copy()
    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    if args.max_depth is not None:
        params["max_depth"] = args.max_depth
    if args.min_samples_split is not None:
        params["min_samples_split"] = args.min_samples_split

    task = Task.init(
        project_name=os.getenv("CLEARML_PROJECT_NAME", "Solar_Energy_Prediction"),
        task_name=args.task_name or os.getenv("CLEARML_TASK_NAME", "training_agent"),
    )
    task.connect(params)
    train_model(params=params, task=task)


if __name__ == "__main__":
    main()
