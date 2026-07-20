from clearml import Task
import json
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

project_name = os.getenv('CLEARML_PROJECT_NAME', 'Solar_Energy_Prediction')
task_name = os.getenv('CLEARML_TASK_NAME', 'training_agent')

# parse optional CLI overrides for hyperparameters (used by sweep)
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, help='Number of trees')
parser.add_argument('--max_depth', type=int, help='Max tree depth')
parser.add_argument('--min_samples_split', type=int, help='Min samples split')
parser.add_argument('--task_name', type=str, help='ClearML task name override')
args, _ = parser.parse_known_args()

if args.task_name:
    task_name = args.task_name

task = Task.init(project_name=project_name, task_name=task_name)

params = {
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
}
# apply CLI overrides if provided
if args.n_estimators is not None:
    params['n_estimators'] = args.n_estimators
if args.max_depth is not None:
    params['max_depth'] = args.max_depth
if args.min_samples_split is not None:
    params['min_samples_split'] = args.min_samples_split

task.connect(params)

DATA_DIR = os.path.join(os.getcwd(), 'data')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
MODEL_DIR = os.path.join(os.getcwd(), 'artifacts')
MODEL_PATH = os.path.join(MODEL_DIR, 'solar_energy_model.joblib')


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training file not found: {path}")
    return pd.read_csv(path)


def prepare_features(df):
    if 'energy_generated' not in df.columns:
        raise ValueError("Expected target column 'energy_generated' in training data")
    X = df.drop(columns=['energy_generated'])
    y = df['energy_generated']
    return X, y


def train():
    print('Starting solar energy regression training...')

    df = load_data(TRAIN_FILE)
    X, y = prepare_features(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=params['test_size'],
        random_state=params['random_state'],
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            random_state=params['random_state'],
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    metrics = {
        'mae': mean_absolute_error(y_val, y_pred),
        'mse': mean_squared_error(y_val, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
        'r2': r2_score(y_val, y_pred),
    }

    logger = task.get_logger()
    for metric_name, metric_value in metrics.items():
        logger.report_scalar(title='metrics', series=metric_name, iteration=1, value=metric_value)
        print(f'{metric_name}: {metric_value:.4f}')

    # Create and upload plots for ClearML
    try:
        import matplotlib.pyplot as plt

        # Metrics bar chart
        fig1, ax1 = plt.subplots()
        names = list(metrics.keys())
        vals = [metrics[n] for n in names]
        ax1.bar(names, vals, color='tab:blue')
        ax1.set_title('Validation metrics')
        ax1.set_ylabel('Value')
        fig1.tight_layout()
        os.makedirs(MODEL_DIR, exist_ok=True)
        metrics_plot = os.path.join(MODEL_DIR, 'metrics_plot.png')
        fig1.savefig(metrics_plot)
        plt.close(fig1)
        task.upload_artifact(name='metrics_plot', artifact_object=metrics_plot)

        # Feature importances
        importances = pipeline.named_steps['model'].feature_importances_
        feature_names = X.columns.tolist()
        # horizontal bar chart sorted
        idx = np.argsort(importances)
        fig2, ax2 = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.3)))
        ax2.barh([feature_names[i] for i in idx], importances[idx], color='tab:green')
        ax2.set_title('Feature importances')
        fig2.tight_layout()
        importances_plot = os.path.join(MODEL_DIR, 'feature_importances.png')
        fig2.savefig(importances_plot)
        plt.close(fig2)
        task.upload_artifact(name='feature_importances', artifact_object=importances_plot)

        # Also report each feature importance as a scalar (visible in ClearML charts)
        for name, imp in zip(feature_names, importances):
            logger.report_scalar(title='feature_importances', series=name, iteration=1, value=float(imp))
    except Exception as e:
        print(f'Plot generation skipped: {e}')

    # Save and upload the model artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(pipeline, MODEL_PATH)
    task.upload_artifact(name='solar_energy_model', artifact_object=MODEL_PATH)
    print(f'Model saved to: {MODEL_PATH}')

    # write metrics and success sentinel so downstream validation can detect success
    try:
        metrics_file = os.path.join(MODEL_DIR, 'metrics.json')
        with open(metrics_file, 'w', encoding='utf8') as f:
            json.dump({'metrics': metrics, 'params': params}, f, indent=2)
        task.upload_artifact(name='metrics_json', artifact_object=metrics_file)

        sentinel = os.path.join(MODEL_DIR, '_SUCCESS')
        with open(sentinel, 'w', encoding='utf8') as f:
            f.write('success')
        print(f'Wrote success sentinel: {sentinel}')
    except Exception as e:
        print(f'Failed writing sentinel or metrics file: {e}')


if __name__ == '__main__':
    train()
