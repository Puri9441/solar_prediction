import argparse
import os

from clearml import PipelineDecorator, Task


PROJECT_NAME = os.getenv("CLEARML_PROJECT_NAME", "Solar_Energy_Prediction")
EXECUTION_QUEUE = os.getenv("CLEARML_EXECUTION_QUEUE", "default")


@PipelineDecorator.component(
    cache=True,
    execution_queue=EXECUTION_QUEUE,
    packages=["clearml", "joblib", "matplotlib", "numpy", "pandas", "scikit-learn"],
)
def train_step(
    data_path: str,
    artifact_dir: str,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
):
    from train import train_model

    train_model(
        params={
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
        },
        task=Task.current_task(),
        data_path=data_path,
        artifact_dir=artifact_dir,
    )
    return os.path.join(artifact_dir, "solar_energy_model.joblib")


@PipelineDecorator.component(
    cache=False,
    execution_queue=EXECUTION_QUEUE,
    packages=["clearml", "joblib", "pandas", "scikit-learn"],
)
def validation_step(model_path: str) -> bool:
    from validation import validate_model

    is_valid = validate_model(model_path)
    if not is_valid:
        raise RuntimeError(f"Model validation failed: {model_path}")
    return is_valid


@PipelineDecorator.pipeline(
    name="solar-energy-training-validation",
    project=PROJECT_NAME,
    version="1.0.0",
    pipeline_execution_queue=EXECUTION_QUEUE,
)
def solar_energy_pipeline(
    data_path: str,
    artifact_dir: str,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
):
    model_path = train_step(
        data_path=data_path,
        artifact_dir=artifact_dir,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )
    validation_passed = validation_step(model_path=model_path)
    return validation_passed


def main():
    parser = argparse.ArgumentParser(description="Run the ClearML solar energy pipeline.")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Submit pipeline components to the configured ClearML agent queue.",
    )
    parser.add_argument("--data-path", default=os.getenv("SOLAR_DATA_PATH", "data/train.csv"))
    parser.add_argument(
        "--artifact-dir",
        default=os.getenv("SOLAR_ARTIFACTS_DIR", "artifacts"),
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=2)
    args = parser.parse_args()

    if not args.remote:
        PipelineDecorator.run_locally()

    solar_energy_pipeline(
        data_path=os.path.abspath(args.data_path),
        artifact_dir=os.path.abspath(args.artifact_dir),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
    )


if __name__ == "__main__":
    main()
