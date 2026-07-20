# Solar Energy Generation Prediction MLOps

This repository contains a simple solar energy generation regression pipeline built with ClearML and scikit-learn.

## Getting Started

1. **Clone the repository**: `git clone <your-repo-url>`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Prepare your dataset**: place `train.csv` inside `./data` with a target column named `energy_generated`.
4. **Set ClearML environment variables**:
   - `CLEARML_API_ACCESS_KEY`
   - `CLEARML_API_SECRET_KEY`
   - optionally `CLEARML_API_HOST=https://api.clear.ml`
5. **Run the pipeline**: `docker-compose up --build`

## Project structure

- `src/train.py` - training pipeline that logs metrics and artifacts to ClearML.
- `src/validation.py` - validates the trained model artifact.
- `data/data_registry.py` - generates a dataset lineage hash from `data/train.csv`.

## Notes

The pipeline uses a `RandomForestRegressor` baseline and reports MAE, MSE, RMSE, and R² metrics to ClearML.
