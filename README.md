# Solar Energy Prediction MLOps

An end-to-end machine-learning project for predicting solar energy generation.
It includes model training and validation, ClearML experiment tracking and
pipelines, Docker Compose services, and a Flask prediction interface.

## Workflow

1. Read `data/train.csv` and split it into training and validation sets.
2. Train a scikit-learn pipeline using `StandardScaler` and
   `RandomForestRegressor`.
3. Evaluate the model with MAE, MSE, RMSE, and R².
4. Save the model and reports under `artifacts/`.
5. Optionally log runs, metrics, and artifacts to ClearML.
6. Validate the saved model and serve predictions through Flask.

The default input features are `temperature`, `humidity`,
`solar_irradiance`, `wind_speed`, and `cloud_cover`. The target column is
`energy_generated`.

## Repository layout

- `src/train.py` — reusable training function and standalone training command.
- `src/validation.py` — loads a saved model and checks that inference works.
- `src/clearml_pipeline.py` — ClearML pipeline coordinating training and validation.
- `src/serve.py` — Flask UI, health endpoint, and prediction API.
- `src/sweep.py` — simple command-line hyperparameter sweep.
- `data/train.csv` — training data.
- `artifacts/` — model, metrics, plots, predictions, and success marker.
- `docker-compose.yml` and `Dockerfile` — container configuration.
- `ARCHITECTURE.md` — architecture overview.

## Requirements

- Python 3.11 or compatible Python version.
- Docker Desktop and Docker Compose for container execution.
- ClearML account and API credentials for ClearML tracking or pipeline runs.
- A CSV at `data/train.csv` with numeric feature columns and the
  `energy_generated` target.

## Run locally

From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Configure ClearML credentials in the same PowerShell session. Use a rotated
access key and secret; never put credentials in source files or commit them:

```powershell
$env:CLEARML_API_ACCESS_KEY = Read-Host "ClearML access key"
$secureSecret = Read-Host "ClearML secret key" -AsSecureString
$env:CLEARML_API_SECRET_KEY = [System.Net.NetworkCredential]::new("", $secureSecret).Password
$env:CLEARML_API_HOST = "https://api.clear.ml"
```

Train, validate, and start the web app in separate terminals:

```powershell
python src\train.py
python src\validation.py
python src\serve.py
```

Open `http://127.0.0.1:5000/`. Check `http://127.0.0.1:5000/health`; a loaded
model returns `{"status":"ok"}`.

## Run with Docker Compose

Set the ClearML environment variables as above in PowerShell, then run:

```powershell
docker compose up --build
```

Compose starts the training, validation, and serving services. Training writes
the model and `_SUCCESS` marker to the shared `artifacts/` directory.
Validation waits for that marker, and the web service is available at
`http://localhost:5000/`.

If only the web service needs rebuilding:

```powershell
docker compose up -d --build serve
```

Inspect status and logs with:

```powershell
docker compose ps
docker compose logs --tail 100 training-agent validation-agent serve
```

## ClearML tracking and pipeline

The standalone `src/train.py` task logs hyperparameters, evaluation metrics,
feature importances, plots, and model/data artifacts to the
`Solar_Energy_Prediction` project by default. Override the project or task name
with `CLEARML_PROJECT_NAME` or `CLEARML_TASK_NAME`.

The ClearML pipeline in `src/clearml_pipeline.py` has two steps:

1. Train the model and write metrics, predictions, plots, and model artifacts.
2. Load the saved model and confirm it can make a prediction.

Run it locally for debugging while still tracking the run in ClearML:

```powershell
python src\clearml_pipeline.py
```

Submit the pipeline and steps to ClearML Agents:

```powershell
python src\clearml_pipeline.py --remote
```

By default, the controller and components use the `default` queue. A ClearML
Agent must be running and servicing that queue. The agent needs valid ClearML
credentials and access to the project code, `data/train.csv`, and the artifacts
directory. For remote workers, those paths must exist on the worker machine or
be provided through shared storage; a local Windows path or Docker volume is
not automatically available to a remote agent.

To use a different queue, set the variable before starting the pipeline and
ensure an agent services that queue:

```powershell
$env:CLEARML_EXECUTION_QUEUE = "my-queue"
python src\clearml_pipeline.py --remote
```

In ClearML, open the `Solar_Energy_Prediction` project and the
`solar-energy-training-validation` pipeline run. The training step's **Scalars**
show metrics and feature importances. Its **Artifacts** include the model,
metrics JSON, metric plot, feature-importance plot, actual-versus-predicted
plot, and validation-predictions CSV.

## Artifacts and metrics

Successful training writes:

- `artifacts/solar_energy_model.joblib` — fitted scikit-learn pipeline.
- `artifacts/metrics.json` — evaluation metrics and training parameters.
- `artifacts/metrics_plot.png` — validation-metric chart.
- `artifacts/feature_importances.png` — feature-importance chart.
- `artifacts/actual_vs_predicted.png` — validation actual-versus-predicted chart.
- `artifacts/validation_predictions.csv` — actual and predicted validation values.
- `artifacts/_SUCCESS` — marker used by the Compose validation service.

The currently documented model result is MAE `0.22375`, MSE `0.09603`, RMSE
`0.30989`, and R² `0.93222`. New training runs may produce different results
when the dataset or training configuration changes.

## Prediction API

`GET /` opens the browser form. `GET /health` reports whether a model is loaded.
Submit named features to `POST /predict` as JSON:

```json
{
  "instances": [
    {
      "temperature": 25.0,
      "humidity": 45.0,
      "solar_irradiance": 650.0,
      "wind_speed": 5.0,
      "cloud_cover": 20.0
    }
  ]
}
```

The response contains a `predictions` array. For example, from PowerShell:

```powershell
$body = @{
    instances = @(
        @{
            temperature = 25.0
            humidity = 45.0
            solar_irradiance = 650.0
            wind_speed = 5.0
            cloud_cover = 20.0
        }
    )
} | ConvertTo-Json -Depth 4

Invoke-RestMethod -Method Post `
    -Uri http://localhost:5000/predict `
    -ContentType "application/json" `
    -Body $body
```

The serving application uses the feature order recorded by the fitted pipeline
when available.

## Troubleshooting

- **ClearML `MissingConfigError`:** configure valid ClearML credentials in the
  environment of the process or agent that runs the task.
- **Pipeline remains Pending:** verify a ClearML Agent is online and listening
  to the queue configured by `CLEARML_EXECUTION_QUEUE` (default: `default`).
- **Agent reports a Git diff error:** ensure the code submitted to the Agent is
  committed and pushed to an accessible repository branch.
- **Model missing in `/health`:** confirm training completed and that
  `artifacts/solar_energy_model.joblib` is present in the serving container's
  mounted artifacts directory.
- **Prediction reports missing features:** send every named input feature
  listed above.
- **Port 5000 already in use:** stop another local Flask process or change the
  Compose port mapping before restarting the service.

## Export this README to PDF

With Pandoc installed, run:

```powershell
pandoc README.md -o README.pdf
```

For production deployment, replace Flask's development server with a
production WSGI server and add appropriate authentication and input validation.
