# Solar Energy Prediction — Project Documentation

This document describes the full project: purpose, architecture, setup, how to run training, validation, serving, ClearML integration, artifacts, troubleshooting, and how to convert this documentation to a PDF.

1. Project Overview

---

- Goal: Train and serve a regression model to predict solar energy generation from meteorological features.
- Stack: Python, scikit-learn, pandas, joblib, Flask, ClearML, Docker Compose.
- Components:
  - `training-agent`: trains a RandomForest pipeline and writes artifacts to `./artifacts`.
  - `validation-agent`: waits for `artifacts/_SUCCESS`, then validates the saved model.
  - `serve`: a Flask app that serves predictions via JSON API and a browser form.

2. Architecture & Workflow

---

- Data: `data/train.csv` (CSV with a target column `energy_generated`).
- Training pipeline (`src/train.py`):
  - Reads `data/train.csv`, splits data, trains a `Pipeline` with `StandardScaler` + `RandomForestRegressor`.
  - Saves model to `artifacts/solar_energy_model.joblib` using `joblib.dump`.
  - Generates `metrics.json`, `metrics_plot.png`, `feature_importances.png` and uploads artifacts to ClearML.
  - Writes `artifacts/_SUCCESS` sentinel on success.
- Validation (`src/validation.py`):
  - Waits for sentinel (in Docker Compose) then loads model and runs a sample inference to confirm the artifact is valid.
- Serving (`src/serve.py`):
  - Loads model from `/app/artifacts/solar_energy_model.joblib` (bind-mounted from host `./artifacts`).
  - Exposes endpoints:
    - `GET /` — browser form for entering features and submitting predictions.
    - `GET /health` — returns `{"status":"ok"}` when model loaded.
    - `POST /predict` — accepts JSON payloads and returns predictions.
  - The server uses the model pipeline's feature order (reads `feature_names_in_` when available) to reorder inputs correctly.

3. Repository Layout

---

- `docker-compose.yml` — defines `training-agent`, `validation-agent`, `serve` services.
- `Dockerfile` — base image and dependency install.
- `requirements.txt` — python deps.
- `data/train.csv` — training dataset (user-provided).
- `artifacts/` — output artifacts and saved model.
- `src/train.py`, `src/validation.py`, `src/serve.py` — core scripts.
- `data/data_registry.py` — optional dataset lineage helper.

4. Data Format

---

- `data/train.csv` must contain columns for features and the target column `energy_generated`.
- Example header order (the training file in this repo):
  temperature,humidity,solar_irradiance,wind_speed,cloud_cover,energy_generated
- Note: During serving, the code reads the pipeline's `feature_names_in_` (if available) and reorders inputs, so your JSON requests must name features (or provide an array matching the expected order).

5. Setup & Prerequisites

---

- System requirements:
  - Docker and Docker Compose installed.
  - (Optional) Python 3.11 for running scripts locally.
- Environment variables required for ClearML (if you use ClearML features):
  - `CLEARML_API_ACCESS_KEY`
  - `CLEARML_API_SECRET_KEY`
  - (optional) `CLEARML_API_HOST=https://api.clear.ml`

6. Quick Start — Docker Compose (recommended)

---

- Place your `train.csv` into `./data`.
- Ensure ClearML env vars are set (if you want ClearML logging):

PowerShell example:

```powershell
$env:CLEARML_API_ACCESS_KEY = "<your_key>"
$env:CLEARML_API_SECRET_KEY = "<your_secret>"
$env:CLEARML_API_HOST = "https://api.clear.ml"  # optional
docker-compose up --build
```

- Services will run in sequence (training writes `_SUCCESS`, validation waits for sentinel). The `serve` service stays running and listens on port 5000.

7. Running Locally without Docker

---

- Create a Python venv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\Activate.ps1 # Windows PowerShell
pip install -r requirements.txt
```

- Train locally:

```bash
python src/train.py
```

- Validate locally (once model saved):

```bash
python src/validation.py
```

- Serve locally:

```bash
python src/serve.py
# then open http://127.0.0.1:5000/ in your browser
```

8. API Specification

---

- `POST /predict`
  - Content-Type: `application/json`
  - Request body (preferred):

```json
{
  "instances": [
    {
      "temperature": 25.0,
      "humidity": 45.0,
      "wind_speed": 5.0,
      "cloud_cover": 20.0,
      "solar_irradiance": 650.0
    }
  ]
}
```

- Response:

```json
{"predictions": [<float>, ...]}
```

- `GET /health` — returns `{"status": "ok"}` (or `model_missing`).
- Browser: `GET /` loads a form and a demo JSON button that POSTs to `/predict`.

9. ClearML Integration

---

- `src/train.py` initializes a ClearML Task and reports metrics and uploads artifacts.
- Artifacts uploaded: the model file and metric plots. You still get local copies under `./artifacts`.
- If ClearML keys are missing, training still runs but remote logging will not occur.

10. Artifacts & Sentinel

---

- After successful training, the following are created under `./artifacts`:
  - `solar_energy_model.joblib` — saved pipeline
  - `metrics.json` — metrics and params
  - `metrics_plot.png` — metrics bar chart
  - `feature_importances.png` — feature importance chart
  - `_SUCCESS` — success sentinel file

- The `validation-agent` waits for `_SUCCESS` before running.

11. Troubleshooting

---

- 404 / 405 on `/predict`:
  - Ensure you're using the correct method: `GET /predict` returns the form; `POST /predict` is for predictions.
- Model not loaded (`model_missing` in `/health`):
  - Confirm `./artifacts/solar_energy_model.joblib` exists and is readable by the `serve` process (check volume mounts).
- Feature name mismatch error:
  - The model expects the same feature names/order used during training. Use named JSON `instances` or ensure arrays match `feature_names_in_`.
- ClearML MissingConfigError:
  - Set the `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY` environment variables.

12. Converting this document to PDF

---

- Option A (recommended): Install `pandoc` and `wkhtmltopdf` (or a LaTeX distribution) and run:

```bash
pandoc README_FULL.md -o README_FULL.pdf
```

- Option B (Windows, PowerShell) using `pandoc` (install from https://pandoc.org/installing.html):

```powershell
pandoc README_FULL.md -o README_FULL.pdf
```

- Option C (no external tools): You can open `README_FULL.md` in a Markdown editor (VS Code, Typora) and export to PDF from the GUI.

13. Security & Production Notes

---

- The Flask dev server is not production-grade. Use a WSGI server (gunicorn, waitress, uWSGI) for production.
- Sanitize and validate input payloads before exposing publicly.
- Consider adding authentication to the serving endpoint.

14. Next Improvements (suggestions)

---

- Add ClearML optimizer sweep integration (connect `src/sweep.py` to ClearML Hyperparameter sweeper).
- Add unit tests for model IO and API inputs.
- Add a production-ready `Dockerfile` for serving (use gunicorn).
- Add CI to run training quickly on example data and upload artifacts to a test ClearML project.

15. Files I created for you

---

- [README_FULL.md](README_FULL.md)
- A PowerShell script `scripts/generate_pdf.ps1` with pandoc instructions (created below).

16. Contact / Help

---

If you want, I can:

- Generate the PDF here (if you give permission to install pandoc in this environment), or
- Add example curl commands, automated tests, or CI configs.

-- End of documentation
