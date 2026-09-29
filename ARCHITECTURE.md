# Solar Energy Prediction — Architecture

## 1. Purpose and scope

This document describes the current repository architecture for data ingestion,
model training, evaluation, validation, prediction serving, and experiment
tracking. Diagrams use Mermaid so they can be rendered by GitHub and compatible
Markdown viewers.

The repository has two orchestration paths:

- **Docker Compose** runs the standalone training script, sentinel-gated
  validation script, and long-running Flask service on one host.
- **ClearML Pipeline** runs the `train_step` and `validation_step` as pipeline
  components. For remote execution, a ClearML Agent must service the configured
  queue, and the agent must be able to access the code, dataset, and artifact
  paths.

These are separate ways to orchestrate training and validation. Running both
against the same artifacts at the same time can cause concurrent writes.

## 2. System context and components

The application predicts `energy_generated` using the features
`temperature`, `humidity`, `solar_irradiance`, `wind_speed`, and `cloud_cover`.
Training and inference share a scikit-learn pipeline containing
`StandardScaler` and `RandomForestRegressor`.

### 2.1 UML component diagram

```mermaid
flowchart LR
    actor["«actor» Developer / API client"]

    subgraph repo["Project repository"]
        dataset[("data/train.csv")]
        registry["data_registry.py<br/>optional MD5 lineage hash"]
        trainer["train.py<br/>load, fit, evaluate, report"]
        validator["validation.py<br/>load model and smoke-test"]
        pipeline["clearml_pipeline.py<br/>train → validate"]
        web["serve.py<br/>Flask UI and REST API"]
        compose["docker-compose.yml<br/>local service orchestration"]
    end

    subgraph storage["Shared host storage"]
        artifacts[("artifacts/<br/>model, metrics, plots, predictions, _SUCCESS")]
    end

    clearml[("ClearML API / UI<br/>tasks, pipeline, scalars, artifacts")]

    actor -->|"supply CSV / start run"| dataset
    registry -.->|"hashes CSV"| dataset
    pipeline -->|"invokes"| trainer
    pipeline -->|"invokes after training"| validator
    compose -->|"starts"| trainer
    compose -->|"starts; waits for _SUCCESS"| validator
    compose -->|"starts"| web
    dataset -->|"training rows"| trainer
    trainer -->|"writes and uploads"| artifacts
    trainer -->|"reports metrics / uploads artifacts"| clearml
    artifacts -->|"model file"| validator
    artifacts -->|"model file"| web
    actor <-->|"HTTP GET /, GET /health,<br/>POST /predict"| web
    clearml <-->|"tracking and pipeline control"| pipeline
```

### 2.2 Component responsibilities

| Component | Responsibility | Inputs | Outputs |
|---|---|---|---|
| `data/train.csv` | Source of labelled training examples | Numeric feature columns and `energy_generated` | Training and validation rows |
| `data/data_registry.py` | Optional dataset lineage helper | CSV path | MD5 hash printed to console |
| `src/train.py` | Fit, evaluate, visualize, and save the model | CSV, parameters, optional ClearML task | Model, metrics, plots, validation predictions, `_SUCCESS` |
| `src/validation.py` | Load a model and verify sample inference | Model path | Pass/fail result and sample prediction |
| `src/clearml_pipeline.py` | Orchestrate training then validation | Dataset/artifact paths and training parameters | ClearML pipeline and component tasks |
| `src/serve.py` | Serve browser UI, health, and prediction routes | HTTP requests and fitted model | HTML or JSON response |
| Docker Compose | Run local training, validation, and serve containers | Compose configuration, mounted data/artifacts | Local running services |
| ClearML | Store run tracking and pipeline metadata | Credentials, scalars, uploaded artifacts | Web UI for task/pipeline inspection |

## 3. UML deployment diagram

This view shows the local Docker Compose deployment and the alternative remote
ClearML Agent deployment. Remote paths must exist or be mounted on the agent;
the local host paths in Docker Compose are not automatically shared with remote
workers.

```mermaid
flowchart TB
    subgraph host["«device» Developer machine"]
        subgraph compose["«executionEnvironment» Docker Compose"]
            trainContainer["«container» training-agent<br/>python src/train.py"]
            validationContainer["«container» validation-agent<br/>wait for _SUCCESS;<br/>python src/validation.py"]
            serveContainer["«container» serve<br/>python src/serve.py<br/>port 5000"]
        end
        csv[("data/train.csv")]
        artifactVolume[("artifacts/ bind mount<br/>/app/artifacts")]
        hostClient["Browser / API client"]
    end

    subgraph clearmlCloud["«external system» ClearML Server"]
        api["API and Files services"]
        ui["Web UI<br/>tasks, pipeline, metrics, artifacts"]
        pipelineController["Pipeline controller<br/>services queue or configured queue"]
        subgraph agentHost["«executionEnvironment» ClearML Agent host"]
            agent["ClearML Agent<br/>configured execution queue"]
            componentTasks["Training and validation<br/>component tasks"]
            remoteData[("Accessible code, data,<br/>and artifact storage")]
        end
    end

    csv -->|"bind mount /app/data"| trainContainer
    trainContainer -->|"write"| artifactVolume
    artifactVolume -->|"bind mount /app/artifacts"| validationContainer
    artifactVolume -->|"bind mount /app/artifacts"| serveContainer
    hostClient <-->|"HTTP :5000"| serveContainer
    trainContainer <-->|"ClearML API / artifact upload"| api
    api <--> ui
    pipelineController -->|"queue tasks"| agent
    agent --> componentTasks
    componentTasks <--> remoteData
    componentTasks <-->|"ClearML API / artifact upload"| api
```

## 4. Data flow

### 4.1 Training and prediction data-flow diagram

```mermaid
flowchart LR
    raw[("train.csv<br/>features + energy_generated")]
    check{"target column<br/>present?"}
    split["train_test_split<br/>test_size = 0.2<br/>random_state = 42"]
    fit["Fit sklearn Pipeline<br/>StandardScaler → RandomForestRegressor"]
    predictVal["Predict validation rows"]
    evaluate["Calculate MAE, MSE,<br/>RMSE, R²"]
    report[("metrics.json")]
    plotMetrics[("metrics_plot.png")]
    plotCompare[("actual_vs_predicted.png")]
    csvPred[("validation_predictions.csv<br/>actual, predicted")]
    plotFeatures[("feature_importances.png")]
    model[("solar_energy_model.joblib")]
    marker[("_SUCCESS")]
    clearml[("ClearML scalars<br/>and task artifacts")]
    valid{"Model load and<br/>smoke inference pass?"}
    serve["Flask serve.py<br/>reorder named fields;<br/>predict"]
    client["Browser / API client"]
    response[("HTML or JSON<br/>predictions")]

    raw --> check
    check -->|"yes"| split
    check -->|"no: fail training"| failure["Stop with ValueError"]
    split --> fit
    fit --> predictVal
    predictVal --> evaluate
    evaluate --> report
    evaluate --> plotMetrics
    evaluate --> plotCompare
    evaluate --> csvPred
    fit --> plotFeatures
    fit --> model
    report --> marker
    model --> valid
    valid -->|"no: fail validation"| failure
    valid -->|"yes"| serve
    model --> serve
    client -->|"POST /predict"| serve
    serve --> response
    report -.->|"report scalar / upload"| clearml
    plotMetrics -.->|"upload"| clearml
    plotCompare -.->|"upload"| clearml
    csvPred -.->|"upload"| clearml
    plotFeatures -.->|"upload"| clearml
    model -.->|"upload"| clearml
```

Training currently creates a 20% validation split. It saves validation
predictions as paired `actual` and `predicted` columns. The serving API returns
predictions to clients; it does not currently log live request predictions to
ClearML.

### 4.2 ClearML pipeline sequence

```mermaid
sequenceDiagram
    autonumber
    actor Developer
    participant Script as clearml_pipeline.py
    participant Server as ClearML API
    participant Controller as Pipeline controller
    participant Queue as Configured agent queue
    participant Train as train_step
    participant Validate as validation_step
    participant Files as Accessible data/artifact storage
    participant UI as ClearML UI

    Developer->>Script: python src/clearml_pipeline.py --remote
    Script->>Server: Create pipeline run in Solar_Energy_Prediction
    Server-->>Controller: Queue controller task
    Controller->>Queue: Request execution
    Queue-->>Controller: Agent accepts controller task
    Controller->>Queue: Queue train_step
    Queue-->>Train: Agent executes training component
    Train->>Files: Read train.csv
    Train->>Train: Fit model and calculate validation metrics
    Train->>Files: Save model, metrics, plots, prediction CSV
    Train->>Server: Report scalars and upload artifacts
    Train-->>Controller: Return model path
    Controller->>Queue: Queue validation_step(model path)
    Queue-->>Validate: Agent executes validation component
    Validate->>Files: Load saved model and run sample inference
    Validate-->>Controller: Return success or fail
    Controller-->>Server: Set pipeline run status
    Server-->>UI: Display steps, logs, scalars, artifacts, status
```

The pipeline controller and components use `CLEARML_EXECUTION_QUEUE`, which
defaults to `default` in `src/clearml_pipeline.py`. A working online ClearML
configuration, an Agent listening on that queue, and shared/available
code/data/artifact paths are required for remote execution.

### 4.3 Local Docker Compose sequence

```mermaid
sequenceDiagram
    autonumber
    actor Developer
    participant Compose as Docker Compose
    participant Trainer as training-agent
    participant Data as /app/data/train.csv
    participant Artifacts as Shared /app/artifacts
    participant Validator as validation-agent
    participant Serve as Flask serve
    participant Client as Browser / API client
    participant ClearML as ClearML API

    Developer->>Compose: docker compose up --build
    Compose->>Trainer: Start python src/train.py
    Trainer->>Data: Read CSV
    Trainer->>Trainer: Fit, predict validation set, compute metrics
    Trainer->>Artifacts: Write model, reports, plots, _SUCCESS
    opt ClearML configured
        Trainer->>ClearML: Report scalars and upload artifacts
    end
    Compose->>Validator: Start after training container starts
    loop Until sentinel exists
        Validator->>Artifacts: Check for _SUCCESS
    end
    Validator->>Artifacts: Load model
    Validator->>Validator: Run sample inference
    Compose->>Serve: Start Flask on port 5000
    Client->>Serve: GET /health or POST /predict
    Serve->>Artifacts: Resolve/load fitted model
    Serve-->>Client: Health status or prediction response
```

In Compose, `depends_on` controls service start order; the validation command
also polls for `/app/artifacts/_SUCCESS`. It does not by itself guarantee that
the training container completed successfully.

## 5. UML class and module diagram

The project is primarily functional and does not define custom domain classes.
This UML-style class diagram therefore distinguishes Python modules/functions
from the concrete scikit-learn and Flask classes they use; module boxes are
stereotyped `«module»` rather than presented as Python classes.

```mermaid
classDiagram
    direction LR

    class train_module {
        <<module>>
        +DEFAULT_PARAMS
        +load_data(path)
        +prepare_features(df)
        +train_model(params, task, data_path, artifact_dir)
        +main()
    }

    class validation_module {
        <<module>>
        +validate_model(model_path, sample_data)
    }

    class pipeline_module {
        <<module>>
        +train_step(data_path, artifact_dir, params)
        +validation_step(model_path)
        +solar_energy_pipeline(data_path, artifact_dir, params)
        +main()
    }

    class serve_module {
        <<module>>
        +resolve_model_path()
        +load_model()
        +get_model_feature_names()
        +home()
        +health()
        +predict()
    }

    class SklearnPipeline {
        <<sklearn>>
        +fit(X, y)
        +predict(X)
    }

    class StandardScaler {
        <<sklearn transformer>>
        +fit_transform(X)
        +transform(X)
    }

    class RandomForestRegressor {
        <<sklearn estimator>>
        +fit(X, y)
        +predict(X)
        +feature_importances_
    }

    class ClearMLTask {
        <<ClearML>>
        +connect(params)
        +get_logger()
        +upload_artifact(name, artifact)
    }

    class FlaskApp {
        <<Flask>>
        +route(path, methods)
        +test_client()
    }

    class ModelArtifact {
        <<joblib file>>
        +solar_energy_model.joblib
    }

    class MetricsArtifact {
        <<JSON file>>
        +metrics.json
    }

    class PredictionArtifact {
        <<CSV file>>
        +validation_predictions.csv
    }

    train_module ..> SklearnPipeline : constructs and fits
    SklearnPipeline *-- StandardScaler : preprocessing step
    SklearnPipeline *-- RandomForestRegressor : estimator step
    train_module ..> ClearMLTask : reports metrics/artifacts
    train_module ..> ModelArtifact : serializes
    train_module ..> MetricsArtifact : writes
    train_module ..> PredictionArtifact : writes
    pipeline_module ..> train_module : invokes train_model
    pipeline_module ..> validation_module : invokes validate_model
    validation_module ..> ModelArtifact : loads and predicts
    serve_module ..> FlaskApp : registers routes
    serve_module ..> ModelArtifact : loads and predicts
```

## 6. UML activity diagram — model lifecycle

```mermaid
flowchart TD
    start((Start))
    read["Read CSV"]
    schema{"energy_generated<br/>column exists?"}
    error["Raise input error"]
    split["Split features and target"]
    train["Fit scaler and random forest"]
    evaluate["Predict validation rows<br/>calculate MAE, MSE, RMSE, R²"]
    save["Save model, metrics,<br/>plots and prediction CSV"]
    sentinel["Write _SUCCESS"]
    validate["Load saved model<br/>run sample inference"]
    pass{"Inference succeeds?"}
    failed["Validation failed"]
    deploy["Serve requests with Flask"]
    finish((Complete))

    start --> read --> schema
    schema -->|"no"| error
    schema -->|"yes"| split --> train --> evaluate --> save --> sentinel --> validate --> pass
    pass -->|"no"| failed
    pass -->|"yes"| deploy --> finish
```

## 7. Artifact and interface contracts

### Training outputs

| Path | Format | Meaning |
|---|---|---|
| `artifacts/solar_energy_model.joblib` | Joblib | Fitted scaler/regressor pipeline |
| `artifacts/metrics.json` | JSON | MAE, MSE, RMSE, R², and parameters |
| `artifacts/metrics_plot.png` | PNG | Validation metric chart |
| `artifacts/actual_vs_predicted.png` | PNG | Actual-vs-predicted validation scatter plot |
| `artifacts/validation_predictions.csv` | CSV | `actual` and `predicted` validation values |
| `artifacts/feature_importances.png` | PNG | Random forest feature-importance chart |
| `artifacts/_SUCCESS` | Text sentinel | Indicates training reached output completion |

### HTTP interface

| Method and path | Purpose | Successful response |
|---|---|---|
| `GET /` | Browser prediction form | HTML |
| `GET /health` | Model readiness check | JSON status |
| `POST /predict` | Predict named feature records | `{"predictions":[...]}` |

The prediction input must provide the model's feature names. The serving code
uses the fitted pipeline's feature order when available. Keep feature schema
changes aligned between the CSV, training process, and request payloads.

## 8. Deployment, reliability, and security

### Current deployment

- Docker Compose is intended for local development and single-host experiments.
- `./data` is mounted at `/app/data`; `./artifacts` is mounted at
  `/app/artifacts`.
- The Flask development server listens on port `5000`.
- ClearML remote pipeline execution is an alternative orchestration route and
  requires an Agent plus storage paths accessible to that Agent.

### Reliability considerations

- A missing dataset or target column stops training.
- A missing model or failed inference causes model validation to fail.
- Compose validation waits for `_SUCCESS`; inspect training logs if it never
  appears.
- A remote agent cannot use a developer's local paths unless those paths are
  mounted or data is transferred through shared/remote storage.
- Use health checks, task timeouts, retries, and artifact versioning when
  evolving this workflow for production.

### Security and production notes

- Store ClearML credentials in environment-backed secrets, not source files,
  documentation, or container images.
- Do not expose the prediction service publicly without authentication,
  request-size limits, and input validation.
- Flask's built-in development server is not production-grade; use a production
  WSGI server behind an appropriately configured reverse proxy.
- Restrict artifact access and avoid loading untrusted serialized model files.

## 9. Extension points

- Add schema validation and dataset quality checks before training.
- Register promoted models and use versioned object storage for artifacts.
- Add prediction logging with privacy-aware retention if operational inference
  monitoring is required.
- Add automated tests for input validation, model loading, and API behavior.
- Add ClearML hyperparameter optimization around the existing training
  parameters.

## 10. Related files

- [README.md](README.md)
- [docker-compose.yml](docker-compose.yml)
- [Dockerfile](Dockerfile)
- [src/train.py](src/train.py)
- [src/validation.py](src/validation.py)
- [src/clearml_pipeline.py](src/clearml_pipeline.py)
- [src/serve.py](src/serve.py)
- [data/data_registry.py](data/data_registry.py)
