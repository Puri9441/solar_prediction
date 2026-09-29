from flask import Flask, request, jsonify, render_template_string
import joblib
import os
import pandas as pd

DEFAULT_FEATURE_NAMES = [
    "temperature",
    "humidity",
    "solar_irradiance",
    "wind_speed",
    "cloud_cover",
]

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join("/app/artifacts", "solar_energy_model.joblib"),
)

app = Flask(__name__)
model = None


def resolve_model_path():
    candidates = []
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        os.path.join("/app/artifacts", "solar_energy_model.joblib"),
        os.path.abspath(os.path.join(os.getcwd(), "artifacts", "solar_energy_model.joblib")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "solar_energy_model.joblib")),
    ])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    return candidates[0] if candidates else os.path.join("artifacts", "solar_energy_model.joblib")


def get_model_feature_names():
    if model is None:
        return DEFAULT_FEATURE_NAMES

    try:
        scaler = getattr(model.named_steps, 'scaler', None)
        if scaler is None:
            scaler = model.named_steps.get('scaler') if hasattr(model, 'named_steps') else None
        feature_names = getattr(scaler, 'feature_names_in_', None)
        if feature_names is not None:
            return list(feature_names)
    except Exception:
        pass

    return DEFAULT_FEATURE_NAMES


def load_model():
    global model, MODEL_PATH
    resolved_path = resolve_model_path()
    MODEL_PATH = resolved_path
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None
    else:
        model = None


load_model()


HTML_PREDICT_FORM = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Solar Energy Prediction</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; }
      label { display: block; margin: 0.75rem 0 0.25rem; }
      input[type=text] { width: 100%; max-width: 360px; padding: 0.4rem; }
      textarea { width: 100%; max-width: 640px; height: 120px; padding: 0.4rem; }
      button { margin-top: 1rem; padding: 0.5rem 1rem; }
      .result { margin-top: 1.25rem; padding: 1rem; border: 1px solid #ccc; background: #f9f9f9; }
    </style>
  </head>
  <body>
    <h1>Solar Energy Prediction</h1>
    <p>Enter values and submit the form, or send JSON to <code>/predict</code>.</p>
    <form id="predict-form" method="post" action="/predict">
      {% for feature in features %}
        <label for="{{ feature }}">{{ feature.replace('_', ' ').title() }}</label>
        <input type="text" id="{{ feature }}" name="{{ feature }}" value="" required />
      {% endfor %}
      <button type="submit">Predict</button>
    </form>

    <h2>Browser JSON Demo</h2>
    <p>Click the button to send the same request as JSON.</p>
    <button id="json-predict">Send JSON request</button>
    <div id="json-result" class="result" style="display:none;"></div>

    <h2>JSON API Example</h2>
    <textarea readonly>{
  "instances": [
    {
      "temperature": 25.0,
      "humidity": 45.0,
      "wind_speed": 5.0,
      "cloud_cover": 20.0,
      "solar_irradiance": 650.0
    }
  ]
}</textarea>

    {% if prediction is defined and prediction is not none %}
      <div class="result">
        <strong>Prediction result:</strong>
        <pre>{{ prediction }}</pre>
      </div>
    {% endif %}

    <script>
      document.getElementById('json-predict').addEventListener('click', async function () {
        const payload = {
          instances: [
            {
              temperature: 25.0,
              humidity: 45.0,
              wind_speed: 5.0,
              cloud_cover: 20.0,
              solar_irradiance: 650.0,
            },
          ],
        };

        const resultDiv = document.getElementById('json-result');
        resultDiv.style.display = 'block';
        resultDiv.textContent = 'Sending request...';

        try {
          const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          const data = await response.json();
          resultDiv.textContent = JSON.stringify(data, null, 2);
        } catch (err) {
          resultDiv.textContent = 'Request failed: ' + err;
        }
      });
    </script>
  </body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PREDICT_FORM, features=get_model_feature_names())


@app.route("/health", methods=["GET"])
def health():
    load_model()
    return jsonify({"status": "ok" if model is not None else "model_missing"})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    feature_names = get_model_feature_names()

    if request.method == "GET":
        return render_template_string(HTML_PREDICT_FORM, features=feature_names)

    load_model()
    if model is None:
        return jsonify({"error": "model not loaded"}), 500

    payload = None
    if request.is_json:
        payload = request.get_json(force=True)
    elif request.form:
        record = {}
        for feature in feature_names:
            if feature in request.form:
                try:
                    record[feature] = float(request.form[feature])
                except ValueError:
                    return jsonify({"error": f"invalid value for {feature}"}), 400
        payload = {"instances": [record]}
    else:
        return jsonify({"error": "unsupported payload format"}), 400

    # Accept either a single record dict or a list of records or a feature array
    if isinstance(payload, dict):
        if "instances" in payload:
            records = payload["instances"]
        elif "instance" in payload:
            records = [payload["instance"]]
        elif "features" in payload and isinstance(payload["features"], list):
            records = [payload["features"]]
        else:
            # assume it's a single record mapping
            records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        return jsonify({"error": "unsupported payload format"}), 400

    # Normalize to DataFrame where possible and preserve feature order
    try:
        if len(records) == 0:
            return jsonify({"predictions": []})
        if isinstance(records[0], list):
            X = pd.DataFrame(records, columns=feature_names)
        else:
            X = pd.DataFrame.from_records(records)
            missing = [f for f in feature_names if f not in X.columns]
            if missing:
                return jsonify({"error": f"missing required features: {missing}"}), 400
            X = X[feature_names]
    except Exception as e:
        return jsonify({"error": f"failed to parse input: {e}"}), 400

    try:
        preds = model.predict(X)
        if request.is_json:
            return jsonify({"predictions": preds.tolist()})
        return render_template_string(
            HTML_PREDICT_FORM + '<div class="result"><strong>Prediction:</strong> {{ prediction }}</div>',
            features=feature_names,
            prediction=preds.tolist(),
        )
    except Exception as e:
        return jsonify({"error": f"prediction failed: {e}"}), 500


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
