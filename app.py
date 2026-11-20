import calendar
from datetime import date
from pathlib import Path
import json
import math
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "solar_energy_model.joblib"
TRAIN_DATA_PATH = ROOT / "data" / "train.csv"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

DEFAULT_FEATURES = [
    "temperature",
    "humidity",
    "solar_irradiance",
    "wind_speed",
    "cloud_cover",
]

FEATURE_CONFIG = {
    "temperature": {"label": "Temperature", "unit": "°C", "icon": "🌡️"},
    "humidity": {"label": "Humidity", "unit": "%", "icon": "💧"},
    "solar_irradiance": {"label": "Solar irradiance", "unit": "W/m²", "icon": "☀️"},
    "wind_speed": {"label": "Wind speed", "unit": "m/s", "icon": "💨"},
    "cloud_cover": {"label": "Cloud cover", "unit": "%", "icon": "☁️"},
}


st.set_page_config(
    page_title="Solar Energy Predictor",
    page_icon="☀️",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_data():
    if not TRAIN_DATA_PATH.exists():
        return None
    return pd.read_csv(TRAIN_DATA_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(ttl=24 * 60 * 60)
def fetch_solar_history(latitude, longitude, history_years):
    end_year = date.today().year - 1
    start_year = end_year - history_years + 1
    query = urlencode(
        {
            "parameters": "ALLSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": longitude,
            "latitude": latitude,
            "start": start_year,
            "end": end_year,
            "format": "JSON",
        }
    )
    url = f"https://power.larc.nasa.gov/api/temporal/monthly/point?{query}"
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    monthly_values = payload["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    records = []
    for year in range(start_year, end_year + 1):
        annual_irradiation = 0.0
        for month in range(1, 13):
            value = monthly_values.get(f"{year}{month:02d}")
            if value is None or value <= -900:
                raise ValueError(
                    f"Solar data is incomplete for {year}; "
                    "try another location or year range."
                )
            annual_irradiation += float(value) * calendar.monthrange(year, month)[1]
        records.append(
            {
                "Year": year,
                "Solar irradiation (kWh/m²/year)": annual_irradiation,
            }
        )

    if not records:
        raise ValueError("No solar history was returned for these coordinates.")
    return pd.DataFrame(records).set_index("Year")


def get_feature_names(model):
    try:
        scaler = model.named_steps.get("scaler")
        names = getattr(scaler, "feature_names_in_", None)
        if names is not None:
            return list(names)
    except Exception:
        pass
    return DEFAULT_FEATURES


model = load_model()
training_data = load_training_data()
metrics = load_metrics()
feature_names = get_feature_names(model) if model is not None else DEFAULT_FEATURES


st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2rem; }
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #f8fafc; }
    .hero {
        background: linear-gradient(135deg, #ffb703 0%, #fb8500 100%);
        border-radius: 20px;
        padding: 28px 32px;
        color: #172554;
        margin-bottom: 18px;
    }
    .hero h1 { margin: 0 0 8px 0; color: #172554; }
    .hero p { margin: 0; font-size: 1.05rem; }
    div[data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 16px;
        color: #f8fafc;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #cbd5e1;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f8fafc;
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 8px;
    }
    div[data-testid="stAlert"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>☀️ Solar Energy Predictor</h1>
      <p>Estimate solar energy generation from current weather and environmental conditions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## System status")
    if model is not None:
        st.success("Model loaded")
    else:
        st.error("Model not found")

    st.markdown("### Model")
    st.caption("Random Forest + Standard Scaler")

    if metrics:
        metric_values = metrics["metrics"]
        st.markdown("### Validation")
        st.write(f"R²  **{metric_values['r2']:.3f}**")
        st.write(f"MAE **{metric_values['mae']:.3f}**")
        st.write(f"RMSE **{metric_values['rmse']:.3f}**")

    st.markdown("---")
    st.caption("Training data and model artifacts are loaded locally.")

tab_proposal, tab_predict, tab_batch, tab_insights = st.tabs(
    ["🏠 Customer proposal", "⚡ Predict", "📄 Batch prediction", "📊 Insights"]
)

with tab_proposal:
    st.subheader("Customer solar proposal")
    st.warning(
        "Enter the customer's site coordinates to retrieve historical solar "
        "irradiance from NASA POWER. Generation is estimated from this resource "
        "data; it is not measured panel output or a roof survey."
    )

    with st.form("customer_proposal_form"):
        proposal_left, proposal_right = st.columns(2)
        with proposal_left:
            site_location = st.text_input(
                "Customer / site name",
                value="Customer site",
            )
            latitude_text = st.text_input(
                "Latitude",
                placeholder="e.g. 19.076000",
                help="Decimal degrees, from -90 to 90.",
            )
            longitude_text = st.text_input(
                "Longitude",
                placeholder="e.g. 72.877700",
                help="Decimal degrees, from -180 to 180.",
            )
            monthly_bill = st.number_input(
                "Average monthly electricity bill (₹)",
                min_value=1.0,
                value=3000.0,
                step=500.0,
            )
        with proposal_right:
            electricity_rate = st.number_input(
                "Electricity rate (₹ per kWh)",
                min_value=0.1,
                value=8.0,
                step=0.5,
            )
            history_years = st.selectbox(
                "History period",
                options=[3, 4, 5],
                index=2,
                format_func=lambda value: f"{value} years",
            )
        proposal_submitted = st.form_submit_button(
            "Build customer proposal",
            type="primary",
            use_container_width=True,
        )

    if proposal_submitted:
        if not site_location.strip():
            st.error("Enter a customer or site name.")
        else:
            solar_history = None
            try:
                latitude = float(latitude_text)
                longitude = float(longitude_text)
                if not math.isfinite(latitude) or not math.isfinite(longitude):
                    raise ValueError("Coordinates must be finite numbers.")
                if not -90 <= latitude <= 90:
                    raise ValueError("Latitude must be between -90 and 90.")
                if not -180 <= longitude <= 180:
                    raise ValueError("Longitude must be between -180 and 180.")
            except ValueError as error:
                st.error(f"Enter valid coordinates: {error}")
            else:
                try:
                    solar_history = fetch_solar_history(
                        latitude,
                        longitude,
                        history_years,
                    )
                except (
                    URLError,
                    OSError,
                    ValueError,
                    UnicodeDecodeError,
                    KeyError,
                ) as error:
                    st.error(f"Could not retrieve NASA POWER solar data: {error}")
                    solar_history = None

            if solar_history is not None:
                performance_ratio = 0.80
                solar_history["Estimated yield (kWh/kW)"] = (
                    solar_history["Solar irradiation (kWh/m²/year)"]
                    * performance_ratio
                )
                average_yield = float(
                    solar_history["Estimated yield (kWh/kW)"].mean()
                )
                annual_consumption = monthly_bill / electricity_rate * 12
                required_capacity = annual_consumption / average_yield
                system_sizes = [
                    1,
                    2,
                    3,
                    4,
                    5,
                    7.5,
                    10,
                    12,
                    15,
                    20,
                    25,
                    30,
                    40,
                    50,
                ]
                recommended_capacity = min(
                    system_sizes,
                    key=lambda size: abs(size * average_yield - annual_consumption),
                )
                solar_history["Estimated generation (kWh)"] = (
                    solar_history["Estimated yield (kWh/kW)"]
                    * recommended_capacity
                )

                capacity_comparison = pd.DataFrame(
                    {
                        "System size (kW)": system_sizes,
                        "Estimated annual generation (kWh)": [
                            size * average_yield for size in system_sizes
                        ],
                        "Estimated annual use covered (%)": [
                            size * average_yield / annual_consumption * 100
                            for size in system_sizes
                        ],
                    }
                )

                st.markdown(f"### Proposal for {site_location.strip()}")
                st.caption(
                    f"Coordinates: {latitude:.6f}, {longitude:.6f} "
                    f"· NASA POWER · {solar_history.index.min()}–"
                    f"{solar_history.index.max()}"
                )
                result_columns = st.columns(3)
                result_columns[0].metric(
                    "Recommended system",
                    f"{recommended_capacity:g} kW",
                )
                result_columns[1].metric(
                    "Estimated annual use",
                    f"{annual_consumption:,.0f} kWh",
                )
                result_columns[2].metric(
                    "Estimated system needed",
                    f"{required_capacity:.1f} kW",
                )

                st.caption(
                    f"Estimated yield averages {average_yield:,.0f} kWh per kW "
                    f"per year, using a {performance_ratio:.0%} performance "
                    "ratio. This is a preliminary estimate; roof orientation, "
                    "shading, equipment, and local regulations require a site survey."
                )

                st.markdown(
                    f"#### {history_years}-year estimated generation potential "
                    f"for a {recommended_capacity:g} kW system"
                )
                st.line_chart(
                    solar_history,
                    y="Estimated generation (kWh)",
                )
                st.dataframe(solar_history, use_container_width=True)

                st.markdown("#### Compare system sizes")
                st.dataframe(
                    capacity_comparison.style.format(
                        {
                            "System size (kW)": "{:g}",
                            "Estimated annual generation (kWh)": "{:,.0f}",
                            "Estimated annual use covered (%)": "{:.0f}%",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                st.download_button(
                    "Download site solar history",
                    solar_history.reset_index().to_csv(index=False).encode("utf-8"),
                    file_name="site_solar_history_estimate.csv",
                    mime="text/csv",
                )

with tab_predict:
    left, right = st.columns([1.35, 0.65], gap="large")

    with left:
        st.subheader("Weather conditions")

        data_ranges = {}
        if training_data is not None:
            for name in feature_names:
                if name in training_data.columns:
                    data_ranges[name] = (
                        float(training_data[name].min()),
                        float(training_data[name].max()),
                        float(training_data[name].mean()),
                    )

        inputs = {}
        columns = st.columns(2)
        for index, name in enumerate(feature_names):
            config = FEATURE_CONFIG.get(
                name,
                {"label": name.replace("_", " ").title(), "unit": "", "icon": "•"},
            )
            minimum, maximum, average = data_ranges.get(name, (0.0, 100.0, 50.0))
            if maximum <= minimum:
                maximum = minimum + 1.0

            with columns[index % 2]:
                label = f"{config['icon']} {config['label']}"
                step = 0.1 if maximum - minimum < 100 else 1.0
                inputs[name] = st.slider(
                    label,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(average),
                    step=step,
                    help=f"{minimum:.1f} to {maximum:.1f} {config['unit']} in training data",
                )

        predict_clicked = st.button(
            "Generate prediction",
            type="primary",
            use_container_width=True,
        )

    with right:
        st.subheader("Prediction")
        result_placeholder = st.container()

        if predict_clicked:
            if model is None:
                st.error("Train the model first: `python src/train.py`")
            else:
                input_frame = pd.DataFrame([inputs], columns=feature_names)
                prediction = float(model.predict(input_frame)[0])
                with result_placeholder:
                    st.metric(
                        "Estimated energy generated",
                        f"{prediction:.3f}",
                        help="Model output using the entered conditions.",
                    )
                    st.progress(
                        min(max(prediction / 10.0, 0.0), 1.0),
                        text="Relative output level",
                    )

                    st.markdown("### Input summary")
                    summary_rows = []
                    for name, value in inputs.items():
                        config = FEATURE_CONFIG.get(
                            name,
                            {"label": name.replace("_", " ").title(), "unit": ""},
                        )
                        unit = f" {config['unit']}" if config["unit"] else ""
                        summary_rows.append(
                            f"| {config['label']} | {value:.1f}{unit} |"
                        )
                    st.markdown(
                        "| Condition | Value |\n|---|---:|\n"
                        + "\n".join(summary_rows)
                    )
        else:
            with result_placeholder:
                st.info("Adjust the conditions, then generate a prediction.")

with tab_batch:
    st.subheader("Predict multiple records")
    uploaded_file = st.file_uploader(
        "Upload a CSV",
        type=["csv"],
        help=f"CSV must contain: {', '.join(feature_names)}",
    )

    if uploaded_file is not None:
        try:
            batch = pd.read_csv(uploaded_file)
            missing = [name for name in feature_names if name not in batch.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            elif model is None:
                st.error("The model is not available.")
            else:
                predictions = model.predict(batch[feature_names])
                output = batch.copy()
                output["predicted_energy_generated"] = predictions
                st.success(f"Predicted {len(output)} record(s)")
                st.dataframe(output, use_container_width=True)
                st.download_button(
                    "Download predictions",
                    output.to_csv(index=False).encode("utf-8"),
                    file_name="solar_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as error:
            st.error(f"Could not process the CSV: {error}")
    else:
        template = pd.DataFrame(
            [{name: 0.0 for name in feature_names}]
        )
        st.download_button(
            "Download CSV template",
            template.to_csv(index=False).encode("utf-8"),
            file_name="solar_prediction_template.csv",
            mime="text/csv",
        )

with tab_insights:
    st.subheader("Model and dataset insights")

    insight_left, insight_right = st.columns(2)
    with insight_left:
        st.markdown("### Validation metrics")
        if metrics:
            values = metrics["metrics"]
            metric_columns = st.columns(2)
            metric_columns[0].metric("R²", f"{values['r2']:.4f}")
            metric_columns[1].metric("MAE", f"{values['mae']:.4f}")
            metric_columns = st.columns(2)
            metric_columns[0].metric("MSE", f"{values['mse']:.4f}")
            metric_columns[1].metric("RMSE", f"{values['rmse']:.4f}")
        else:
            st.info("Run training to create metrics.json.")

    with insight_right:
        st.markdown("### Training dataset")
        if training_data is not None:
            st.metric("Rows", f"{len(training_data):,}")
            st.metric("Features", len(feature_names))
            st.dataframe(
                training_data[feature_names].describe().T,
                use_container_width=True,
            )
        else:
            st.info("Training data was not found.")

    charts = [
        ("Feature importance", "artifacts/feature_importances.png"),
        ("Actual vs. predicted", "artifacts/actual_vs_predicted.png"),
        ("Validation metrics", "artifacts/metrics_plot.png"),
    ]
    available_charts = [
        (title, ROOT / relative)
        for title, relative in charts
        if (ROOT / relative).exists()
    ]
    if available_charts:
        chart_columns = st.columns(2)
        for index, (title, path) in enumerate(available_charts):
            with chart_columns[index % 2]:
                st.markdown(f"### {title}")
                st.image(str(path), use_container_width=True)
