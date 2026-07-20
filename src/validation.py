import os
import pandas as pd
from joblib import load


def validate_model(model_path, sample_data=None):
    print(f'Validating model at: {model_path}...')

    if not os.path.exists(model_path):
        print('Validation Failed: Model file not found.')
        return False

    try:
        model = load(model_path)

        if sample_data is None:
            feature_names = getattr(model.named_steps['scaler'], 'feature_names_in_', None)
            if feature_names is None:
                raise ValueError('Unable to infer feature names from model pipeline.')
            sample_data = pd.DataFrame([dict.fromkeys(feature_names, 0.0)])

        predictions = model.predict(sample_data)
        print('Validation Passed: Model loaded and inference succeeded.')
        print(f'Sample prediction: {predictions[0]:.4f}')
        return True

    except Exception as e:
        print(f'Validation Failed: {e}')
        return False


if __name__ == '__main__':
    MODEL_PATH = 'artifacts/solar_energy_model.joblib'
    validate_model(MODEL_PATH)
