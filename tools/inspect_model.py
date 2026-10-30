"""Inspect this trusted project model under its matching dependencies."""
from pathlib import Path
from src.model import load_bundle
m=load_bundle(Path(__file__).resolve().parents[1]/'models'/'psi_weather_xgboost.joblib')
print('Target:',m['target_column'])
print('Forecast horizon:',m['forecast_horizon_hours'])
print('Numeric feature count:',len(m['numeric_features']))
print('Categorical feature count:',len(m['categorical_features']))
print('Feature names:')
for x in m['feature_columns']:print(' -',x)
