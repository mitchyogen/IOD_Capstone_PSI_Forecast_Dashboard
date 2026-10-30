"""Load only a trusted local project artifact, verify exact training dependencies."""
from pathlib import Path
import importlib.metadata
import joblib
import numpy as np

class ModelError(RuntimeError):pass

EXPECTED_SKLEARN='1.9.1'
EXPECTED_XGBOOST='3.2.0'


def check_versions():
    for name, expected in [('scikit-learn',EXPECTED_SKLEARN),('xgboost',EXPECTED_XGBOOST)]:
        actual=importlib.metadata.version(name)
        if actual!=expected:
            raise ModelError(f'{name} {actual} is installed; saved artifact requires {expected}')


def load_bundle(path: str | Path):
    check_versions()
    bundle=joblib.load(path)  # Security: only load this trusted bundled artifact.
    required={'pipeline','numeric_features','categorical_features','feature_columns','target_column',
              'forecast_horizon_hours','sklearn_version','xgboost_version'}
    if not isinstance(bundle,dict) or not required.issubset(bundle):
        raise ModelError('Unexpected model artifact schema')
    if str(bundle['sklearn_version'])!=EXPECTED_SKLEARN or str(bundle['xgboost_version'])!=EXPECTED_XGBOOST:
        raise ModelError('Saved model metadata does not match configured versions')
    if bundle['target_column']!='psi_3h_ahead' or bundle['forecast_horizon_hours']!=3:
        raise ModelError('Artifact does not forecast 3-hour PSI')
    features=list(bundle['feature_columns'])
    if len(features)!=61 or len(features)!=len(set(features)):
        raise ModelError('Expected 61 unique input features')
    if features!=list(bundle['numeric_features'])+list(bundle['categorical_features']):
        raise ModelError('Saved feature order does not match model preprocessing')
    return bundle


def predict(bundle,row):
    cols=list(bundle['feature_columns'])
    if list(row.columns)!=cols:
        raise ModelError('Prediction input has missing, additional or incorrectly ordered columns')
    # Missing lagged values are handled by the model's fitted imputer.
    # Missing current observations are rejected upstream in features_for_region.

    raw=bundle['pipeline'].predict(row)
    pred=float(np.asarray(raw).reshape(-1)[0])
    if not np.isfinite(pred):raise ModelError('Prediction was not finite')
    return pred
