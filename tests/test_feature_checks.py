import pytest
import pandas as pd
from src.features import features_for_region
from src.nea_client import NEAError

def test_missing_history_fails_closed():
    df=pd.DataFrame([{'timestamp':pd.Timestamp('2026-01-01T10:00:00+08:00'),'region':'central',
                      'psi_twenty_four_hourly':50}])
    with pytest.raises(NEAError):
        features_for_region(df,pd.DataFrame(),{'central':(1.30,103.80)},'central',
                            pd.Timestamp('2026-01-01T10:00:00+08:00'),['psi_twenty_four_hourly'])
