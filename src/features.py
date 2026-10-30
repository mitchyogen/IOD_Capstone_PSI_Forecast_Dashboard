"""Feature reconstruction for the saved 3-hour PSI forecasting pipeline.

Training notebook parity should be confirmed on a known historical row before enabling
live deployment. All timestamps are Asia/Singapore and all lags are true hourly lags.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .nea_client import NEAError

REGIONS = ('central','north','south','east','west')
POLLUTANTS = ['pm10_twenty_four_hourly','pm25_twenty_four_hourly','o3_eight_hour_max',
              'co_eight_hour_max','so2_twenty_four_hourly','no2_one_hour_max','psi_twenty_four_hourly']
WX = {'wind_speed':'weather_wind_speed','temperature':'weather_temperature',
      'rainfall':'weather_rainfall','humidity':'weather_humidity'}
WX_COLS = list(WX.values()) + ['weather_wind_sin','weather_wind_cos']


def geo_distance(lat1,lon1,lat2,lon2):
    p1,p2=np.radians(lat1),np.radians(lat2)
    a=np.sin((p2-p1)/2)**2+np.cos(p1)*np.cos(p2)*np.sin(np.radians(lon2-lon1)/2)**2
    return 2*6371*np.arcsin(np.sqrt(np.clip(a,0,1)))


def aggregate_weather(weather:pd.DataFrame, region_coords:dict) -> pd.DataFrame:
    if weather.empty:raise NEAError('No station weather observations were returned')
    station_coords=weather.groupby('station_id')[['latitude','longitude']].median()
    mapping={}
    for sid,row in station_coords.iterrows():
        near=min(region_coords,key=lambda reg:geo_distance(row.latitude,row.longitude,*region_coords[reg]))
        mapping[sid]=near
    df=weather.copy();df['region']=df.station_id.map(mapping)
    df['observation_time']=df['timestamp']
    # Match notebook: an observation within hour h is used at hour h+1.
    df['timestamp']=df.timestamp.dt.floor('h')+pd.Timedelta(hours=1)
    # Do not average direction degrees (e.g. 359 and 1); use components.
    directions=df.variable.eq('wind_direction')
    df['weather_wind_sin']=np.where(directions,np.sin(np.deg2rad(df.value)),np.nan)
    df['weather_wind_cos']=np.where(directions,np.cos(np.deg2rad(df.value)),np.nan)
    # To match the notebook's fallback rainfall semantics, use the last observation
    # for each station/hour. No summation is performed without interval verification.
    station_hour=(df.sort_values('observation_time')
                  .groupby(['region','station_id','timestamp','variable'],as_index=False)
                  .agg(value=('value','last'),weather_wind_sin=('weather_wind_sin','mean'),
                       weather_wind_cos=('weather_wind_cos','mean')))
    region_hour=station_hour.groupby(['region','timestamp','variable'],as_index=False)[
        ['value','weather_wind_sin','weather_wind_cos']].mean()
    vals=region_hour.pivot_table(index=['region','timestamp'],columns='variable',values='value',aggfunc='mean')
    vals=vals.rename(columns=WX)
    direction=(region_hour[region_hour.variable.eq('wind_direction')]
               .set_index(['region','timestamp'])[['weather_wind_sin','weather_wind_cos']])
    return vals.join(direction).reset_index()


def hourly_psi(psi:pd.DataFrame) -> pd.DataFrame:
    # Original PSI timestamps can be every 15 min. Keep the last available
    # measurement within each hour and do not interpolate absent hours.
    df=psi.copy();df['timestamp']=df.timestamp.dt.floor('h')
    return (df.sort_values('timestamp').groupby(['region','timestamp'],as_index=False).last())


def _region_grid(data:pd.DataFrame,region:str) -> pd.DataFrame:
    if data.empty or not {'region','timestamp'}.issubset(data.columns):
        raise NEAError(f'No regional hourly observations for {region}')
    g=data[data.region.eq(region)].set_index('timestamp').sort_index()
    if g.empty:raise NEAError(f'No readings for {region}')
    if g.index.has_duplicates:g=g[~g.index.duplicated(keep='last')]
    idx=pd.date_range(g.index.min().tz_convert('Asia/Singapore'),g.index.max().tz_convert('Asia/Singapore'),freq='h')
    g=g.reindex(idx);g['region']=region;g.index.name='timestamp'
    return g.reset_index()


def features_for_region(psi:pd.DataFrame, weather_hourly:pd.DataFrame, coords:dict,
                        region:str,as_of:pd.Timestamp,expected:list[str]) -> tuple[pd.DataFrame,dict]:
    if region not in REGIONS:raise NEAError('Unknown PSI region')
    if region not in coords:raise NEAError(f'No PSI region coordinates for {region}')
    cutoff=pd.Timestamp(as_of)
    if cutoff.tzinfo is None:cutoff=cutoff.tz_localize('Asia/Singapore')
    cutoff=cutoff.tz_convert('Asia/Singapore').floor('h')
    p=_region_grid(hourly_psi(psi),region)
    w=_region_grid(weather_hourly,region)
    frame=p.merge(w[['timestamp']+[c for c in WX_COLS if c in w.columns]],on='timestamp',how='left',validate='one_to_one')
    frame=frame[frame.timestamp.le(cutoff)].sort_values('timestamp').copy()
    frame=frame.set_index('timestamp').reindex(pd.date_range(cutoff-pd.Timedelta(hours=26),cutoff,freq='h',tz='Asia/Singapore'))
    frame.index.name='timestamp';frame=frame.reset_index();frame['region']=region
    for col in POLLUTANTS:
        if col not in frame:raise NEAError(f'PSI API missing model feature: {col}')
        frame[col]=pd.to_numeric(frame[col],errors='coerce')
    for col in WX_COLS:
        if col not in frame:raise NEAError(f'Weather API missing model feature: {col}')
    # Require complete current measurements and essential PSI lag history.
    latest=frame.iloc[-1]
    if latest[POLLUTANTS+WX_COLS].isna().any():
        missing=latest[POLLUTANTS+WX_COLS][latest[POLLUTANTS+WX_COLS].isna()].index.tolist()
        raise NEAError(f'Incomplete current-hour input for {region}: {", ".join(missing)}')
    if frame.psi_twenty_four_hourly.iloc[-25:-1].isna().any():
        raise NEAError(f'Incomplete 24-hour PSI history for {region}')
    if frame.pm25_twenty_four_hourly.iloc[-7:-1].isna().any():
        raise NEAError(f'Incomplete PM2.5 history for {region}')
    # Training uses an imputer for lagged weather features. Requiring a perfect
    # seven-hour weather series is stricter than the training notebook and
    # unnecessarily prevents live predictions. Current weather remains mandatory.
    # Missing lag/rolling values may be passed to the fitted training imputer.

    # Predictor naming follows the metadata stored in joblib.
    for lag in [1,2,3,6,12,24]:frame[f'psi_lag_{lag}h']=frame.psi_twenty_four_hourly.shift(lag)
    for lag in [1,3,6]:frame[f'pm25_lag_{lag}h']=frame.pm25_twenty_four_hourly.shift(lag)
    frame['psi_rolling_mean_3h']=frame.psi_twenty_four_hourly.rolling(3,min_periods=3).mean()
    frame['psi_rolling_mean_6h']=frame.psi_twenty_four_hourly.rolling(6,min_periods=6).mean()
    frame['psi_rolling_std_6h']=frame.psi_twenty_four_hourly.rolling(6,min_periods=6).std()
    frame['pm25_rolling_mean_3h']=frame.pm25_twenty_four_hourly.rolling(3,min_periods=3).mean()
    frame['pm25_rolling_mean_6h']=frame.pm25_twenty_four_hourly.rolling(6,min_periods=6).mean()
    frame['psi_change_3h']=frame.psi_twenty_four_hourly-frame.psi_twenty_four_hourly.shift(3)
    frame['pm25_change_3h']=frame.pm25_twenty_four_hourly-frame.pm25_twenty_four_hourly.shift(3)
    for col in WX_COLS:
        for lag in [1,3,6]:frame[f'{col}_lag_{lag}h']=frame[col].shift(lag)
        frame[f'{col}_rolling_6h']=frame[col].shift(1).rolling(6,min_periods=3).mean()
    frame['location_latitude']=coords[region][0];frame['location_longitude']=coords[region][1]
    t=frame.timestamp.dt
    frame['hour_sin']=np.sin(2*np.pi*t.hour/24)
    frame['hour_cos']=np.cos(2*np.pi*t.hour/24)
    frame['month_sin']=np.sin(2*np.pi*t.month/12)
    frame['month_cos']=np.cos(2*np.pi*t.month/12)
    frame['day_of_week']=t.dayofweek
    row=frame.iloc[[-1]].copy()
    missing=[c for c in expected if c not in row.columns]
    if missing:raise NEAError('Model features unavailable: '+', '.join(missing))
    # Do not invent values: the fitted preprocessing pipeline imputes missing
    # historical predictors, exactly as it did during training. The observed
    # current PSI and five weather groups are explicitly checked above.
    # Guard against an entirely absent weather history.
    weather_history_cols = [c for c in expected if c.startswith('weather_') and c not in WX_COLS]
    if weather_history_cols and row[weather_history_cols].notna().sum(axis=1).iloc[0] == 0:
        raise NEAError('All engineered weather history is missing at this forecast hour')
    context={'as_of':cutoff,'region':region,'psi':float(row.psi_twenty_four_hourly.iloc[0]),
             'weather':{col:float(row[col].iloc[0]) for col in WX_COLS}}
    return row[expected].copy(),context
