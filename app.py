"""Singapore PSI dashboard. Observed NEA data is shown separately from modelled forecasts."""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.nea_client import NEAClient, NEAError, parse_psi, fetch_recent
from src.features import REGIONS, aggregate_weather, features_for_region
from src.model import ModelError, load_bundle, predict

SG=ZoneInfo('Asia/Singapore')
BASE=Path(__file__).resolve().parent
MODEL=BASE/'models'/'psi_weather_xgboost.joblib'
REGION_NAMES={r:r.title() for r in REGIONS}
st.set_page_config(page_title='Singapore PSI Forecast | Capstone',page_icon='🌤️',layout='wide',initial_sidebar_state='expanded')

CSS='''
<style>
.block-container {padding-top: 1.55rem; max-width: 1280px;}
.hero {padding:1.3rem 1.5rem;background:linear-gradient(115deg,#102F4C,#15696F);color:#fff;border-radius:18px;margin-bottom:1.0rem}
.hero h1 {font-size:2.15rem;margin:.2rem 0 .5rem;font-weight:730;color:white;}
.hero p {color:#DBECEF;margin:0;font-size:.97rem;}
.subtle {font-size:.85rem;color:#607488}
[data-testid="stMetric"] {background:#F0F6F9;border:1px solid #DCE8EF;padding:1.05rem;border-radius:14px}
[data-testid="stMetricLabel"] {color:#536C80}
.section-title {font-size:1.19rem;font-weight:740;color:#18324B;margin-top:1rem}
</style>'''
st.markdown(CSS,unsafe_allow_html=True)

@st.cache_data(ttl=900,show_spinner=False)
def cached_latest(api_key: str):
    client=NEAClient(api_key=api_key)
    raw,metadata=client.fetch('psi')
    return parse_psi(raw,metadata)

@st.cache_data(ttl=900,show_spinner=False)
def cached_recent(api_key: str, last_day: str):
    end=pd.Timestamp(last_day).date()
    days=[end-timedelta(days=i) for i in range(3,-1,-1)]
    return fetch_recent(NEAClient(api_key=api_key),days)

@st.cache_resource(show_spinner=False)
def cached_model():
    return load_bundle(MODEL)


def status_for_psi(value):
    if pd.isna(value):return 'Not available'
    if value<=50:return 'Good'
    if value<=100:return 'Moderate'
    if value<=200:return 'Unhealthy'
    if value<=300:return 'Very unhealthy'
    return 'Hazardous'


def psi_plot(data:pd.DataFrame, region:str, prediction:float|None=None, target_time=None):
    g=data[data.region.eq(region)].dropna(subset=['psi_twenty_four_hourly']).sort_values('timestamp')
    g=g[g.timestamp.ge(pd.Timestamp.now(tz=SG)-pd.Timedelta(hours=30))]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=g.timestamp,y=g.psi_twenty_four_hourly,mode='lines+markers',
                             line=dict(color='#3976B8',width=3),marker=dict(size=4),name='Observed 24h PSI'))
    if prediction is not None and not g.empty:
        fig.add_trace(go.Scatter(x=[g.timestamp.iloc[-1],target_time],
                                 y=[g.psi_twenty_four_hourly.iloc[-1],prediction],
                                 mode='lines+markers',line=dict(color='#16877D',width=3,dash='dash'),
                                 name='Model forecast (+3h)'))
    fig.update_layout(height=340,margin=dict(l=12,r=12,t=20,b=10),
        plot_bgcolor='white',paper_bgcolor='white',font=dict(color='#18324B'),
        xaxis=dict(title='',showgrid=True,gridcolor='#EDF2F5'),
        yaxis=dict(title='24-hour PSI',showgrid=True,gridcolor='#EDF2F5'),
        legend=dict(orientation='h',y=-.22))
    return fig


def data_age(ts):
    latest=pd.Timestamp(ts)
    if latest.tzinfo is None:latest=latest.tz_localize(SG)
    return pd.Timestamp.now(tz=SG)-latest

with st.sidebar:
    st.markdown('### 🌤️ Air Quality Intelligence')
    region=st.selectbox('PSI region',REGIONS,format_func=lambda r:REGION_NAMES[r])
    st.markdown('**Source:** National Environment Agency via data.gov.sg')
    st.caption('New API data is checked every 15 minutes while the dashboard is open.')
    if st.button('Refresh now',use_container_width=True):
        cached_latest.clear();cached_recent.clear();st.rerun()
    st.divider()
    st.caption('Experimental capstone project. Not affiliated with NEA. Use NEA advisories for health and safety decisions.')

st.markdown('''<div class="hero"><div style="font-size:.8rem;letter-spacing:2px;color:#A7E4DC;font-weight:700">SINGAPORE · REGIONAL AIR QUALITY</div>
<h1>Know the air. Plan the next three hours.</h1><p>Official observations with a clearly labelled research forecast when inputs have been validated.</p></div>''',unsafe_allow_html=True)

try:api_key=st.secrets.get('NEA_API_KEY','')
except Exception:api_key=''

@st.fragment(run_every='15m')
def live_dashboard():
    now=datetime.now(SG)
    st.caption(f'Dashboard checked: {now:%d %b %Y, %I:%M %p} SGT  |  Scheduled refresh: every 15 minutes in an active session')
    try:
        latest,latlon=cached_latest(api_key)
    except Exception as exc:
        st.error('NEA PSI data is currently unavailable. No values or forecasts have been generated.')
        st.caption(f'Diagnostic: {type(exc).__name__}: {exc}')
        return
    observed=latest[latest.region.eq(region)].sort_values('timestamp')
    if observed.empty:
        st.warning('No current PSI record found for this region.');return
    current=observed.iloc[-1]
    value=pd.to_numeric(current.get('psi_twenty_four_hourly'),errors='coerce')
    latest_ts=current['timestamp']
    age=data_age(latest_ts)
    staleness=age>pd.Timedelta(minutes=75) or age<pd.Timedelta(minutes=-10)
    if staleness:
        st.warning('The latest observation is stale or timestamped unexpectedly. Check the source before acting on it.')
    cols=st.columns([1,1,1,1])
    cols[0].metric('Current 24-hour PSI',f'{value:.0f}' if pd.notna(value) else 'Unavailable')
    cols[1].metric('Current PSI category',status_for_psi(value))
    cols[2].metric('Latest NEA observation',latest_ts.strftime('%I:%M %p SGT'))
    cols[3].metric('Data age',f'{max(0,age.total_seconds()/60):.0f} min')
    st.caption('The PSI shown above is an official observation. The model forecast, if enabled, is a separate research estimate.')
    st.divider()
    prediction=None;context=None;forecast_status='Forecast unavailable'
    historical=None;weather=None
    try:
        historical,weather,coords=cached_recent(api_key,now.date().isoformat())
        coords.update(latlon)
    except Exception as exc:
        st.info('Historical API data could not be assembled. Showing official observations only.')
        st.caption(f'Historical-data diagnostic: {type(exc).__name__}: {exc}')

    # A source-and-training parity test is required before enabling numeric forecasts.
    # A model load alone is not enough to validate live feature engineering.
    try:enabled=bool(st.secrets.get('ENABLE_FORECAST',False))
    except Exception:enabled=False
    if enabled and historical is not None and not staleness:
        try:
            bundle=cached_model()
            wx=aggregate_weather(weather,coords)
            # Only hourly completed observations are used, and historic feature windows
            # must be complete. No imputation outside the model's fitted preprocessor.
            as_of=pd.Timestamp(latest_ts).floor('h')
            row,context=features_for_region(historical,wx,coords,region,as_of,list(bundle['feature_columns']))
            prediction=predict(bundle,row)
            forecast_status='Experimental model forecast'
        except Exception as exc:
            forecast_status='Forecast unavailable: incomplete or incompatible inputs'
            st.info(forecast_status)
            with st.expander('Technical reason'):
                st.code(f'{type(exc).__name__}: {exc}')
    elif not enabled:
        st.info('Experimental forecasts are disabled until the live NEA feature reconstruction has passed training-parity and dependency checks. Observed data remains available.')
    elif staleness:
        st.info('Forecast disabled because current PSI data is stale.')

    a,b=st.columns([1,1])
    with a:
        st.markdown('#### PSI forecast in three hours')
        if prediction is None:
            st.metric('Predicted 24-hour PSI','Not available')
            st.caption('No unverified numerical prediction is shown.')
        else:
            st.metric('Predicted 24-hour PSI',f'{prediction:.1f}',delta=f'{prediction-value:+.1f} vs observed PSI',delta_color='off')
            st.caption(f'Model origin: {context["as_of"]:%d %b %Y %H:%M} SGT | Target: {context["as_of"]+pd.Timedelta(hours=3):%H:%M} SGT')
            st.caption('Research estimate, not an NEA forecast or health advisory. Test MAE 1.3086 is an average from 2024, not a prediction interval.')
    with b:
        st.markdown('#### What this means')
        st.write('The 24-hour PSI reflects recent air quality. A three-hour forecast can help with planning, but the official NEA advisory remains the source for health decisions.')
        st.markdown('[Check official NEA air-quality information](https://www.haze.gov.sg/)')

    st.markdown('#### Air-quality history')
    trend=historical if historical is not None else latest
    st.plotly_chart(psi_plot(trend,region,prediction,
                           (context['as_of']+pd.Timedelta(hours=3)) if context else None),use_container_width=True)
    st.caption('Observed 24-hour PSI is shown in blue. A teal dashed endpoint appears only if a validated forecast is enabled.')

    st.markdown('#### Regional PSI at the most recent NEA update')
    per_region=(latest.sort_values('timestamp').groupby('region',as_index=False).last())
    per_region=per_region[per_region.region.isin(REGIONS)].copy()
    per_region['category']=per_region.psi_twenty_four_hourly.map(status_for_psi)
    per_region['region']=per_region.region.str.title()
    st.dataframe(per_region[['region','psi_twenty_four_hourly','category','timestamp']].rename(columns={
        'region':'Region','psi_twenty_four_hourly':'24h PSI','category':'Category','timestamp':'Observation time'}),
        use_container_width=True,hide_index=True)

    st.markdown('#### Meteorological observations')
    if weather is not None and not weather.empty:
        latest_station=(weather[weather.timestamp.ge(weather.timestamp.max()-pd.Timedelta(minutes=30))]
                        .sort_values('timestamp').groupby(['variable','station_id'],as_index=False).last())
        if not latest_station.empty:
            latest_station=latest_station.rename(columns={'variable':'Weather variable','station_id':'Station',
                                        'value':'Reading','timestamp':'Observed at'})
            st.dataframe(latest_station[['Weather variable','Station','Reading','Observed at']],
                         use_container_width=True,hide_index=True,height=250)
            st.caption('These are station-level NEA readings, not invented region averages. Stations are mapped to PSI regions only during model feature preparation.')
    else:st.info('Recent station observations are unavailable.')
    with st.expander('About the model and limitations'):
        st.markdown('''**Model:** saved XGBoost regression pipeline, 61 input features including region.

**Forecast target:** regional 24-hour PSI three hours ahead. **Training:** 2019 to 2023.

**Historical test results:** 2024 matched-sample MAE 1.3086 with meteorology versus 1.3466 without.
This measures average error over the test set, not a guarantee for any one prediction.

**Caveats:** station-to-region assignment is approximate, PSI changes slowly as a 24-hour index,
and streaming data completeness and publication delays may differ from the historical training set.
No forecast is generated if required inputs are missing or the model environment is incompatible.''')

live_dashboard()
