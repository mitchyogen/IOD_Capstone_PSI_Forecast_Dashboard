"""Official data.gov.sg v2 historical and latest observation access.

The API response format varies between PSI and station weather endpoints.
Unknown formats raise rather than silently guessing values.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
import requests
import pandas as pd

BASE = 'https://api-open.data.gov.sg/v2/real-time/api'
ENDPOINTS = {
    'psi': 'psi', 'wind_speed': 'wind-speed', 'wind_direction': 'wind-direction',
    'temperature': 'air-temperature', 'humidity': 'relative-humidity', 'rainfall': 'rainfall',
}

class NEAError(RuntimeError):
    pass

@dataclass
class NEAClient:
    api_key: str = ''
    timeout: int = 25
    max_pages: int = 50

    def _request(self, variable: str, params: dict | None = None) -> dict:
        if variable not in ENDPOINTS:
            raise ValueError(f'Unknown data source: {variable}')
        headers = {'Accept': 'application/json', 'User-Agent': 'psi-capstone-streamlit/1.0'}
        if self.api_key:
            headers['x-api-key'] = self.api_key
        url = f'{BASE}/{ENDPOINTS[variable]}'
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1)); continue
                if response.status_code >= 400:
                    raise NEAError(f'NEA {variable} HTTP {response.status_code}: {response.text[:180]}')
                payload = response.json()
                if not isinstance(payload, dict) or payload.get('code') not in (0, None):
                    raise NEAError(f'NEA {variable} API error: code={payload.get("code")}, message={str(payload.get("errorMsg", payload.get("message", "")))[:180]}')
                return payload
            except requests.RequestException as exc:
                if attempt == 2:
                    raise NEAError(f'Unable to retrieve {variable}: {type(exc).__name__}') from exc
                time.sleep(1.5 * (attempt + 1))
        raise NEAError('Unreachable request state')

    def fetch(self, variable: str, when: date | None = None) -> tuple[list[dict],dict]:
        """Fetch all response pages of a date, or latest when omitted."""
        params = {'date': when.isoformat()} if when is not None else {}
        readings, metadata = [], {}
        seen_tokens = set()
        for _ in range(self.max_pages):
            payload = self._request(variable, params)
            body = payload.get('data', {})
            if isinstance(body, dict):
                current = body.get('readings', body.get('items', []))
                if not isinstance(current, list):
                    raise NEAError(f'Unexpected {variable} readings schema')
                readings.extend(current)
                if 'stations' in body:
                    metadata['stations'] = body['stations']
                if 'region_metadata' in body:
                    metadata['region_metadata'] = body['region_metadata']
                token = body.get('paginationToken') or body.get('pagination_token')
            else:
                raise NEAError(f'Unexpected {variable} data schema')
            token = token or payload.get('paginationToken') or payload.get('pagination_token')
            if not token:
                return readings, metadata
            if token in seen_tokens:
                raise NEAError('NEA returned a repeated pagination token')
            seen_tokens.add(token)
            params['paginationToken'] = token
        raise NEAError('NEA response exceeded configured pagination limit')


def _local_time(value):
    t = pd.Timestamp(value)
    return t.tz_localize('Asia/Singapore') if t.tzinfo is None else t.tz_convert('Asia/Singapore')


def parse_psi(rows: list[dict], meta: dict) -> tuple[pd.DataFrame,dict]:
    coords = {}
    for item in meta.get('region_metadata', []):
        name = str(item.get('name', '')).lower()
        loc = item.get('label_location') or item.get('location') or {}
        if name and 'latitude' in loc and 'longitude' in loc:
            coords[name] = (float(loc['latitude']),float(loc['longitude']))
    records = []
    for row in rows:
        stamp = row.get('timestamp') or row.get('update_timestamp') or row.get('updatedTimestamp')
        fields = row.get('readings') or row.get('data')
        if not stamp or not isinstance(fields, dict):
            continue
        regions = set()
        for value in fields.values():
            if isinstance(value, dict): regions.update(value)
        for region in regions:
            if region == 'national':continue
            item = {'timestamp': _local_time(stamp), 'region': region.lower()}
            for field, value in fields.items():
                if isinstance(value, dict): item[field] = pd.to_numeric(value.get(region), errors='coerce')
            records.append(item)
    result = pd.DataFrame(records)
    if result.empty: raise NEAError(f'No parseable regional PSI readings. Received {len(rows)} API records; first record keys: {list(rows[0]) if rows else []}')
    return result.drop_duplicates(['timestamp','region'], keep='last'),coords


def parse_weather(rows: list[dict], meta: dict, variable: str) -> pd.DataFrame:
    stations = {}
    for st in meta.get('stations', []):
        sid = str(st.get('id') or st.get('stationId') or '')
        loc = st.get('location') or {}
        if sid and 'latitude' in loc and 'longitude' in loc:
            stations[sid] = (float(loc['latitude']),float(loc['longitude']))
    output = []
    for row in rows:
        stamp = row.get('timestamp')
        observations = row.get('data') or row.get('readings') or []
        if not stamp or not isinstance(observations, list): continue
        for obs in observations:
            sid = str(obs.get('stationId') or obs.get('station_id') or obs.get('station') or '')
            if sid not in stations:continue
            value = pd.to_numeric(obs.get('value'), errors='coerce')
            if pd.isna(value):continue
            output.append({'timestamp':_local_time(stamp), 'station_id':sid,
                           'latitude':stations[sid][0], 'longitude':stations[sid][1],
                           'value':float(value),'variable':variable})
    return pd.DataFrame(output,columns=['timestamp','station_id','latitude','longitude','value','variable'])


def fetch_recent(client: NEAClient, days: list[date]) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    all_psi, all_weather, coords = [], [], {}
    for day in days:
        entries, meta = client.fetch('psi',day)
        psi, locations = parse_psi(entries,meta)
        all_psi.append(psi);coords.update(locations)
        for variable in ENDPOINTS:
            if variable=='psi':continue
            entries, meta = client.fetch(variable,day)
            all_weather.append(parse_weather(entries,meta,variable))
    psi = pd.concat(all_psi,ignore_index=True).drop_duplicates(['timestamp','region'],keep='last')
    weather = pd.concat(all_weather,ignore_index=True).drop_duplicates(['timestamp','station_id','variable'],keep='last')
    if not coords:raise NEAError('PSI API did not provide regional coordinates')
    return psi,weather,coords
