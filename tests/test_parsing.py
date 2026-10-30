from src.nea_client import parse_psi, parse_weather
from src.features import geo_distance

def test_parse_psi_sample():
    rows=[{'timestamp':'2024-02-02T10:00:00+08:00','readings':{'psi_twenty_four_hourly':{'central':51,'north':60},'pm25_twenty_four_hourly':{'central':12,'north':14}}}]
    meta={'region_metadata':[{'name':'central','label_location':{'latitude':1.31,'longitude':103.83}}]}
    df,coords=parse_psi(rows,meta)
    assert len(df)==2 and coords['central']==(1.31,103.83)
    assert df.set_index('region').loc['central','psi_twenty_four_hourly']==51

def test_parse_weather_sample():
    rows=[{'timestamp':'2024-02-02T10:00:00+08:00','data':[{'stationId':'S1','value':2.3}]}]
    meta={'stations':[{'id':'S1','location':{'latitude':1.3,'longitude':103.8}}]}
    df=parse_weather(rows,meta,'wind_speed')
    assert df.iloc[0].value==2.3 and df.iloc[0].station_id=='S1'

def test_geo_distance():
    assert geo_distance(1.3,103.8,1.3,103.8)==0
