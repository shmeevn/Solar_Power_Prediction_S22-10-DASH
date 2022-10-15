import pandas as pd

pv = pd.read_csv('Formatted_PV_data.csv', sep=';')
wd = pd.read_csv('carbondale_weather_data.csv', sep=',')

#pv = pv.drop(['PCS_AC_Power', 'Battery_Power', 'Microgrid_Total_Load', 'Microgrid_Net_Power'], axis = 1)

wd = wd.drop(['dt', 'dt_iso', 'timezone', 'city_name', 'lat', 'lon', 'dew_point', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'sea_level', 'grnd_level', 'wind_deg', 'rain_3h', 'snow_3h', 'weather_id', 'weather_main', 'weather_description', 'weather_icon'], axis = 1)

wd.drop(wd.loc[0:397777].index, inplace=True)
wd = wd[:409]

tdf = []

for index, row in wd.iterrows():
    tdf.append(row)
    tdf.append(row)
    tdf.append(row)
    tdf.append(row)

tempdf = pd.DataFrame(tdf)
result = pd.concat([pv.reset_index(drop=True), tempdf.reset_index(drop=True)], axis = 1)
result = result[:1633]
result.fillna(0, inplace=True)

result.to_csv('Combined_PV_Weather_Data3.csv', sep=';', index = False)