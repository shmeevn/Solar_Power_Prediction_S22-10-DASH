Combined_PV_Weather_Data.csv: PV data set with all relevant weather features
Combined_PV_Weather_Data2.csv: PV data set with only rain and clouds
Formatted_PV_data.csv: PV power data with date and time. Time is in 15 minute intervals (1.15 = 1:15 AM, 16:45 = 16:45 = 4:45 PM)
PVdata.csv: Original raw PV file

CombineData.py: used to combine Formatted_PV_data.csv and weather data from OpenWeatherMap
Format.py: reformat original raw PV file so that date and time and seperated
