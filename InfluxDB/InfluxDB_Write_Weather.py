import requests
import json
import influxdb_client
import os
import time
from datetime import datetime
from influxdb_client.client.write_api import SYNCHRONOUS

api_key = "725b8b744d565459c216cdf0c153095f"
base_url = "http://api.openweathermap.org/data/2.5/weather?"
city_name = "Carbondale"
urlw = base_url + "appid=" + api_key + "&q=" + city_name + "&units=imperial"

token = "CTsee8YzfdrH9z56ngEI5CsnyNyNvMj06BRczLWARODEi8DEq_Y4UtFhdKDnAlQJTWCA5XEiPwzu91UYH6aHFg=="
org = "stephen.berg24@gmail.com"
bucket = "Testing Bucket"
url="https://us-east-1-1.aws.cloud2.influxdata.com"

client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org
)

write_api = client.write_api(write_options=SYNCHRONOUS)

while (1):
	response = requests.get(urlw)
	main = json.loads(response.text)
	data = main["main"]
	ctemp = data["temp"]
	cpres = data["pressure"]
	chumi = data["humidity"]
	cclou = main["clouds"]["all"]
	cvisi = main["visibility"]
	ccond = main["weather"][0]["id"]
	
	a = influxdb_client.Point("System").tag("Weather", "Current").field("Temperature (C)", ctemp)
	write_api.write(bucket=bucket, org=org, record=a)
	b = influxdb_client.Point("System").tag("Weather", "Current").field("Pressure (hPa)", cpres)
	write_api.write(bucket=bucket, org=org, record=b)
	c = influxdb_client.Point("System").tag("Weather", "Current").field("Humidity (%)", chumi)
	write_api.write(bucket=bucket, org=org, record=c)
	d = influxdb_client.Point("System").tag("Weather", "Current").field("Cloud Coverage (%)", cclou)
	write_api.write(bucket=bucket, org=org, record=d)
	e = influxdb_client.Point("System").tag("Weather", "Current").field("Visibility (km)", cvisi)
	write_api.write(bucket=bucket, org=org, record=e)
	f = influxdb_client.Point("System").tag("Weather", "Current").field("Weather Condition", ccond)
	write_api.write(bucket=bucket, org=org, record=f)
	time.sleep(1800)

