import influxdb_client
import random
import os
import psutil
import time
from datetime import datetime
from influxdb_client.client.write_api import SYNCHRONOUS

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

interval = 0
upper = 100
lower = 0
ctx_prev = psutil.cpu_stats().ctx_switches
int_prev = psutil.cpu_stats().interrupts
while (1):
	if interval < 5:
		new = random.randrange(lower,upper,1)
	else:
		upper = 100
		lower = 0
		interval = 0
		new = random.randrange(lower,upper,1)
	upper = new + 25
	lower = new - 25
	interval = interval + 1
	a = influxdb_client.Point("System").tag("William's PC", "Stats").field("Random", new)
	write_api.write(bucket=bucket, org=org, record=a)
	print("Run complete with data point of ",new,": ",datetime.now())
	
	b = influxdb_client.Point("System").tag("William's PC", "Stats").field("CPU Usage (%)", psutil.cpu_percent(interval=1))
	write_api.write(bucket=bucket, org=org, record=b)
	c = influxdb_client.Point("System").tag("William's PC", "Stats").field("Virtual Memory Usage (%)", psutil.virtual_memory().percent)
	write_api.write(bucket=bucket, org=org, record=c)
	d = influxdb_client.Point("System").tag("William's PC", "Stats").field("Swap Memory Usage (%)", psutil.swap_memory().percent)
	write_api.write(bucket=bucket, org=org, record=d)
	e = influxdb_client.Point("System").tag("William's PC", "Stats").field("Virtual Memory Used (bytes)", psutil.virtual_memory().used)
	write_api.write(bucket=bucket, org=org, record=e)
	f = influxdb_client.Point("System").tag("William's PC", "Stats").field("Swap Memory Used (bytes)", psutil.swap_memory().used)
	write_api.write(bucket=bucket, org=org, record=f)
	h = influxdb_client.Point("System").tag("William's PC", "Stats").field("CPU Context Switches (per 10s)", psutil.cpu_stats().ctx_switches-ctx_prev)
	ctx_prev = psutil.cpu_stats().ctx_switches
	write_api.write(bucket=bucket, org=org, record=h)
	i = influxdb_client.Point("System").tag("William's PC", "Stats").field("CPU Interrupts (per 10s)", psutil.cpu_stats().interrupts-int_prev)
	int_prev = psutil.cpu_stats().interrupts
	write_api.write(bucket=bucket, org=org, record=i)
	j = influxdb_client.Point("System").tag("William's PC", "Stats").field("CPU Frequency (GHz)", psutil.cpu_freq().current)
	write_api.write(bucket=bucket, org=org, record=j)
	k = influxdb_client.Point("System").tag("William's PC", "Stats").field("Disk Space Used (bytes)", psutil.disk_usage('/').used)
	write_api.write(bucket=bucket, org=org, record=k)
	time.sleep(10)

