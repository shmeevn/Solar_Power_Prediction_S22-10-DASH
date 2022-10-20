import LSTM as lstm
import tb_util as tbu
import matplotlib.pyplot as plt #for plotting
import time
import numpy as np
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

##influx stuff
bucket = 'PV_Data'
org = 'f230e8be61c0d754'
token = 'v-CuzclHO2yI9YOjjZYTKTm_5WPPfu9Tl0reLRO44ul0BfDbe3IW3elxZ0sDAjFWtmhguHBcOqMJfkLdZzOktw=='
url='https://us-east-1-1.aws.cloud2.influxdata.com'

client = influxdb_client.InfluxDBClient(
   url=url,
   token=token,
   org=org
)
write_api = client.write_api(write_options=SYNCHRONOUS)

##Parameters
input_size = 2              #Number of input features
output_size = 1             #Number of output features
recurrent_units = 32        #Number of RNN/LSTM cells
seq_length = 5              #Number of input-vectors per sample
numEpochs = 200              #Number of training passes
batch_size = 1              #How many samples should be ran in parallel per training step
drop = 0.0                  #Dropout parameter
train_test_split = 0.8      #Training/Validation split percentage as decimal, 0.8 -> 80% training, 20% Validation
loss_function = 'mse'       #Loss function to use with the optimizer, 'mse' and 'mae' should work
n = 1

#Load Dataset, specify columns by name, None means all
inputColumns = ['Time', 'PV_Power']
outputColumns = ['PV_Power']
Xt, Yt, Xv, Yv, X, Y = tbu.loadData("Formatted_PV_data.csv", seq_length, train_test_split, inputColumns, outputColumns)

#for the LSTM
model = lstm.makeModel(input_size, seq_length, batch_size, recurrent_units, output_size, drop, loss_function)
model, khistory, trainHistory2 = lstm.trainModel(model, Xt, Yt, Xv, Yv, numEpochs, batch_size)

while True:
    try:
        pred = lstm.Predict(model, X[n:n+1], batch_size)
        if n <= X.size:
            n = n + 1
        else:
            n = 1
        p = influxdb_client.Point('test').field('PV_Power_Pred', float(pred[0][0]))
        write_api.write(bucket=bucket, org=org, record=p)
        p = influxdb_client.Point('test').field('PV_Power_Actual', float(Y[n-1:n][0][0]))
        write_api.write(bucket=bucket, org=org, record=p)
        time.sleep(0.01)
        p = influxdb_client.Point('test').field('PV_Power_Pred', float(pred[0][1]))
        write_api.write(bucket=bucket, org=org, record=p)
        p = influxdb_client.Point('test').field('PV_Power_Actual', float(Y[n-1:n][0][1]))
        write_api.write(bucket=bucket, org=org, record=p)
        time.sleep(0.01)
        p = influxdb_client.Point('test').field('PV_Power_Pred', float(pred[0][2]))
        write_api.write(bucket=bucket, org=org, record=p)
        p = influxdb_client.Point('test').field('PV_Power_Actual', float(Y[n-1:n][0][2]))
        write_api.write(bucket=bucket, org=org, record=p)
        time.sleep(0.01)
        p = influxdb_client.Point('test').field('PV_Power_Pred', float(pred[0][3]))
        write_api.write(bucket=bucket, org=org, record=p)
        p = influxdb_client.Point('test').field('PV_Power_Actual', float(Y[n-1:n][0][3]))
        write_api.write(bucket=bucket, org=org, record=p)
        time.sleep(0.01)
        p = influxdb_client.Point('test').field('PV_Power_Pred', float(pred[0][4]))
        write_api.write(bucket=bucket, org=org, record=p)
        p = influxdb_client.Point('test').field('PV_Power_Actual', float(Y[n-1:n][0][4]))
        write_api.write(bucket=bucket, org=org, record=p)
        time.sleep(0.01)
    except Exception:
        print("Error")
    