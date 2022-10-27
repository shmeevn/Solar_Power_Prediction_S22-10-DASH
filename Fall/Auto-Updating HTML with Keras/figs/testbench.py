import LSTM as lstm
import tb_util as tbu
import matplotlib.pyplot as plt #for plotting
import time
import numpy as np
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import pdb   

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
t = 0

#Load Dataset, specify columns by name, None means all
inputColumns = ['Time', 'PV_Power']
outputColumns = ['PV_Power']
Xt, Yt, Xv, Yv, X, Y = tbu.loadData("Formatted_PV_data.csv", seq_length, train_test_split, inputColumns, outputColumns)

#for the LSTM
model = lstm.makeModel(input_size, seq_length, batch_size, recurrent_units, output_size, drop, loss_function)
model, khistory, trainHistory2 = lstm.trainModel(model, Xt, Yt, Xv, Yv, numEpochs, batch_size)

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.style.use('dark_background')

yy1 = []
yy2 = []
xx = []

while True:
    try:
        pred = lstm.Predict(model, X[n:n+1], batch_size)
        if n + 1 <= X.size:                     ## counter to get the real data slice from Y
            n = n + 1
        else:
            n = 1
            
        plt.clf()                               ## clear the plot otherwise we will be writing the same data overlayed onto the previous
        
        yy1.append(float(pred[0][0]))           ## append the data to the list, do this 5 times since the predict function takes and returns 5 data points
        yy2.append(float(Y[n-1:n][0][0]))
        xx.append(t)
        
        yy1.append(float(pred[0][1]))
        yy2.append(float(Y[n-1:n][0][1]))
        xx.append(t+1)
        
        yy1.append(float(pred[0][2]))
        yy2.append(float(Y[n-1:n][0][2]))
        xx.append(t+2)
        
        yy1.append(float(pred[0][3]))
        yy2.append(float(Y[n-1:n][0][3]))
        xx.append(t+3)
        
        yy1.append(float(pred[0][4]))
        yy2.append(float(Y[n-1:n][0][4]))
        xx.append(t+4)
        
        if len(yy1) > 100:                      ## if we reach over 100 on the list, remove the first element
            yy1 = yy1[5:]
            yy2 = yy2[5:]
            xx = xx[5:]
        
        plt.plot(xx, yy1, 'r-', label="Prediction")
        plt.plot(xx, yy2, 'g-', label="True")
        plt.legend()
        
        time.sleep(1)
        plt.savefig("fig.png")
        t = t + 5
    except Exception:
        print("Error")
