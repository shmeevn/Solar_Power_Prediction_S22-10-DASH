import LSTM as lstm
import tb_util as tbu
import matplotlib.pyplot as plt #for plotting
import matplotlib.dates as mfile
import time
import pandas as pd
import keras
import numpy as np
import sklearn
from sklearn.feature_selection import r_regression
import sys  
import datetime as dt

recurrent_units = 32        #Number of RNN/LSTM cells
seq_length = 5              #Number of input-vectors per sample
numEpochs = 200             #Number of training passes
batch_size = 1              #How many samples should be ran in parallel per training step
drop = 0.0                  #Dropout parameter
train_test_split = 0.8      #Training/Validation split percentage as decimal, 0.8 -> 80% training, 20% Validation
loss_function = 'mse'       #Loss function to use with the optimizer, 'mse' and 'mae' should work
n = 1
t = 0

if (str(sys.argv[1]) == 'save'):
    if (str(sys.argv[2]) == 'weather'):
        ##Parameters
        input_size = 9              #Number of input features
        output_size = 1             #Number of output features

        #Load Dataset, specify columns by name, None means all
        inputColumns = ['PV_Demand', 'Hour', 'Minute', 'Month', 'Day', 'clouds_all', 'precip_1h', 'temp', 'wind_speed']
        #inputColumns = ['PV_Demand', 'Hour', 'Minute', 'Month', 'Day']
        outputColumns = ['PV_Demand']
        Xt, Yt, Xv, Yv, X, Y, data = tbu.loadData("combined.csv", seq_length, train_test_split, inputColumns, outputColumns)
        
        #hr_r2 = data[['Hour']].corr()['PV_Demand'][:]
        #print('Hour R2 = ' + str(hr_r2) + '\n')

        #for the LSTM
        model = lstm.makeModel(input_size, seq_length, batch_size, recurrent_units, output_size, drop, loss_function)
        model, khistory, trainHistory2 = lstm.trainModel(model, Xt, Yt, Xv, Yv, numEpochs, batch_size)
        model.save('LSTM_weather')
        
        exit()
    elif (str(sys.argv[2]) == 'noweather'):
        ##Parameters
        input_size = 5              #Number of input features
        output_size = 1             #Number of output features

        #Load Dataset, specify columns by name, None means all
        #inputColumns = ['PV_Demand', 'Hour', 'Minute', 'Month', 'Day', 'clouds_all', 'precip_1h']
        inputColumns = ['PV_Demand', 'Hour', 'Minute', 'Month', 'Day']
        outputColumns = ['PV_Demand']
        Xt, Yt, Xv, Yv, X, Y, data = tbu.loadData("combined.csv", seq_length, train_test_split, inputColumns, outputColumns)

        #for the LSTM
        model = lstm.makeModel(input_size, seq_length, batch_size, recurrent_units, output_size, drop, loss_function)
        model, khistory, trainHistory2 = lstm.trainModel(model, Xt, Yt, Xv, Yv, numEpochs, batch_size)
        model.save('LSTM_noweather')
        exit()
    else:
        print('argument not valid')
        exit()  
elif (str(sys.argv[1]) == 'load'):
    if (str(sys.argv[2]) == 'weather'):
        model = keras.models.load_model('LSTM_weather')
        inputColumns = ['PV_Demand', 'Hour', 'Minute', 'Month', 'Day', 'clouds_all', 'precip_1h', 'temp', 'wind_speed']
        outputColumns = ['PV_Demand']
        Xt, Yt, Xv, Yv, X, Y, data = tbu.loadData("combined.csv", seq_length, train_test_split, inputColumns, outputColumns)
        
        valLoss, trueValues, predictions = lstm.ValidationTest(model, Xv, Yv, batch_size)

        rmse = str(round(np.sqrt(sklearn.metrics.mean_squared_error(trueValues, predictions)) * 1000) / 1000)
        mae = str(round(sklearn.metrics.mean_absolute_error(trueValues, predictions) * 1000) / 1000)
        r2 = str(round(sklearn.metrics.r2_score(trueValues, predictions) * 1000) / 1000)

        cor = data[data.columns[1:]].corr()['PV_Demand'][:]

        mon_cor = str(round(cor[0] * 1000) / 1000)
        day_cor = str(round(cor[1] * 1000) / 1000)
        hr_cor = str(round(cor[2] * 1000) / 1000)
        min_cor = str(round(cor[2] * 1000) / 1000)
        temp_cor = str(round(cor[5] * 1000) / 1000)
        wind_cor = str(round(cor[6] * 1000) / 1000)
        clds_cor = str(round(cor[9] * 1000) / 1000)
        prcp_cor = str(round(cor[10] * 1000) / 1000)

        strg = 'Validation Metrics:\n\n\tRMSE Score:\t' + rmse + '\n\tMAE:\t\t' + mae + '\n\tR2 Score:\t' + r2 + '\n\n'
        strg = strg + 'Feature Correlations:\n\n\tMonth:\t\t' + mon_cor + '\n\tDay:\t\t' + day_cor + '\n\tHour:\t\t' + hr_cor + '\n\tMinute:\t\t' + min_cor
        strg = strg + '\n\tTemperature:\t' + temp_cor + '\n\tWind Speed:\t' + wind_cor + '\n\tCloud Coverage:\t' + clds_cor + '\n\tPrecipitation:\t' + prcp_cor
        with open('cor.txt', 'w') as f:
            f.write(strg)
    elif (str(sys.argv[2]) == 'noweather'):
        model = keras.models.load_model('LSTM_noweather')
        inputColumns = ['PV_Demand', 'Hour', 'Minute', 'Month', 'Day']
        outputColumns = ['PV_Demand']
        Xt, Yt, Xv, Yv, X, Y, data = tbu.loadData("combined.csv", seq_length, train_test_split, inputColumns, outputColumns)
        
        valLoss, trueValues, predictions = lstm.ValidationTest(model, Xv, Yv, batch_size)

        rmse = str(round(np.sqrt(sklearn.metrics.mean_squared_error(trueValues, predictions)) * 1000) / 1000)
        mae = str(round(sklearn.metrics.mean_absolute_error(trueValues, predictions) * 1000) / 1000)
        r2 = str(round(sklearn.metrics.r2_score(trueValues, predictions) * 1000) / 1000)

        cor = data[data.columns[1:]].corr()['PV_Demand'][:]

        mon_cor = str(round(cor[0] * 1000) / 1000)
        day_cor = str(round(cor[1] * 1000) / 1000)
        hr_cor = str(round(cor[2] * 1000) / 1000)
        min_cor = str(round(cor[2] * 1000) / 1000)

        strg = 'Validation Metrics:\n\n\tRMSE Score:\t' + rmse + '\n\tMAE:\t\t' + mae + '\n\tR2 Score:\t' + r2 + '\n\n'
        strg = strg + 'Feature Correlations:\n\n\tMonth:\t\t' + mon_cor + '\n\tDay:\t\t' + day_cor + '\n\tHour:\t\t' + hr_cor + '\n\tMinute:\t\t' + min_cor
        with open('cor.txt', 'w') as f:
            f.write(strg)
    else:
        print('argument not valid')
        exit()
else:
    print('argument not valid')
    exit()

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.style.use('dark_background')

plt.scatter(predictions, trueValues, label="Correlation", s=5)
plt.xlabel('PV Demand (kW) for Prediction')
plt.ylabel('PV Demand (kW) for True Data')
plt.legend()
plt.axis('square')
plt.savefig("cor_fig.png")
plt.clf()

pred_y = []
true_y = []
clouds_y = []   # 5
precip_y = []   # 6
temp_y = []     # 7
wind_y = []     # 8
xx = []

while True:
    pred = lstm.Predict(model, X[n:n+1], batch_size)
    if n + 1 <= X.size:
        n = n + 1
    else:
        n = 1

    pred_y.append(float(pred[0][0])) if float(pred[0][0]) > 0 else pred_y.append(0)
    true_y.append(float(Y[n-1:n][0][0]))
    xx.append(t)
    
    pred_y.append(float(pred[0][1])) if float(pred[0][1]) > 0 else pred_y.append(0)
    true_y.append(float(Y[n-1:n][0][1]))
    xx.append(t+1)
    
    pred_y.append(float(pred[0][2])) if float(pred[0][2]) > 0 else pred_y.append(0)
    true_y.append(float(Y[n-1:n][0][2]))
    xx.append(t+2)

    
    pred_y.append(float(pred[0][3])) if float(pred[0][3]) > 0 else pred_y.append(0)
    true_y.append(float(Y[n-1:n][0][3]))
    xx.append(t+3)

    pred_y.append(float(pred[0][4])) if float(pred[0][4]) > 0 else pred_y.append(0)
    true_y.append(float(Y[n-1:n][0][4]))
    xx.append(t+4)

    if (str(sys.argv[2]) == 'weather'):
        clouds_y.append(float(X[n-1:n][0][0][5]))
        precip_y.append(float(X[n-1:n][0][0][6]))
        temp_y.append(float(X[n-1:n][0][0][7]))
        wind_y.append(float(X[n-1:n][0][0][8]))
        
        clouds_y.append(float(X[n-1:n][0][1][5]))
        precip_y.append(float(X[n-1:n][0][1][6]))
        temp_y.append(float(X[n-1:n][0][1][7]))
        wind_y.append(float(X[n-1:n][0][1][8]))
        
        clouds_y.append(float(X[n-1:n][0][2][5]))
        precip_y.append(float(X[n-1:n][0][2][6]))
        temp_y.append(float(X[n-1:n][0][2][7]))
        wind_y.append(float(X[n-1:n][0][2][8]))
        
        clouds_y.append(float(X[n-1:n][0][3][5]))
        precip_y.append(float(X[n-1:n][0][3][6]))
        temp_y.append(float(X[n-1:n][0][3][7]))
        wind_y.append(float(X[n-1:n][0][3][8]))
        
        clouds_y.append(float(X[n-1:n][0][4][5]))
        precip_y.append(float(X[n-1:n][0][4][6]))
        temp_y.append(float(X[n-1:n][0][4][7]))
        wind_y.append(float(X[n-1:n][0][4][8]))
        
        if len(clouds_y) > 100:
            clouds_y = clouds_y[5:]
            precip_y = precip_y[5:]
            temp_y = temp_y[5:]
            wind_y = wind_y[5:]
    
    if len(pred_y) > 100:
        pred_y = pred_y[5:]
        true_y = true_y[5:]
        xx = xx[5:]

    plt.plot(xx, pred_y, 'r-', label="Prediction")
    plt.plot(xx, true_y, 'g-', label="True")
    plt.xlabel('Time Step (15 minutes)')
    plt.ylabel('PV Demand (kW)')
    plt.legend()
    plt.savefig("pv_fig.png")
    plt.clf()
    
    if (str(sys.argv[2]) == 'weather'):
    
        plt.plot(xx, temp_y, 'b-', label="Temperature")
        plt.xlabel('Time Step (15 minutes)')
        plt.ylabel('Temperature (K)')
        plt.legend()
        plt.savefig("temp_fig.png")
        plt.clf()
        
        plt.plot(xx, wind_y, 'b-', label="Wind")
        plt.xlabel('Time Step (15 minutes)')
        plt.ylabel('Wind (m/s)')
        plt.legend()
        plt.savefig("wind_fig.png")
        plt.clf()
        
        plt.plot(xx, precip_y, 'b-', label="Precipitation")
        plt.xlabel('Time Step (15 minutes)')
        plt.ylabel('Precipitation (mm/hr)')
        plt.legend()
        plt.savefig("precip_fig.png")
        plt.clf()
        
        plt.plot(xx, clouds_y, 'b-', label="Cloud Coverage")
        plt.xlabel('Time Step (15 minutes)')
        plt.ylabel('Cloud Coverage (%)')
        plt.legend()
        plt.savefig("clouds_fig.png")
        plt.clf()
    
    t = t + 5
    time.sleep(0)