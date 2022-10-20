import keras                    #For loading model
import matplotlib.pyplot as plt #for plotting
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import time
import pdb                      #for debugging, and exiting to interactive shell
from os import mkdir, path

##Parameters
input_size = 2              #Number of input features
output_size = 1             #Number of output features
recurrent_units = 32        #Number of RNN/LSTM cells
seq_length = 10              #Number of input-vectors per sample
numEpochs = 40              #Number of training passes
batch_size = 1              #How many samples should be ran in parallel per training step
drop = 0.0                  #Dropout parameter
train_test_split = 1.0      #Training/Validation split percentage as decimal, 0.8 -> 80% training, 20% Validation
loss_function = 'mse'       #Loss function to use with the optimizer, 'mse' and 'mae' should work

data = pd.read_csv("Formatted_PV_data.csv", sep=';')
data = np.array(data)
Xt = np.array(data[:-1,[1, 2]], dtype=float)
model = keras.models.load_model("LSTM3")
if not path.exists("page"):
    mkdir("page")
    
if not path.exists("page/figs"):
    mkdir("page/figs")

#Plot the predictions
def plotPrediction(dates, predictions, ct):
    plt.style.use('dark_background')
    fig2, ax2 = plt.subplots(1, 1, constrained_layout=True, figsize=(15, 8.43))
    #XP = [ x for x in range(len(predictions))]
    ax2.plot(predictions, 'r-', label="Power Output (kW)")
    ax2.set_xticks(range(len(predictions)))
    ax2.set_xticklabels(dates, rotation=90)
    #ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.legend(loc='best')
    #save figure
    fig2.savefig("page/figs/fig_{:n}_{:n}.png".format(ct.tm_hour, ct.tm_min//15*15), dpi=200)
    plt.close(fig2)
    
def generatePage(ct):
    head = '<html lang="en">\n\t<head>\n\t\t<title>S22-10-DASH RTPO</title>\n\t\t<meta charset = "utf-8" />\n\t\t<style>\n\t\tbody {background-color: Black;}\n\t\tcaption {color: Gainsboro; font-size: 200%; font-weight: bold;}\n\t\tth {color: Gainsboro; font-size: 150%;}\n\t\ttd {color: Gainsboro; padding: 5px; font-size: 150%;}\n\t\t</style>\n\t</head>\n\t\n'
    body = '\t<body>\n\t\t<table border="0">\n\t\t<caption>S22-10-DASH Real Time Predictive Output</caption>\n\t\t\t<tbody>\n\t\t\t\t<tr>\n\t\t\t\t\t<th rowspan = "10"><img src="{}" alt="Real Time Predictive Output" width=1500 height=850 ></th>\n\t\t\t\t\t<th colspan = "2" height="10px">Times</th>\n\t\t\t\t</tr>\n\t\t\t\t<tr height= "20px">\n\t\t\t\t\t<th width="200px">Generated<br>(h-m-s m/d/y)</th>\n\t\t\t\t\t<td width="150">{:n}:{:n}:{:n} {:n}/{:n}/{:n}</td>\n\t\t\t\t</tr>\n\t\t\t\t<tr height = "20px">\n\t\t\t\t\t<th colspan = "2"><br>Prediction Window<br>(h-m-s m/d/y)</th>\n\t\t\t\t</tr>\n\t\t\t\t<tr height = "20px">\n\t\t\t\t\t<th>From</th>\n\t\t\t\t\t<td>{:n}:{:n}:{:n} {:n}/{:n}/{:n}</td>\n\t\t\t\t</tr>\n\t\t\t\t<tr height = "20px">\n\t\t\t\t\t<th>To</th>\n\t\t\t\t\t<td>{:n}:{:n}:{:n} {:n}/{:n}/{:n}</td>\n\t\t\t\t</tr>\n\t\t\t\t<tr><th></th></tr>\n\t\t\t</tbody>\n\t\t</table>\n\t</body>\n</html>'
    
    generated = [ct.tm_hour, ct.tm_min, ct.tm_sec, ct.tm_mon, ct.tm_mday, ct.tm_year]
    pFrom = [ct.tm_hour, ct.tm_min//15*15, ct.tm_sec, ct.tm_mon, ct.tm_mday, ct.tm_year]
    pTo = time.localtime(time.mktime(ct) + 60*60*24)
    pTo = [pTo.tm_hour, pTo.tm_min//15*15, pTo.tm_sec, pTo.tm_mon, pTo.tm_mday, pTo.tm_year]
    data = head + body.format("figs/fig_{:n}_{:n}.png".format(ct.tm_hour, ct.tm_min//15*15), *(generated+pFrom+pTo))
    with open("page/page_{:n}_{:n}.html".format(ct.tm_hour, ct.tm_min//15*15), 'w') as page: page.write(data)

#set start
tstart = time.time()
fticks = 0
while True:
    ct = time.localtime()
    ch = ct.tm_hour
    cm = ct.tm_min//15
    #index 96 is the first sample of day 2 of dataset |[0:95]| = 96 15-minute points of day 1
    #to predict we will consider seq_length samples with the last sample being the 'current' point
    #so if ch == cm == 0 then we want index 96 to be the last in our sequence -> data[92:97] to start with
    #when ch==1 then data[93:98] is desired, this pattern is : data[96+(4*ch+cm)-4:96+(4*ch+cm)+1]
    X = np.reshape(Xt[96+(4*ch+cm)-(seq_length-1):96+(4*ch+cm)+1], (1,seq_length, input_size))
    #pdb.set_trace()
    yhat = model.predict(X, batch_size, 0)
    dates = [str(ch)+':'+str(cm*15)]
    prediction = [X[0, -1, 1]]
    #pdb.set_trace()
    for wstep in range(1, 97):#predict 96 timesteps forward
        ts = (ch*4+cm+wstep)
        hours = str((ts//4)%24)
        if len(hours) == 1: hours = '0'+hours
        minutes = str((ts%4)*15)
        if len(minutes) == 1: minutes = '00'
        dates.append(hours+":"+minutes)
        prediction.append(yhat[0,-1,0])
        X = X[0, 1:].tolist()
        nextT = X[-1][0]+0.15
        if ((nextT + 0.4) == ((nextT//1) + 1)): nextT+=0.4
        X.append([nextT if nextT<24 else 0.0, yhat[0, -1, 0]])
        X = np.reshape(X, (1,seq_length, input_size))
        yhat = model.predict(X, batch_size, 0)
    #pdb.set_trace()
    plotPrediction(dates, prediction, ct)
    generatePage(ct)
    fticks += 1
    print("Last tick time: " + time.ctime() + "\tElapsed ticks: " + str(fticks), end='\r')
    time.sleep((15*60.0) - ((time.time() - tstart) % (15*60.0)))