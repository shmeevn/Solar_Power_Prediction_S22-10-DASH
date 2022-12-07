from sklearn import metrics as skm
import pandas as pd
import numpy as np

def getSKmetrics(yTrue, yPred):#helper function to return tuple (rmse, mae, r2)
    return (np.sqrt(skm.mean_squared_error(yTrue, yPred)),
           skm.mean_absolute_error(yTrue, yPred),
           skm.r2_score(yTrue, yPred))

def loadData(filepath, seq_length, vsp, iFeatLst = None, oFeatLst = None):
    data1 = pd.read_csv(filepath, sep=',')#read file into dataframe
    if iFeatLst != None:#Resolve Column(feature) name -> index after drop
        iFeatLst = [data1.columns.tolist().index(x) for x in iFeatLst if any([y==x for y in data1.columns])]
    if oFeatLst != None:#Also resolve output
        oFeatLst = [data1.columns.tolist().index(x) for x in oFeatLst if any([y==x for y in data1.columns])]
    data = np.array(data1)#make numpy array from dataframe (index-able)
    X = data[:-1] if iFeatLst == None else data[:-1, iFeatLst]#only use desired features
    Y = data[1:] if oFeatLst == None else data[1:, oFeatLst]#if ==None then use all features
    seq_lim = (X.shape[0]//seq_length)*seq_length#find greatest index which is divisible by seq_length
    X = X[:seq_lim]#truncate data to fit in sequences
    Y = Y[:seq_lim]
    X = X.reshape((-1, seq_length, X.shape[1]))#reshape data into N samples of seq_length input-vectors of length input_size
    Y = Y.reshape((-1, seq_length, Y.shape[1]))
    vs = int(X.shape[0]*vsp)#split data into training and validation
    X = np.array(X, dtype=float)#enforce proper data type for np array
    Y = np.array(Y, dtype=float)
    return X[:vs], Y[:vs], X[vs:], Y[vs:], X, Y, data1

