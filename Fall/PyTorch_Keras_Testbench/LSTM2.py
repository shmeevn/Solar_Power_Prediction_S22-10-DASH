#For validation test:
import numpy as np
from sklearn.metrics import mean_squared_error

#For the LSTM:
import keras
from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping

#This class is used so that the per-step training losses can be recorded for parity with pyTorch
class batch_loss_Callback(keras.callbacks.Callback):
    def __init__(self, metric):
        self.losses = []
        self.metric = metric
    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs[self.metric])

#This function instatiates the Keras LSTM model, using Keras Sequential
def makeModel(inputSize, seq_length, batchSize, recurrentUnits, outputSize, drop = 0.0, lossf = 'mse'):
    model = Sequential()#Keras has 3 methods of implementing models, Sequential, Functional, and subclassing
    #Sequential is used here as it was used in the kaggle sample, but the Functional API is fully-featured (MIMO architectures, complicated models)
    #The architecture used here was chosen to be 'comparable' to the PyTorch architecture, consisting of a reccurent layer, dropout, and a dense (Linear) output.
    model.add(Input(shape=(seq_length, inputSize), batch_size = batchSize))#The input layers defines the input shape
    model.add(LSTM(recurrentUnits, return_sequences=True))
    model.add(Dropout(drop))
    model.add(Dense(outputSize))
    #The adam optimizer is also used in the PyTorch model, Keras offers SGD, RMSprop, Adadelta, and some others in addition to adam
    #The optimizer can be configured by instantiating an optimizer object and passing parameters to its constructor,
    #   that object is then passed to the 'optimizer' argment of the compile() function below. Passing a string is a shortcut to using default parameters.
    model.compile(loss=lossf, optimizer='adam')
    return model

#This function trains the model
def trainModel(model, train_x, train_y, test_x, test_y, numEpochs, batchSize):
    #This is the list of callbacks to call during training
    cbl = [batch_loss_Callback('loss'), EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, restore_best_weights=True)]
    #These give the indices to truncate the train/test sets to avoid an error
    lim1 = (len(train_x)//batchSize)*batchSize
    lim2 = (len(test_x)//batchSize)*batchSize
    #This call trains the model
    history = model.fit(train_x[:lim1], train_y[:lim1], epochs=numEpochs, batch_size=batchSize, validation_data=(test_x[:lim2], test_y[:lim2]), verbose=2, shuffle=False, callbacks=cbl)
    return model, history, cbl[0].losses

#This function tests the model
def ValidationTest(model, test_x, test_y, batch_size):
    lim = (len(test_x)//batch_size)*batch_size
    pred = model.predict(test_x[:lim], batch_size)
    rmse = [ mean_squared_error(test_y[i], pred[i]) for i in range(len(pred))]
    return rmse, np.reshape(test_y[:lim], (-1)), np.reshape(pred, (-1))