import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop=0.0, loss = 'mse'):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_size
        self.layers = n_layers
        #rnn layer from torch
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, dropout=drop, batch_first=True)
        #last layer is fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        
        if loss == "mse":
            self.loss = nn.MSELoss(reduction='none')
        elif loss == "mae":
            self.loss = nn.L1Loss(reduction='none')
        else:
            self.loss = nn.MSELoss(reduction='none')
    
    #do forward propagation on self
    def forward(self, x, hidden):
        #x has shape (batch_size, seq_length, input_size)
        #hidden has shape (n_layers, batch_size, hidden_size)
        #r_out has shape (batch_size, seq_length, hidden_size)
        #output has shape (batch_size, seq_length, output_size)
        
        r_out, hidden = self.rnn(x, hidden)
        
        #get final output
        output = self.fc(r_out)
        return output, hidden
        
    #train the RNN
    def train_C(self, X, Y, epochs, batch_size=1, device=None, print_every=None):
        lossHistory = []
        steps = (X.shape[0]//batch_size) #(total number of sequences / number of sequences per batch) = number of training steps per epoch.
        if print_every == None: print_every = steps-1#print every epoch unless otherwise specified
        print('')#newline
        for epn in range(epochs):
            #Adam optimizer with a learning rate of 0.01
            optimizer = torch.optim.Adam(self.rnn.parameters(), lr=0.01)
            for batch_i in range(steps):
                hidden = torch.randn(self.layers, batch_size, self.hidden_dim)
                # convert data into Tensors
                x_tensor = torch.Tensor(X[(batch_i)*batch_size:(batch_i+1)*batch_size])#support batching
                y_tensor = torch.Tensor(Y[(batch_i)*batch_size:(batch_i+1)*batch_size])
                # outputs from the rnn
                prediction, hidden = self(x_tensor, hidden)
                #Detach hidden state to avoid tensorflow error
                hidden = hidden.data
                #loss
                criterion = nn.MSELoss(reduction='none') if self.loss == None else self.loss
                loss = criterion(prediction, y_tensor)
                lossHistory.append(torch.mean(loss.data))#save the mean loss across the batch
                # zero gradients
                optimizer.zero_grad()
                # perform backprop and update weights
                loss.sum().backward()#backprop with the sum of batch losses
                optimizer.step()
                # display loss and predictions
                if (batch_i%print_every == 0):
                    print("{:.2f}%".format(((batch_i+(epn*steps))/(steps*epochs))*100.0) + '\t' + str(batch_i+(epn*steps)) + '/' + str(steps*epochs) + '\t\t\t' + 'Loss: ' + str(np.array(torch.mean(loss.data))), end='\r')
        print('')#newline
        return lossHistory
    
    #Do validation test on model
    def Validation(self, X, Y, batch_size):
        lossHistory = []
        predictions = []
        trueValues = []
        steps = (X.shape[0]//batch_size)
        for batch_i in range(steps):#validate for all data in test set
            hidden = torch.randn(self.layers, batch_size, self.hidden_dim)
            # convert data into Tensors
            x_tensor = torch.Tensor(X[batch_i*batch_size:(batch_i+1)*batch_size])#support batching
            y_tensor = torch.Tensor(Y[batch_i*batch_size:(batch_i+1)*batch_size])
            
            # outputs from the rnn
            prediction, hidden = self(x_tensor, hidden)
            hidden = hidden.data
            # loss
            criterion = nn.MSELoss(reduction='none') if self.loss == None else self.loss
            # calculate the loss
            loss = criterion(prediction, y_tensor)
            
            lossHistory.append(torch.mean(loss.data))
            predictions.append(prediction.data.numpy().flatten())
            trueValues.append(y_tensor.data.numpy().flatten())
        return lossHistory, np.reshape(predictions, (-1)), np.reshape(trueValues, (-1))