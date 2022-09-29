import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pdb

class RNN(nn.Module):
    #Ctor
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        #rnn layer from torch
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers)
        #last layer is fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    #do forward propagation on self
    def forward(self, x, hidden):
        #x has shape (batch_size, seq_length, input_size)
        #hidden has shape (n_layers, batch_size, hidden_dim)
        #r_out has shape (batch_size, time_step, hidden_size)
        batch_size = x.size(0)
        #get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        #reshape output to shape (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)
        #get final output
        output = self.fc(r_out)
        
        return output, hidden

    #train the RNN
    def train(self, n_steps, print_every, seq_length = 20):
        #initialize the hidden state
        hidden = None
        for batch_i, step in enumerate(range(n_steps)):
            # defining the training data
            time_steps = np.linspace(step * np.pi, (step+1)*np.pi, seq_length + 1)
            data = np.sin(time_steps)
            data.resize((seq_length + 1, 1)) # input_size=1
            x = data[:-1]
            y = data[1:]
            # convert data into Tensors
            x_tensor = torch.Tensor(x).unsqueeze(0) # unsqueeze gives a 1, batch_size dimension
            y_tensor = torch.Tensor(y)
            # outputs from the rnn
            prediction, hidden = self(x_tensor, hidden)
            #Representing Memory
            # make a new variable for hidden and detach the hidden state from its history
            # this way, we don't backpropagate through the entire history
            hidden = hidden.data
            # calculate the loss
            loss = criterion(prediction, y_tensor)
            # zero gradients
            optimizer.zero_grad()
            # perform backprop and update weights
            loss.backward()
            optimizer.step()
            # display loss and predictions
            if batch_i%print_every == 0:
                print('Loss: ', loss.item())
                plt.plot(time_steps[1:], x, 'r.') # input
                #pdb.set_trace()
                plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.') #predictions
                plt.show()
        return rnn
        
    #train the RNN
    def train_C(self, X, Y, n_steps, print_every, seq_length = 1440):
        #initialize the hidden state
        hidden = None
        for batch_i, step in enumerate(range(n_steps)):
            # defining the training data
            time_steps = np.linspace(step * 24, (step+1)*24, seq_length + 1)
            #data = np.sin(time_steps)
            #data.resize((seq_length + 1, 1)) # input_size=1
            #x = data[:-1]
            #y = data[1:]
            # convert data into Tensors
            x_tensor = torch.Tensor(X[batch_i%X.shape[0]]).unsqueeze(0) # unsqueeze gives a 1, batch_size dimension
            y_tensor = torch.Tensor(Y[batch_i%X.shape[0]])
            # outputs from the rnn
            prediction, hidden = self(x_tensor, hidden)
            #Representing Memory
            # make a new variable for hidden and detach the hidden state from its history
            # this way, we don't backpropagate through the entire history
            hidden = hidden.data
            # MSE loss and Adam optimizer with a learning rate of 0.01
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
            # calculate the loss
            loss = criterion(prediction, y_tensor)
            # zero gradients
            optimizer.zero_grad()
            # perform backprop and update weights
            loss.backward()
            optimizer.step()
            # display loss and predictions
            if (batch_i%print_every == 0) and False:
                print('Loss: ', loss.item())
                #pdb.set_trace()
                plt.plot(time_steps[1:], [k[2] for k in X[batch_i%X.shape[0]]], 'r.', label='input, x') # input
                #pdb.set_trace()
                plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.', label='forecast, y') #predictions
                plt.legend(loc='best')
                #plt.show()
        return rnn
    
    def Validation(self, X, Y, print_every, seq_length = 1440):
        #initialize the hidden state
        hidden = None
        for batch_i, step in enumerate(range(X.shape[0])):
            # defining the val data
            time_steps = np.linspace(step * 24, (step+1) * 24, seq_length + 1)
            # convert data into Tensors
            x_tensor = torch.Tensor(X[batch_i%X.shape[0]]).unsqueeze(0) # unsqueeze gives a 1, batch_size dimension
            y_tensor = torch.Tensor(Y[batch_i%X.shape[0]])
            # outputs from the rnn
            prediction, hidden = self(x_tensor, hidden)
            hidden = hidden.data
            # MSE loss
            criterion = nn.MSELoss()
            # calculate the loss
            loss = criterion(prediction, y_tensor)
            # display loss and predictions
            if batch_i%print_every == 0:
                print('Loss: ', loss.item())
                #pdb.set_trace()
                plt.plot(time_steps[1:], [k[2] for k in X[batch_i%X.shape[0]]], 'r.', label='input, x') # input
                #pdb.set_trace()
                plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.', label='forecast, y') #predictions
                plt.legend(loc='best')
                plt.show()
        return rnn

def dimensionTest():
    test_rnn = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)
    #generate evenly spaced, test points
    time_steps = np.linspace(0, np.pi, seq_length)
    data = np.sin(time_steps)
    data.resize((seq_length, 1))
    
    test_input = torch.Tensor(data).unsqueeze(0) # give it a batch_size of 1 as first dimension
    print('Input size: ', test_input.size())
    # test out rnn sizes
    test_out, test_h = test_rnn(test_input, None)
    print('Output size: ', test_out.size())
    print('Hidden state size: ', test_h.size())

def fakeSineData():
    plt.figure(figsize=(8,5))
    seq_length = 20
    time_steps = np.linspace(0, np.pi, seq_length + 1)
    data = np.sin(time_steps)
    data.resize((seq_length + 1, 1))
    x = data[:-1]
    y = data[1:]
    #plot
    plt.plot(time_steps[1:], x, 'r.', label='input, x') # x
    plt.plot(time_steps[1:], y, 'b.', label='target, y') # y
    plt.legend(loc='best')
    plt.show()
    
    return x, y
    
def doTutorial():
    seq_length = 20
    fakeSineData()
    
    dimensionTest()
    
    # decide on hyperparameters
    input_size = 1
    output_size = 1
    hidden_dim = 32
    n_layers = 1
    
    # instantiate an RNN
    rnn = RNN(input_size, output_size, hidden_dim, n_layers)
    print(rnn)
    
    # MSE loss and Adam optimizer with a learning rate of 0.01
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
    
    #Train the rnn
    rnn.train(75, 15)#train for 75 steps, show progress every 15
    pdb.set_trace()


if __name__ == "__main__":
    plt.style.use('dark_background')
    data = []
    raw = []
    with open("household_power_consumption.txt", 'r') as file:
        raw = [x.split(';') for x in file][1:]
    for line in raw:
        if any([(x.find('?') is not -1) for x in line]):
            continue
        data.append([int(line[0].split('/')[1]), int(line[1].split(':')[0]), *[float(x) for x in line[2:]]])#get month, hour, and rest of data
    X = np.array(data[:-1])
    Y = np.array([x[2] for x in data[1:]])
    raw = None
    data = None
    seq_length = 1440
    
    lim = len(X)//seq_length*seq_length#2075260//1440*1440=2075040 
    #drop ~3 hours of data to make 1441 seqeuences of 24 hours
    X = X[:lim]
    Y = Y[:lim]
    
    # decide on hyperparameters
    input_size = 9
    output_size = 1
    hidden_dim = 32
    n_layers = 1
    X = X.reshape((-1, seq_length, input_size))
    Y = Y.reshape((-1, seq_length, output_size))
    vs = int(X.shape[0]*0.8)
    X_train = X[:vs]
    Y_train = Y[:vs]
    X_val = X[vs:]
    Y_val = Y[vs:]
    
    # instantiate an RNN
    rnn = RNN(input_size, output_size, hidden_dim, n_layers)
    print(rnn)
    
    #Train the rnn
    rnn.train_C(X_train, Y_train, X_train.shape[0]*4, X_train.shape[0]//5, seq_length)
    rnn.Validation(X_val, Y_val, X_val.shape[0]//5, seq_length)
    pdb.set_trace()
    
    
    