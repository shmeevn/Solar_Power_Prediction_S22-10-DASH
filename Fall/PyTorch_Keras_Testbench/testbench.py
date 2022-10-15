import RNN3 as mod1
import LSTM2 as mod2
import tb_util as tbu

import matplotlib.pyplot as plt #for plotting
import time                     #for measureing Time To Train and Time To Validate
import pdb                      #for debugging, and exiting to interactive shell

##Parameters
input_size = 2              #Number of input features
output_size = 1             #Number of output features
recurrent_units = 32        #Number of RNN/LSTM cells
seq_length = 5              #Number of input-vectors per sample
numEpochs = 20              #Number of training passes
batch_size = 1              #How many samples should be ran in parallel per training step
drop = 0.0                  #Dropout parameter
train_test_split = 0.8      #Training/Validation split percentage as decimal, 0.8 -> 80% training, 20% Validation
loss_function = 'mse'       #Loss function to use with the optimizer, 'mse' and 'mae' should work

#Load Dataset, specify columns by name, None means all
inputColumns = ['Time', 'PV_Power']
outputColumns = ['PV_Power']
Xt, Yt, Xv, Yv = tbu.loadData("Formatted_PV_data.csv", seq_length, train_test_split, inputColumns, outputColumns)

#for the first model
model1 = mod1.RNN(input_size, output_size, recurrent_units, 1, drop, loss_function)     #Get torch object
m1t1 = time.time()                                                                      #Time before training
trainHistory1 = model1.train_C(Xt, Yt, numEpochs, batch_size)                           #TrainHistory contains the losses per step
m1t2 = time.time()                                                                      #Time after training and before validating
valHistory1, predictions1, trueValues1 = model1.Validation(Xv, Yv, batch_size)          #valHistory is losses per step
m1t3 = time.time()                                                                      #Time after validating
m1skm = tbu.getSKmetrics(trueValues1, predictions1)                                     #Get tuple : (rmse, mae, r2) for the validation test

#for the second model
model2 = mod2.makeModel(input_size, seq_length, batch_size, recurrent_units, output_size, drop, loss_function)
m2t1 = time.time()
model2, khistory, trainHistory2 = mod2.trainModel(model2, Xt, Yt, Xv, Yv, numEpochs, batch_size)
m2t2 = time.time()
valLoss2, trueValues2, predictions2 = mod2.ValidationTest(model2, Xv, Yv, batch_size)
m2t3 = time.time()
m2skm = tbu.getSKmetrics(trueValues2, predictions2)

#Plot the losses during training
plt.style.use('dark_background')
fig1, ax1 = plt.subplots(2, 1, constrained_layout=True)
ax1[0].set_title("PyTorch Training Losses")
ax1[1].set_title("Keras Training Losses")
ax1[0].plot([x for x in range(len(trainHistory1))], trainHistory1, 'b-', label="Loss (Torch)")
ax1[1].plot([x for x in range(len(trainHistory2))], trainHistory2, 'b-', label="Loss (Keras)")
ax1[0].legend(loc='best')
ax1[1].legend(loc='best')

#Plot the validation test results
fig2, ax2 = plt.subplots(2, 1, constrained_layout=True)
ax2[0].set_title("PyTorch Pred vs. True")
ax2[1].set_title("Keras Pred vs. True")
XP = [ x for x in range(len(predictions1))]
ax2[0].plot(XP, predictions1, 'r-', label="Prediction")
ax2[0].plot(XP, trueValues1, 'g-', label="True Value")
ax2[1].plot(XP, predictions2, 'r-', label="Prediction")
ax2[1].plot(XP, trueValues2, 'g-', label="True Value")
ax2[0].legend(loc='best')
ax2[1].legend(loc='best')

#Print out the metrics for the validation test in a table, starting with Header
print("\n\tPerformance Metrics:\n\n\tTorch\t\tKeras")
#Timings
print("TTL:\t{:.4f}\t\t{:.4f}\nTTV:\t{:.4f}\t\t{:.4f}"\
.format(m1t2-m1t1, m2t2-m2t1, m1t3-m1t2, m2t3-m2t2))
#Metrics
print("RMSE:\t{:.4f}\t\t{:.4f}\nMAE:\t{:.4f}\t\t{:.4f}\nR2:\t{:.4f}\t\t{:.4f}"\
.format(*[x for y in zip(m1skm, m2skm) for x in y]), end="\n\n")
# ^^ This list comprehension zips and flattens the values from the getSKmetrics function
#Exit to interactive shell after printing prompt:
print("Use 'fig1.show()' to display training losses, and 'fig2.show()' to display validation test results\nUse 'exit()' when finised.\n")
pdb.set_trace()