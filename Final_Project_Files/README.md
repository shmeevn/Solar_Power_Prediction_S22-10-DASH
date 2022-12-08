This folder contains all files and the data set used for the finalized version of the S22-10-DASH Senior Design project

To run the code, run the testbench.py file with two arguments. The first argument should be save or load depending on if you want to train a new model and save it, or load a previously trained model. The second argument should be weather or noweather depending on if you want to use the dataset with weather features or not.

An example of this argument string is this:

python testbench.py save weather

This command will train a new model and save it based on the weather parameters along with time and PV power.
