from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

inputsize = 8      #number of features(inputs)
seed = 7
np.random.seed(seed)

def generate_model(inputsize, LR, HLS, BS,ES)
	filename='pimadiabetes.csv'
	#skips first row, avoids conversion of string to float error
	dataset=dataset=np.loadtxt(filename, delimiter=",",skiprows=1)
	#X & Y
	features = dataset[:,0:inputsize]
	dkpts = dataset[:,inputsize]
	print ("Feature size: " + str(features.shape))
	print ("Output size: " + str(dkpts.shape))


	#PARAMETERS


	#NOTE TO SELF: DON'T FORGET TO ADD A COLUMN OF 1'S THAT WILL REPRESENT THE BIAS NEURON

	Network = Sequential()
	Network.add(Dense(HLS,  input_dim=inputsize, init='uniform', activation='relu'))
	Network.add(Dense(1, init='uniform', activation ='relu'))
	Network.compile(loss='binary_crossentropy', optimizer ='sgd', metrics=['accuracy', 'mean_squared_error'])
	Network.fit(features, dkpts, nb_epoch = ES, batch_size=BS, verbose=2)  #verbose = 0 to stop data output
	scores = Network.evaluate(features, dkpts)
	print("%s: %.2f%%" % (Network.metrics_names[1], scores[1]*100))

inputsize=8
LR = .1
HLS = 30
BS = 10
ES = 150
generate_model(inputsize, LR, HLS, BS, ES)