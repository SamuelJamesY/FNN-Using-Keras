'''
keras neural network for diabetes pima dataset, predict the data 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data():
	'''
	Change column names and scale the data 
	'''
	df = pd.read_csv('diabetes.csv')
	df.columns = ['preg','glucose','bloodpressure','skinthickness','insulin','bmi','bpf','age','diabetes']
	scaler = MinMaxScaler()
	df.iloc[:,:-1] = scaler.fit_transform(df.iloc[:,:-1])
	return df

def train_test_split_data(df,expnum):
	'''
	Split the data into train and test sets
	'''
	X = df.iloc[:,:-1].to_numpy()
	y = df.iloc[:,-1].to_numpy()
	xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=expnum)
	return xtrain,xtest,ytrain,ytest

def neural_network_architecture():
	'''
	build a network with two hidden layers and an output of either 0 or 1
	eight input dimensions correponding to each column of data set
	'''
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(30,input_dim=8,activation='relu'),
		tf.keras.layers.Dense(10,activation='relu'),
		tf.keras.layers.Dense(1,activation='sigmoid')
	])
	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
	return model

def train_neural_network(model,xtrain,xtest,ytrain,ytest):
	'''
	fit and evaluate the neural network model
	'''
	history = model.fit(xtrain,ytrain,validation_data=(xtest,ytest), epochs=120,verbose=0)
	train_loss,train_acc = model.evaluate(xtrain,ytrain,verbose=0)
	test_loss,test_acc = model.evaluate(xtest,ytest,verbose=0)
	return model,history,train_acc,test_acc

def plot_train_test(history):
	'''
	Plot the train and test accuracy over time
	'''
	plt.plot(history.history['accuracy'], label='train')
	plt.plot(history.history['val_accuracy'], label='test')
	plt.legend()
	plt.savefig('nodp.png')
	plt.clf()

def main():
	df = load_data()
	train_acc_lst = np.empty(10)
	test_acc_lst = np.empty(10)
	for exp in range(10):
		xtrain,xtest,ytrain,ytest = train_test_split_data(df,exp)
		model = neural_network_architecture()
		model,history,train_acc_lst[exp],test_acc_lst[exp] = train_neural_network(model,xtrain,xtest,ytrain,ytest)
	print(train_acc_lst,'Training accuracy 10 experinments')
	print(test_acc_lst,'Test accuracy 10 experinments')

if __name__ == '__main__':
	main()