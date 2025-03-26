import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Attention
from tensorflow.keras.models import Model
from tensorflow.keras import layers

import numpy as np
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
import pandas as pd
import os
import random as rn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from matplotlib import pyplot
from keras import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Input
from keras.layers import Reshape


# # ==== fix random seed for reproducibility =====
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(10)
rn.seed(1)
tf.random.set_seed(100)

import time
start_time = time.time()

def evaluation_forecasts (xTest, yTest, model1, model2, model3, model4, model5, model6, model7):
    yhat1 = model1.predict(xTest[0]).reshape(-1,1,1)
    xTest[1,:,-1:,-1] = yhat1[:,:,0]
    yhat2 = model2.predict(xTest[1]).reshape(-1,1,1)
    xTest[2,:,-2:-1,-1] = yhat1[:,:,0]
    xTest[2,:,-1:,-1] = yhat2[:,:,0]
    yhat3 = model3.predict(xTest[2]).reshape(-1,1,1)
    xTest[3,:,-3:-2,-1] = yhat1[:,:,0]
    xTest[3,:,-2:-1,-1] = yhat2[:,:,0]
    xTest[3,:,-1:,-1] = yhat3[:,:,0]
    yhat4 = model4.predict(xTest[3]).reshape(-1,1,1)
    xTest[4,:,-4:-3,-1] = yhat1[:,:,0]
    xTest[4,:,-3:-2,-1] = yhat2[:,:,0]
    xTest[4,:,-2:-1,-1] = yhat3[:,:,0]
    xTest[4,:,-1:,-1] = yhat4[:,:,0]
    yhat5 = model5.predict(xTest[4]).reshape(-1,1,1)
    xTest[5,:,-5:-4,-1] = yhat1[:,:,0]
    xTest[5,:,-4:-3,-1] = yhat2[:,:,0]
    xTest[5,:,-3:-2,-1] = yhat3[:,:,0]
    xTest[5,:,-2:-1,-1] = yhat4[:,:,0]
    xTest[5,:,-1:,-1] = yhat5[:,:,0]
    yhat6 = model6.predict(xTest[5]).reshape(-1,1,1)
    xTest[6,:,-6:-5,-1] = yhat1[:,:,0]
    xTest[6,:,-5:-4,-1] = yhat2[:,:,0]
    xTest[6,:,-4:-3,-1] = yhat3[:,:,0]
    xTest[6,:,-3:-2,-1] = yhat4[:,:,0]
    xTest[6,:,-2:-1,-1] = yhat5[:,:,0]
    xTest[6,:,-1:,-1] = yhat6[:,:,0]
    yhat7 = model7.predict(xTest[6]).reshape(-1,1)
    array_list = [yhat1.reshape(-1,1), yhat2.reshape(-1,1), yhat3.reshape(-1,1), yhat4.reshape(-1,1), yhat5.reshape(-1,1), yhat6.reshape(-1,1), yhat7]
    yhat = np.concatenate (array_list, axis=1)

    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
    ypred= scalery.inverse_transform(yhat)
    r2_test = r2_score(ori_yTest,ypred)
    print ("Total r2 score {}".format(r2_test))

    test_scores =[]
    for i in range (yTest.shape[1]):
        nse_test = r2_score( ori_yTest[:,i], ypred[:,i])
        print ("The day {} test score {}".format(i, nse_test))
        test_scores.append(nse_test)

    return r2_test, test_scores

def evaluation_forecasts_new (xTest, yTest, model):
    yhat = model.predict(xTest, verbose=1)
    r2_test = r2_score(yTest[:,Day_shap],yhat)
    print ("Total r2 score {}".format(r2_test))
    return r2_test

def split_dataset(data, Ntrain, nfuture, res):
    # split into standard weeks
    train, test = data[:Ntrain], data[Ntrain: -res]
    # restructure into windows of weekly data
    train = array(split(train, len(train)/nfuture))
    test = array(split(test, len(test)/nfuture))
    print (" train shape: ,  test.shape: ",train.shape, test.shape)
    return train, test    

# convert history into inputs and outputs
def convert_supervised(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :-1])
            y.append(data[in_end:out_end, -1])
        # move along one time step
        in_start += 1

    return np.asarray(X).astype(np.float32), np.asarray(y).astype(np.float32)

def convert_xTrain(train, n_input, n_out=7):
    # flatten data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))

    X1, X2, X3, X4, X5, X6, X7 = list(), list(), list(), list(), list(), list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X1.append(data[in_start:in_end, :-1])
            X2.append(data[in_start+1:in_end+1, :-1])
            X3.append(data[in_start+2:in_end+2, :-1])
            X4.append(data[in_start+3:in_end+3, :-1])
            X5.append(data[in_start+4:in_end+4, :-1])
            X6.append(data[in_start+5:in_end+5, :-1])
            X7.append(data[in_start+6:in_end+6, :-1])
        # move along one time step
        in_start += 1
    variables = [X1, X2, X3, X4, X5, X6, X7]
    variables = [np.asarray(var).astype(np.float32) for var in variables]
    X1, X2, X3, X4, X5, X6, X7 = variables
    return X1, X2, X3, X4, X5, X6, X7

def load_inflow_timeseries():

        rawData_np = np.loadtxt(os.path.join(data_dir, data_name + '.dat'), delimiter=',')

        print ("-- Data name:", data_name)
        print ("-- Raw data shape: {}".format(rawData_np.shape))
        print ("-- Num of inputs: {}".format(ninputs))

        #split into train and test
        train, test = split_dataset(rawData_np, Ntrain, nfuture, res)
        xTrain, yTrain = convert_supervised(train, ndays, nfuture)
        x1,x2,x3,x4,x5,x6,x7 = convert_xTrain(train, ndays, nfuture)
        variables = [x1, x2, x3, x4, x5, x6, x7]
        variables = [var.reshape((var.shape[0], -1)) for var in variables]
        x1, x2, x3, x4, x5, x6, x7 = variables

        history = [x for x in train]
        xTest1, xTest2, xTest3, xTest4, xTest5, xTest6, xTest7 = [], [], [], [], [], [], []
        for i in range (len(test)):
            history.append(test[i,:])
            data = array(history)
            data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
            xTest1.append (data[-(ndays+nfuture-0):-7, :ninputs])            
            xTest2.append (data[-(ndays+nfuture-1):-6, :ninputs])
            xTest3.append (data[-(ndays+nfuture-2):-5, :ninputs])
            xTest4.append (data[-(ndays+nfuture-3):-4, :ninputs])
            xTest5.append (data[-(ndays+nfuture-4):-3, :ninputs])
            xTest6.append (data[-(ndays+nfuture-5):-2, :ninputs])
            xTest7.append (data[-(ndays+nfuture-6):-1, :ninputs])



        variables = [xTest1, xTest2, xTest3, xTest4, xTest5, xTest6, xTest7]
        variables = [array(var).reshape((array(var).shape[0],-1)) for var in variables]
        xTest1, xTest2, xTest3, xTest4, xTest5, xTest6, xTest7 = variables

        yTest = test[:,:,-1]

        print ('--- x1 shape: {}, yTrain shape: {}'.format(x1.shape, yTrain.shape))
        print ('--- xTest1 shape: {}, yTest shape: {}'.format(xTest1.shape, yTest.shape))

        #### to float32
        variables = [x1, x2, x3, x4, x5, x6, x7]
        variables = [var.astype(np.float32) for var in variables]
        x1, x2, x3, x4, x5, x6, x7 = variables
        yTrain = yTrain.astype(np.float32)

        variables = [xTest1, xTest2, xTest3, xTest4, xTest5, xTest6, xTest7]
        variables = [var.astype(np.float32) for var in variables]
        xTest1, xTest2, xTest3, xTest4, xTest5, xTest6, xTest7 = variables
        yTest = yTest.astype(np.float32)

        ### keep the original y data without scaling
        ori_yTrain = yTrain   
        ori_yTest = yTest

        # ---scale training data both X and y
        #scalerx = StandardScaler()
        scalerx = MinMaxScaler(feature_range=(0, 1))
        x1 = scalerx.fit_transform(x1)
        variables = [x2, x3, x4, x5, x6, x7]
        variables = [scalerx.transform(var) for var in variables]
        x2, x3, x4, x5, x6, x7 = variables
        variables = [xTest1, xTest2, xTest3, xTest4, xTest5, xTest6, xTest7]
        variables = [scalerx.transform(var) for var in variables]
        xTest1, xTest2, xTest3, xTest4, xTest5, xTest6, xTest7 = variables

        #scalery = StandardScaler() 
        scalery = MinMaxScaler(feature_range=(0,1))
        yTrain = scalery.fit_transform(yTrain)
        yTest  = scalery.transform(yTest)

        # reshape input to 3D [samples, timesteps, features]        
        variables = [x1, x2, x3, x4, x5, x6, x7]
        variables = [var.reshape((1, var.shape[0], ndays, ninputs)) for var in variables]
        x1, x2, x3, x4, x5, x6, x7 = variables
        xTrain = np.concatenate((x1, x2, x3, x4, x5, x6, x7), axis=0)

        variables = [xTest1, xTest2, xTest3, xTest4, xTest5, xTest6, xTest7]
        variables = [var.reshape((1, var.shape[0], ndays, ninputs)) for var in variables]
        xTest1, xTest2, xTest3, xTest4, xTest5, xTest6, xTest7 = variables
        xTest = np.concatenate((xTest1, xTest2, xTest3, xTest4, xTest5, xTest6, xTest7), axis=0)

        yTrain = yTrain.reshape((yTrain.shape[0], nfuture, 1))
        yTest  = yTest.reshape((yTest.shape[0], nfuture, 1))

        # yTrain = np.repeat(yTrain, nfuture, axis=0)
        # yTest = np.repeat(yTest, nfuture, axis=0)

        print ('--- xTrain shape: {}, yTrain shape: {}'.format(xTrain.shape, yTrain.shape))
        print ('--- xTest shape: {}, yTest shape: {}'.format(xTest.shape, yTest.shape))

        return xTrain, xTest, yTrain, yTest, ninputs, nfuture, scalerx, scalery, ori_yTrain, ori_yTest


data_dir = 'Data'
data_name = 'CO00004' 

ndays = 30
nfuture =  7
ninputs =  4
Ntrain = 10745
res = 2

xTrain, xTest, yTrain, yTest, ninputs, nfuture, scalerx, scalery, ori_yTrain, ori_yTest = load_inflow_timeseries()


opt = tf.keras.optimizers.Adam(learning_rate=0.002)

def build_model(node):
    # define parameters
    n_timesteps, n_features, n_outputs = ndays, ninputs, 1
    # define model
    model = Sequential()
    model.add(LSTM(node, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(node, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(node/2, activation='relu')))
    # model.add(TimeDistributed(Dense(10, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer=opt)
    return model

def build_model1(node):
    # define parameters
    n_timesteps, n_features, n_outputs = ndays, ninputs, 1
    # define model
    model = Sequential()
    model.add(LSTM(node, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(node, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(node/2, activation='relu')))
    model.add(TimeDistributed(Dense(10, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer=opt)
    return model

def build_model2(node):
    # define parameters
    n_timesteps, n_features, n_outputs = ndays, ninputs, 1
    # define model
    model = Sequential()
    model.add(LSTM(node, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=opt)
    #model.summary()
    return model


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

node1 = 200
verbose = 1
epochs  = 100
batch_size = 64
epochs2 = 20
epochs3 = 30

model1 = build_model(node1)
model1.fit(xTrain[0], yTrain[:,:1,:], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(xTest[0], yTest[:,:1,:]), callbacks=[early_stopping])
model1.save_weights(filepath='final_weight.h5')
y1 = model1.predict(xTrain[0]).reshape(-1,1,1)

model2 = build_model(node1)
model2.load_weights("final_weight.h5")
xTrain[1,:,-1:,-1] = y1[:,:,0]
model2.fit(xTrain[1], yTrain[:,1:2,:], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(xTest[1], yTest[:,1:2,:]), callbacks=[early_stopping])
model2.save_weights(filepath='final_weight1.h5')
y2 = model2.predict(xTrain[1]).reshape(-1,1,1)
xTrain[2,:,-2:-1,-1] = y1[:,:,0]
xTrain[2,:,-1:,-1] = y2[:,:,0]

model3 = build_model(node1)
model3.load_weights("final_weight1.h5")
model3.fit(xTrain[2], yTrain[:,2:3,:], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(xTest[2], yTest[:,2:3,:]), callbacks=[early_stopping])
model3.save_weights(filepath='final_weight1.h5')
y3 = model3.predict(xTrain[2]).reshape(-1,1,1)
xTrain[3,:,-3:-2,-1] = y1[:,:,0]
xTrain[3,:,-2:-1,-1] = y2[:,:,0]
xTrain[3,:,-1:,-1] = y3[:,:,0]

model4 = build_model(node1)
model4.load_weights("final_weight1.h5")
model4.fit(xTrain[3], yTrain[:,3:4,:], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(xTest[3], yTest[:,3:4,:]), callbacks=[early_stopping])
model4.save_weights(filepath='final_weight1.h5')
y4 = model4.predict(xTrain[3]).reshape(-1,1,1)
xTrain[4,:,-4:-3,-1] = y1[:,:,0]
xTrain[4,:,-3:-2,-1] = y2[:,:,0]
xTrain[4,:,-2:-1,-1] = y3[:,:,0]
xTrain[4,:,-1:,-1] = y4[:,:,0]

model5 = build_model(node1)
model5.load_weights("final_weight1.h5")
model5.fit(xTrain[4], yTrain[:,4:5,:], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(xTest[4], yTest[:,4:5,:]), callbacks=[early_stopping])
model5.save_weights(filepath='final_weight1.h5')
y5 = model5.predict(xTrain[4]).reshape(-1,1,1)
xTrain[5,:,-5:-4,-1] = y1[:,:,0]
xTrain[5,:,-4:-3,-1] = y2[:,:,0]
xTrain[5,:,-3:-2,-1] = y3[:,:,0]
xTrain[5,:,-2:-1,-1] = y4[:,:,0]
xTrain[5,:,-1:,-1] = y5[:,:,0]


model6 = build_model(node1)
model6.load_weights("final_weight1.h5")
model6.fit(xTrain[5], yTrain[:,5:6,:], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(xTest[5], yTest[:,5:6,:]), callbacks=[early_stopping])
model6.save_weights(filepath='final_weight1.h5')
y6 = model6.predict(xTrain[5]).reshape(-1,1,1)
xTrain[6,:,-6:-5,-1] = y1[:,:,0]
xTrain[6,:,-5:-4,-1] = y2[:,:,0]
xTrain[6,:,-4:-3,-1] = y3[:,:,0]
xTrain[6,:,-3:-2,-1] = y4[:,:,0]
xTrain[6,:,-2:-1,-1] = y5[:,:,0]
xTrain[6,:,-1:,-1] = y6[:,:,0]

model7 = build_model(node1)
model7.load_weights("final_weight1.h5")
model7.fit(xTrain[6], yTrain[:,6:7,:], epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(xTest[6], yTest[:,6:7,:]), callbacks=[early_stopping])

r2_test, test_scores = evaluation_forecasts(xTest, yTest, model1, model2, model3, model4, model5, model6, model7)
print ("Overall score", r2_test)
print (test_scores)



from Path_explain import PathExplainerTF 
batch_size = 16
num_samples = xTest[0].shape[0]

def interpret_ig(model, inputs, baseline):
    x_test = inputs

    explainer = PathExplainerTF (model)

    gradients = np.zeros ((x_test.shape[0], x_test.shape[1], x_test.shape[2] ))
    batch_samples = x_test
    
    attributions = explainer.attributions(inputs=batch_samples,
                                              baseline=baseline,
                                              batch_size=batch_size,
                                              num_samples=num_samples,
                                              use_expectation=True,
                                              output_indices=None,
                                              verbose=True)

    return attributions

# Compute the integrated gradients for the input sequence
ig1 = interpret_ig(model=model1, inputs=xTest[0], baseline=xTrain[0])

yhat1 = model1.predict(xTest[0]).reshape(-1,1,1)
xTest[1,:,-1:,-1] = yhat1[:,:,0]
ig2 = interpret_ig(model=model2, inputs=xTest[1], baseline=xTrain[1])

yhat2 = model2.predict(xTest[1]).reshape(-1,1,1)
xTest[2,:,-2:-1,-1] = yhat1[:,:,0]
xTest[2,:,-1:,-1] = yhat2[:,:,0]
ig3 = interpret_ig(model=model3, inputs=xTest[2], baseline=xTrain[2])

yhat3 = model3.predict(xTest[2]).reshape(-1,1,1)
xTest[3,:,-3:-2,-1] = yhat1[:,:,0]
xTest[3,:,-2:-1,-1] = yhat2[:,:,0]
xTest[3,:,-1:,-1] = yhat3[:,:,0]
ig4 = interpret_ig(model=model4, inputs=xTest[3], baseline=xTrain[3])

yhat4 = model4.predict(xTest[3]).reshape(-1,1,1)
xTest[4,:,-4:-3,-1] = yhat1[:,:,0]
xTest[4,:,-3:-2,-1] = yhat2[:,:,0]
xTest[4,:,-2:-1,-1] = yhat3[:,:,0]
xTest[4,:,-1:,-1] = yhat4[:,:,0]
ig5 = interpret_ig(model=model5, inputs=xTest[4], baseline=xTrain[4])

yhat5 = model5.predict(xTest[4]).reshape(-1,1,1)
xTest[5,:,-5:-4,-1] = yhat1[:,:,0]
xTest[5,:,-4:-3,-1] = yhat2[:,:,0]
xTest[5,:,-3:-2,-1] = yhat3[:,:,0]
xTest[5,:,-2:-1,-1] = yhat4[:,:,0]
xTest[5,:,-1:,-1] = yhat5[:,:,0]
ig6 = interpret_ig(model=model6, inputs=xTest[5], baseline=xTrain[5])

yhat6 = model6.predict(xTest[5]).reshape(-1,1,1)
xTest[6,:,-6:-5,-1] = yhat1[:,:,0]
xTest[6,:,-5:-4,-1] = yhat2[:,:,0]
xTest[6,:,-4:-3,-1] = yhat3[:,:,0]
xTest[6,:,-3:-2,-1] = yhat4[:,:,0]
xTest[6,:,-2:-1,-1] = yhat5[:,:,0]
xTest[6,:,-1:,-1] = yhat6[:,:,0]
ig7 = interpret_ig(model=model7, inputs=xTest[6], baseline=xTrain[6])

print (ig1.shape, ig2.shape, ig3.shape, ig4.shape, ig5.shape, ig6.shape, ig7.shape)

ig_arrays = [ig1, ig2, ig3, ig4, ig5, ig6, ig7]
integrated_grads = np.concatenate(ig_arrays, axis=0)
print (integrated_grads.shape)
np.save(f'IG_data/IG_path_{data_name}.npy', integrated_grads)




