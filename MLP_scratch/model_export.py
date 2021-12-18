import pickle
import numpy as np
import os

def export_linear(x, weight, bias):
    z = x @ weight + bias #can change this to a RELU function instead too
    return sigmoid(z)

def sigmoid(x):
    return 1/(1+np.exp(-x))


#weights
model_weights = pickle.load(open(os.getcwd() + "/MLP_scratch/model_weights.pickle", "rb" ))

fc0_weight = model_weights[0]
fc1_weight = model_weights[1]
fc2_weight = model_weights[2]
fc3_weight = model_weights[3]
fc4_weight = model_weights[4]
fc0_bias = model_weights[5]
fc1_bias = model_weights[6]
fc2_bias = model_weights[7]
fc3_bias = model_weights[8]
fc4_bias = model_weights[9]


def model(input_string):
    #input_string = str(np.array(raw_data_test.iloc[0,1:]).tolist()) #string input '[1, 2, ..., 3]' 784 elements
    x = np.fromstring(input_string[1:-1],sep=',').astype(int)/255
    x = export_linear(x, fc0_weight, fc0_bias) 
    x = export_linear(x, fc1_weight, fc1_bias)
    x = export_linear(x, fc2_weight, fc2_bias)
    x = export_linear(x, fc3_weight, fc3_bias)
    x = export_linear(x, fc4_weight, fc4_bias)
    return prediction = np.argmax(x)

print(model(input_string))