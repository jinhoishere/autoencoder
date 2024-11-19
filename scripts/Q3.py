import pdb
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from simpleANNClassifierPyTorch import NeuralNet


# Load training and test dataset
train_set = np.loadtxt('../data/MNISTTrData.csv', delimiter=',')
# print("Shape of train_set:", train_set.shape) # (1000, 784)
train_label = np.loadtxt('../data/MNISTTrLabels.csv')
# print("Shape of train_label:", train_label.shape) # (1000, )
test_set = np.loadtxt('../data/MNISTTestData.csv', delimiter=',')
# print("Shape of test_set:", test_set.shape) # (200, 784)
test_label = np.loadtxt('../data/MNISTTestLabels.csv', delimiter=',')
# print("Shape of test_label:", test_label.shape) # (200, )

# Create one hidden_layer
input_size = train_set.shape[1] # 784
# print(f"Input size: {input_size}")

# Set hyperparameters for the model
hidden_layers = [10, 20, 40, 80, 160, 320, 640, 1280] # the number of neurons
error_rates = [] # the error rates for each hidden layer
num_classes = 10
predicted_test_label = ""

# Run the experiment 10 times
for current_hidden_layer in hidden_layers:
    print(f"\n-> Current # of neurons: {current_hidden_layer}")
    for i in range(10):
        print(f"Experiment: [{i+1} / 10]")

        # Create my NN model
        my_NN = NeuralNet(input_size, [current_hidden_layer], num_classes)
        # print(my_NN)

        # Train my NN model
        my_NN.fit(train_set, train_label, standardizeFlag=True, 
                batchSize=16, optimizationFunc='Adam', learningRate=0.001,
                numEpochs=100, cudaDeviceId=None, verbose=True)

        # Predict the test_label by applying the trained model to test_set
        predicted_value, predicted_test_label = my_NN.predict(test_set)

        # Compute the error for the current layer(the number of neurons)
        predicted_test_label = predicted_test_label.flatten() # (200,)
        num_of_errors = 0
        for i in range(len(test_label)):
            if predicted_test_label[i] != test_label[i]:
                num_of_errors += 1
    
    # Compute the current layer's error rate
    error_rate = num_of_errors / len(test_label)
    error_rates.append(error_rate)
    print(f"The current number of neuron({current_hidden_layer})'s error rate is {error_rate}")

# Final result
print("\n------------------------------<Result>------------------------------")
avg_error_rate = num_of_errors / len(test_label)
print("The average error rate is", avg_error_rate)
plt.figure()
plt.plot(hidden_layers, error_rates)
plt.xlabel("The number of neurons in one hidden layer")
plt.ylabel("Error Rate")
plt.title("Q3. How the number of neurons affect on the misclassification rate")
plt.show()