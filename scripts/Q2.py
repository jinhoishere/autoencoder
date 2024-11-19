# Code for Question 2

import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt
from simpleAutoencoder import AE
import torch.utils.data as Data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score

# Load training and test dataset
train_set = np.loadtxt('../data/MNISTTrData.csv', delimiter=',')
# print("Shape of train_set:", train_set.shape) # (1000, 784)
train_label = np.loadtxt('../data/MNISTTrLabels.csv') # Labels
# print("Shape of train_label:", train_label.shape) # (1000, )
test_set = np.loadtxt('../data/MNISTTestData.csv', delimiter=',')
# print("Shape of test_set:", test_set.shape) # (200, 784)
test_label = np.loadtxt('../data/MNISTTestLabels.csv', delimiter=',')
# print("Shape of test_label:", test_label.shape) # (200, )

# hyper-parameters for Adam optimizer
num_epochs = 200 # run the experiment 10 times means num_epochs = 10?
miniBatch_size = 64
learning_rate = 0.001
cudaDeviceId = None
accuracies = []

for i in range(10):
    print(f"\nExperiment: [{i+1} / 10]")
    # print("Start training the model with train_set...")
    
    # create a model and train it 
    model = AE() 
    model.fit(train_set, learning_rate, num_epochs, miniBatch_size, cudaDeviceId)
    # print("Training model with train_set has been finished...")

    # With the trained model, get bottlenecks for train_set and test_set
    # print("\nCreating 2 bottlenecks of the trained model with train_set and test_set...")
    bottleneck_train = model.get_bottleneck(torch.from_numpy(train_set).float())
    # print("The shape of bottleneck_train is", bottleneck_train.shape)
    bottleneck_test = model.get_bottleneck(torch.from_numpy(test_set).float())
    # print("The shape of bottleneck_test is", bottleneck_test.shape)

    # Calculate KNN where k = 5
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    # print("\nCreating 5NN with train_set...")
    five_NN = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    five_NN.fit(bottleneck_train, train_label)
    # print("Predicting test labels by applying the created 5NN to test_set...")
    predicted_test_label = five_NN.predict(bottleneck_test)
    
    # Look over the first 5 elements from predicted_test_label
    # for i in range(5):
        # print(predicted_test_label[i])

    # Calculate accuracy = (True Positive + True Negative) / Total # of predictions
    # https://scikit-learn.org/dev/modules/generated/sklearn.metrics.accuracy_score.html
    # print("\nCalculating accuracy...")
    accuracy = accuracy_score(test_label, predicted_test_label)
    accuracies.append(accuracy)
    # print("Accuracy:", accuracy * 100,"%")

    # Display confusion matrix
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(test_label, predicted_test_label))
    # ConfusionMatrixDisplay.from_predictions(test_label, predicted_test_label)
    # plt.show()

    # Plot a graph that shows losses of the model
    # plt.figure()
    # print("\nDisplay losses:")
    # plt.plot(model.losses)
    # plt.show()

print("\n------------------------------<Result>------------------------------")
print("Accuracy:", accuracies)
print("Mean in accuracies:", "%.2f"%np.mean(accuracies))
print("Standard deviation in accuracies:", "%.2f"%np.std(accuracies))



