# Code for Question 1

import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt
from simpleAutoencoder import AE
import torch.utils.data as Data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load train_set data
train_set = np.loadtxt('../data/MNISTTrData.csv', delimiter=',')
print("Shape of train_set:", train_set.shape) # (1000, 784)
train_label = np.loadtxt('../data/MNISTTrLabels.csv') # Labels
print("Shape of train_label:", train_label.shape) # (1000, )

# hyper-parameters for Adam optimizer
num_epochs = 200
miniBatch_size = 64
learning_rate = 0.001
cudaDeviceId = None

# create a model and train it 
print("\nStart training the model with train_set...")
model = AE() 
model.fit(train_set, learning_rate, num_epochs, miniBatch_size, cudaDeviceId)
print("Training model with train_set has been finished...\n")


# Question 1 - 2D plotting (Line 37 - 105)
# -> Remove lines 37 and 104 in <Q1.py>
# -> Uncomment lines 18, 51, and 52 in <simpleAutoencoder.py>
"""
# Autoencoder in 2-D: Run the trained model with one hidden layer
print("Encoding from the input layer to the bottleneck layer(2D)...")
latent_space = model.get_bottleneck(torch.from_numpy(train_set).float())
latent_space = latent_space.detach().numpy()
print("The shape of latent space(2D):", latent_space.shape) # should be (1000, 2)
print("The encoding has been finished...\n")

# PCA in 2-D:
print("Performing PCA(2D) on the original training dataset...")
mu = np.mean(train_set, axis=0)
train_set = train_set - mu
train_set = train_set.T
eVals, eVecs = np.linalg.eigh(np.matmul(train_set, train_set.T))
eVals_flip, eVecs_flip = np.flip(eVals), np.flip(eVecs, axis=1)
projectedData_2D = np.dot(train_set.T, eVecs_flip[:, 0:2])
print("The shape of projectedData is", projectedData_2D.shape)
print("End performing PCA...\n")

##################################################################
print("Drawing 2-D scatter plots of bottleneck and PCA...")
fig = plt.figure("Q1.2-D scattor plots")

# Autoencoder in 2-D: Draw a 2-D scatter plot
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(latent_space[0:99, 0], latent_space[0:99, 1], label = "0")
ax1.scatter(latent_space[100:199, 0], latent_space[100:199, 1], label = "1")
ax1.scatter(latent_space[200:299, 0], latent_space[200:299, 1], label = "2")
ax1.scatter(latent_space[300:399, 0], latent_space[300:399, 1], label = "3")
ax1.scatter(latent_space[400:499, 0], latent_space[400:499, 1], label = "4")
ax1.scatter(latent_space[500:599, 0], latent_space[500:599, 1], label = "5")
ax1.scatter(latent_space[600:699, 0], latent_space[600:699, 1], label = "6")
ax1.scatter(latent_space[700:799, 0], latent_space[700:799, 1], label = "7")
ax1.scatter(latent_space[800:899, 0], latent_space[800:899, 1], label = "8")
ax1.scatter(latent_space[900:999, 0], latent_space[900:999, 1], label = "9")
ax1.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
ax1.set_xlabel("Feature 1 \n\n  \
               Encoder will learn more complex relationship than PCA(Non-linearity).\n" +
               "But Sigmoid activation function restricts its ability to do so.\n" +
               "The data lies in a linear subspace as the graph above shows, \n" +
               "in this case, PCA might be more suitbale.")
ax1.set_ylabel("Feature 2")
ax1.set_title("<Bottleneck space in 2-D>")

# PCA in 2-D: Draw a 2-D scatter plot
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(projectedData_2D[0:99, 0], projectedData_2D[0:99, 1], label = "0")
ax2.scatter(projectedData_2D[100:199, 0], projectedData_2D[100:199, 1], label = "1")
ax2.scatter(projectedData_2D[200:299, 0], projectedData_2D[200:299, 1], label = "2")
ax2.scatter(projectedData_2D[300:399, 0], projectedData_2D[300:399, 1], label = "3")
ax2.scatter(projectedData_2D[400:499, 0], projectedData_2D[400:499, 1], label = "4")
ax2.scatter(projectedData_2D[500:599, 0], projectedData_2D[500:599, 1], label = "5")
ax2.scatter(projectedData_2D[600:699, 0], projectedData_2D[600:699, 1], label = "6")
ax2.scatter(projectedData_2D[700:799, 0], projectedData_2D[700:799, 1], label = "7")
ax2.scatter(projectedData_2D[800:899, 0], projectedData_2D[800:899, 1], label = "8")
ax2.scatter(projectedData_2D[900:999, 0], projectedData_2D[900:999, 1], label = "9")
ax2.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
ax2.set_xlabel("PC 1 \n\n" +
               "Even with Sigmoid function, PCA still projects data on linear subspaces(PC1 and PC2)\n" + 
               "and these linear subspaces are independent to each other.")
ax2.set_ylabel("PC 2")
ax2.set_title("<PCA in 2-D>")

plt.tight_layout() # minimize the overlap of subplots
print("A new window for two 2-D scatter plots has been created...")
plt.show()
##################################################################
""" 
# End of Question 1 - 2D plotting


# Question 1 - 3D plotting (Line 108 - 169)
# -> Remove lines 110 and 170 in <Q1.py>
# -> Uncomment lines 21, 55, and 56 in <simpleAutoencoder.py>
""" 
# Autoencoder in 3-D: Run the trained model with one hidden layer
print("Encoding from the input layer to the bottleneck layer(3D)...")
latent_space = model.get_bottleneck(torch.from_numpy(train_set).float())
latent_space = latent_space.detach().numpy()
print("The shape of latent space:(3D)", latent_space.shape) # should be (1000, 3)
print("The encoding has been finished...\n")

# PCA in 3-D:
print("Performing PCA on the original training dataset...")
mu = np.mean(train_set, axis=0)
train_set = train_set - mu
train_set = train_set.T
eVals, eVecs = np.linalg.eigh(np.matmul(train_set, train_set.T)) # Lambda(Eigenvalue), train_set(Eigenvector)
eVals_flip, eVecs_flip = np.flip(eVals), np.flip(eVecs, axis=1)
projectedData_3D = np.dot(train_set.T, eVecs_flip[:, 0:3])
print("The shape of projectedData is", projectedData_3D.shape)
print("End performing PCA...\n")

# Autoencoder in 3-D: Draw a 3-D scatter plot
fig = plt.figure("Q1.3D scatter plots")
fig.text(0.4, 0.95, "In 3-D plot, autoencoder with tanh function can represent data better than PCA.\n" +
        "since the dataset has non-linear relationships. \n" +
        "PCA only transforms data into 2 or 3 unique linear-relationships")
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(latent_space[0:99, 0], latent_space[0:99, 1], latent_space[0:99, 2], label = "0")
ax1.scatter(latent_space[100:199, 0], latent_space[100:199, 1], latent_space[100:199, 2], label = "1")
ax1.scatter(latent_space[200:299, 0], latent_space[200:299, 1], latent_space[200:299, 2], label = "2")
ax1.scatter(latent_space[300:399, 0], latent_space[300:399, 1], latent_space[300:399, 2], label = "3")
ax1.scatter(latent_space[400:499, 0], latent_space[400:499, 1], latent_space[400:499, 2], label = "4")
ax1.scatter(latent_space[500:599, 0], latent_space[500:599, 1], latent_space[500:599, 2], label = "5")
ax1.scatter(latent_space[600:699, 0], latent_space[600:699, 1], latent_space[600:699, 2], label = "6")
ax1.scatter(latent_space[700:799, 0], latent_space[700:799, 1], latent_space[700:799, 2], label = "7")
ax1.scatter(latent_space[800:899, 0], latent_space[800:899, 1], latent_space[800:899, 2], label = "8")
ax1.scatter(latent_space[900:999, 0], latent_space[900:999, 1], latent_space[900:999, 2], label = "9")
ax1.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.set_zlabel("Feature 3")
ax1.set_title("<Bottleneck space in 3-D>")

# PCA in 3-D: Draw a 3-D scatter plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(projectedData_3D[0:99, 0], projectedData_3D[0:99, 1], projectedData_3D[0:99, 2], label = "0")
ax2.scatter(projectedData_3D[100:199, 0], projectedData_3D[100:199, 1], projectedData_3D[100:199, 2], label = "1")
ax2.scatter(projectedData_3D[200:299, 0], projectedData_3D[200:299, 1], projectedData_3D[200:299, 2], label = "2")
ax2.scatter(projectedData_3D[300:399, 0], projectedData_3D[300:399, 1], projectedData_3D[300:399, 2], label = "3")
ax2.scatter(projectedData_3D[400:499, 0], projectedData_3D[400:499, 1], projectedData_3D[400:499, 2], label = "4")
ax2.scatter(projectedData_3D[500:599, 0], projectedData_3D[500:599, 1], projectedData_3D[500:599, 2], label = "5")
ax2.scatter(projectedData_3D[600:699, 0], projectedData_3D[600:699, 1], projectedData_3D[600:699, 2], label = "6")
ax2.scatter(projectedData_3D[700:799, 0], projectedData_3D[700:799, 1], projectedData_3D[700:799, 2], label = "7")
ax2.scatter(projectedData_3D[800:899, 0], projectedData_3D[800:899, 1], projectedData_3D[800:899, 2], label = "8")
ax2.scatter(projectedData_3D[900:999, 0], projectedData_3D[900:999, 1], projectedData_3D[900:999, 2], label = "9")
ax2.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.set_zlabel("PCA 3")
ax2.set_title("<PCA in 3-D>")
plt.show()
##################################################################
""" 
# End of Question1 - 3D plotting