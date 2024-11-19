import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from utilityDBN import standardizeData

# creating a neural network by sub-classing the nn.Module which is the base class for all ANN defined in PyTorch
class NeuralNet(nn.Module): 
	# use this method to initialize your neural network class
	def __init__(self, input_size, hidden_layer, num_classes, hiddenActivation=torch.relu): 
		super(NeuralNet, self).__init__()
		self.hidden = nn.ModuleList()
		self.hiddenActivation = hiddenActivation
		self.epochError= []
		self.trMu, self.trSd = [], []
		self.device = ''
		# Hidden layers
		if len(hidden_layer) == 1:
			self.hidden.append(nn.Linear(input_size, hidden_layer[0]))
		elif len(hidden_layer) > 1:
			for i in range(len(hidden_layer) - 1):
				if i == 0:
					self.hidden.append(nn.Linear(input_size, hidden_layer[i])) # nn.Linear is used to apply linear transformation on the data using weights and biases
					self.hidden.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
				else:
					self.hidden.append(nn.Linear(hidden_layer[i],hidden_layer[i+1]))
		# Output layer
		self.out = nn.Linear(hidden_layer[-1], num_classes)

	def forward(self, x): #forward pass
		# Feedforward
		for layer in self.hidden:
			x = self.hiddenActivation(layer(x)) # layer(x) is applying the linear transformation. After that activation is applied
		output = F.softmax(self.out(x), dim=1) # calculating softmax(class probability) on the output of last layer.
		return output
		
	def setHiddenWeight(self, W, b):
		for i in range(len(self.hidden)):
			self.hidden[i].bias.data=b[i].float()
			self.hidden[i].weight.data=W[i].float()

	def setOutputWeight(self, W, b):
		self.out.bias.data = b.float()
		self.out.weight.data = W.float()

	def standardizeData(data, mu=[], std=[]):
		# data: a (m * n) matrix where m is the # of observations and n is # of features
		# if any(mu) == None and any(std) == None:
		if not(len(mu) and len(std)):
			#pdb.set_trace()
			std = np.std(data,axis=0)
			mu = np.mean(data,axis=0)
			std[np.where(std==0)[0]] = 1.0 # This is for the constant features.
			standardizeData = (data - mu)/std
			return mu,std,standardizeData
		else:
			standardizeData = (data - mu)/std
			return standardizeData
			
	def unStandardizeData(data, mu, std):
		return std * data + mu

	def fit(self,trData,trLabels,standardizeFlag,batchSize,optimizationFunc='Adam',learningRate=0.001,numEpochs=100,cudaDeviceId=None,verbose=False):
		
		if standardizeFlag:
		#standardize data
			mu,sd,trData = standardizeData(trData)
			self.trMu = mu
			self.trSd = sd

		# prepare data for fine tuning
		torchData = Data.TensorDataset(torch.from_numpy(trData).float(),torch.from_numpy(trLabels.flatten().astype(int))) # convert numpy data into torch tensor
		dataLoader = Data.DataLoader(dataset=torchData,batch_size=batchSize,shuffle=True) # partition data into minibatches

		# Device configuration
		# self.device = torch.device('cuda:'+str(cudaDeviceId))
		self.device = torch.device('cpu')
		criterion = nn.CrossEntropyLoss() # set the loss function
		if optimizationFunc.upper()=='ADAM':
			optimizer = torch.optim.Adam(self.parameters(), lr=learningRate,amsgrad=True)
		elif optimizationFunc.upper()=='SGD':
			optimizer = torch.optim.SGD(self.parameters(), lr=learningRate,momentum=0.8)
		total_step = len(dataLoader)
		self.to(self.device) # get the model to cpu(gpu)
		for epoch in range(numEpochs):
			error=[]
			for i, (sample, labels) in enumerate(dataLoader):  
				# Move tensors to the configured device
				sample = sample.to(self.device) #get the data and labels to gpu/cuda
				labels = labels.to(self.device)
				#pdb.set_trace()

				# Forward pass
				outputs = self.forward(sample) #run forward pass
				loss = criterion(outputs, labels) #calculate cross-entropy loss
				error.append(loss.item()) #get the cross-entropy error in a variable

				# Backward and optimize
				optimizer.zero_grad() # zeros out any previously accumulated gradient
				loss.backward() # calculating the partial derivatives of the loss function with respect to model parameters
				optimizer.step() # updating the model parameters

				if (i+1) % 100 == 0:
					print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, numEpochs, i+1, total_step, loss.item()))
			self.epochError.append(np.mean(error))
			if verbose and ((epoch+1) % (numEpochs*0.1)) == 0:
				print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, numEpochs,self.epochError[-1]))

	def predict(self,x):
		# standarization has been applied on training data so apply on test data
		if len(self.trMu) != 0 and len(self.trSd) != 0:
			x = standardizeData(x,self.trMu, self.trSd)
		x = torch.from_numpy(x).float().to(self.device)

		with torch.no_grad():
			fOut = self.forward(x)
		fOut = fOut.to('cpu').numpy()
		predictedVal = np.max(fOut,axis=1)
		predictedLabels = (np.argmax(fOut,axis=1)).reshape(-1,1)
		return predictedVal,predictedLabels
