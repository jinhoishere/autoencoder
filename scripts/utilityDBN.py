import numpy as np
import random
import pandas
from operator import itemgetter
import itertools
import pdb

def stratifiedSampling(label,sampleSize):

	nSample = len(label)
	uniqueLabels = np.unique(label)
	classCardinality = {}
	for l in uniqueLabels:
		n = len(np.where(label==l)[0])
		p = round((100*n/nSample),5)
		classCardinality[int(l)] = round(p/100*sampleSize)
	return classCardinality

def stratifiedSampleSelection(labeledData,nSample):
	# labeledData: a [n x p] matrix where the last column is the label.
	#				n: no of samples, p-1: no of features
	# nSample: no of randomly selected samples fron labeledData.

	data = []
	label = []
	labelList = labeledData[:,-1]
	uniqueLabels = np.unique(labelList)
	nClass = len(uniqueLabels)
	noSamplePerClass = stratifiedSampling(labelList,nSample)
	for i in range(len(uniqueLabels)):
		indices = np.where(labeledData[:,-1]==uniqueLabels[i])[0]
		n = noSamplePerClass[uniqueLabels[i]]
		data.append(labeledData[indices[:n],:-1])
		label.append(labeledData[indices[:n],-1])
	return np.vstack((data)),np.hstack((label)).reshape(-1,1)

def splitDataByClassLabel(labeledData,noSamplePerClass=0):
	data = []
	label = []
	labelList = labeledData[:,-1]
	uniqueLabels = np.unique(labelList)
	for i in range(len(uniqueLabels)):
		indices = np.where(labeledData[:,-1]==uniqueLabels[i])[0]
		if noSamplePerClass==0:#take all the samples of this class
			data.append(labeledData[indices,:-1])
			label.append(labeledData[indices,-1])
		else:
			data.append(labeledData[indices[:noSamplePerClass],:-1])
			label.append(labeledData[indices[:noSamplePerClass],-1])
	return np.vstack((data)),np.hstack((label)).reshape(-1,1)

def createTrTstSet_n(data,labels,noSamplePerClass):
	# This method will split the dataset into training and test set.
	# The labeled_data will be shuffled.
	# For each class #noSamplePerClass samples will be put together in training set
	# and the rest will be put together in test set
	trData,trLabels = [],[]
	tstData,tstLabels = [],[]
	uniqueLabels = np.unique(labels)
	labeledData = np.hstack((data,labels.reshape(-1,1)))
	np.random.shuffle(labeledData)
	for i in range(len(uniqueLabels)):
		indices = np.where(labeledData[:,-1]==uniqueLabels[i])[0]
		#for training set		
		trData.append(labeledData[indices[:noSamplePerClass],:-1])
		trLabels.append(labeledData[indices[:noSamplePerClass],-1])
		#for test set
		tstData.append(labeledData[indices[noSamplePerClass:],:-1])
		tstLabels.append(labeledData[indices[noSamplePerClass:],-1])
	
	trData,trLabels = np.vstack((trData)),np.hstack((trLabels)).reshape(-1,1)
	tstData,tstLabels = np.vstack((tstData)),np.hstack((tstLabels)).reshape(-1,1)

	return trData,trLabels,tstData,tstLabels

def createTrTstSet_n(data,labels,noSamplePerClass):
	# This method will split the dataset into training and test set.
	# The labeled_data will be shuffled.
	# For each class #noSamplePerClass samples will be put together in training set
	# and the rest will be put together in test set
	trData,trLabels = [],[]
	tstData,tstLabels = [],[]
	uniqueLabels = np.unique(labels)
	labeledData = np.hstack((data,labels.reshape(-1,1)))
	np.random.shuffle(labeledData)
	for i in range(len(uniqueLabels)):
		indices = np.where(labeledData[:,-1]==uniqueLabels[i])[0]
		#for training set		
		trData.append(labeledData[indices[:noSamplePerClass],:-1])
		trLabels.append(labeledData[indices[:noSamplePerClass],-1])
		#for test set
		tstData.append(labeledData[indices[noSamplePerClass:],:-1])
		tstLabels.append(labeledData[indices[noSamplePerClass:],-1])
	
	trData,trLabels = np.vstack((trData)),np.hstack((trLabels)).reshape(-1,1)
	tstData,tstLabels = np.vstack((tstData)),np.hstack((tstLabels)).reshape(-1,1)

	return trData,trLabels,tstData,tstLabels
	
def splitData_n(data,labels,nTrData):
	# This method will split the dataset into training and test set.
	# The labeled_data will be shuffled and then first nTrData will be put in training set
	# and the rest will be put together in test set
	train_set =[]
	test_set =[]
	no_data = len(data)
	dataDim = np.shape(data)[1]
	indices = np.arange(no_data)
	np.random.shuffle(indices)
	trData,trLabels = data[indices[:nTrData],:],labels[indices[:nTrData]].reshape(-1,1)
	lTrData = np.hstack((trData,trLabels))
	sortedTrData = np.array(sorted(lTrData, key=itemgetter(dataDim)))
	trData,trLabels = sortedTrData[:,:-1],sortedTrData[:,-1]
	tstData,tstLabels = data[indices[nTrData:],:],labels[indices[nTrData:]].reshape(-1,1)
	lTstData = np.hstack((tstData,tstLabels))
	sortedTstData = np.array(sorted(lTstData, key=itemgetter(dataDim)))
	tstData,tstLabels = sortedTstData[:,:-1],sortedTstData[:,-1]
	return trData,trLabels.reshape(-1,1),tstData,tstLabels.reshape(-1,1)

def splitDataByGroupId(data,label,groupList,splitRatio=0.8):
	#This method will split the data based on a group. I'm writing this to split data based on subject ids for biological dataset.
	#Multiple sample can be associated to the same group.
	#data: A [m x n] array where m is the no. of samples and n is the no. of features.
	#label: A list with m elements.
	#groupList: A list with m elements. This will contain the corresponding group id for each sample.
	#splitRatio: Ratio to partition the data into training and validation/test set. Default value is 0.8.

	uniqueGroups = np.unique(groupList)
	nTrGrps = int(np.ceil(splitRatio*len(uniqueGroups)))
	nValGrps = len(uniqueGroups) - nTrGrps
	
	#randomly partition uniqueGroups into two.
	trGrps = np.random.choice(uniqueGroups, nTrGrps, replace=False)
	valGrps = np.array(list(set(uniqueGroups) - set(trGrps)))
	
	#capture the corresponding sample indices of trGrps and valGrps
	trIndx = [np.where(groupList==trGrps[i])[0] for i in range(len(trGrps))]
	trIndx = np.hstack((trIndx))
	valIndx = [np.where(groupList==valGrps[i])[0] for i in range(len(valGrps))]
	valIndx = np.hstack((valIndx))

	#using the sample indices partition the original data into two parts along with corresponding labels
	part1Data = data[trIndx,:]
	part1Label = label[trIndx,:]
	part2Data = data[valIndx,:]
	part2Label = label[valIndx,:]
	
	return part1Data,part1Label,part2Data,part2Label

def selectDataByClassLabels(data,label,classLabels):
	#this function will select a subset based on given class label
	subData = []
	subLabels = []
	for c in classLabels:
		idx = np.where(label==c)[0]
		subData.append(data[idx,:])
		subLabels.append(label[idx])
	subData = np.vstack((subData))
	#pdb.set_trace()
	subLabels = np.vstack((subLabels))
	return subData,subLabels

def makeStratifiedSubset(labeledData,noSamplePerClass=0):
	#This function will randomly pick 'noSamplePerClass' samples from each class to create a subset.
	#If noSamplePerClass==0, then all the data will be returned.
	data = []
	label = []
	classList = np.unique(labeledData[:,-1])
	for c in classList:
		indices = np.where(labeledData[:,-1]==c)[0]
		if noSamplePerClass==0:#take all the samples of this class
			data.append(labeledData[indices,:-1])
			label.append(labeledData[indices,-1])
		else:
			data.append(labeledData[indices[:noSamplePerClass],:-1])
			label.append(labeledData[indices[:noSamplePerClass],-1])
	return np.vstack((data)),np.hstack((label)).reshape(-1,1)


def standardizeData(data,mu=[],std=[]):
	#data: a m x n matrix where m is the no of observations and n is no of features
	#if any(mu) == None and any(std) == None:
	if not(len(mu) and len(std)):
		#pdb.set_trace()
		std = np.std(data,axis=0)
		mu = np.mean(data,axis=0)
		std[np.where(std==0)[0]] = 1.0 #This is for the constant features.
		standardizeData = (data - mu)/std
		return mu,std,standardizeData
	else:
		standardizeData = (data - mu)/std
		return standardizeData
		
def unStandardizeData(data,mu,std):
	return std * data + mu

def KMean(data,nCenter,nItr):
	from sklearn.cluster import KMeans
	print('Running kMeans clustering with no of centers:',nCenter)
	kmeans = KMeans(n_clusters=nCenter, n_init=50, max_iter=nItr, random_state=0).fit(data)
	centers = kmeans.cluster_centers_
	vRegion = {}
	for c in range(nCenter):
		key = 'Region'+str(c+1)
		vRegion[key] = {}
		vRegion[key]['Center'] = kmeans.cluster_centers_[c]
		vRegion[key]['PatternIndex'] = np.where(c == kmeans.labels_)[0]
	vRegion['Inertia']=np.sqrt(kmeans.inertia_)/nCenter #Inertia=Sum of the squared distances of samples from their nearest cluster center.		
	print('Inertia per cluster:',vRegion['Inertia'])		
	return vRegion

def Agglomerative(data,nCenter,nItr,algo):
	from sklearn.cluster import AgglomerativeClustering
	clustering = AgglomerativeClustering(linkage=algo, n_clusters=nCenter).fit(data)
	vRegion = {}		
	for c in range(nCenter):
		key = 'Region'+str(c+1)
		vRegion[key] = {}
		vRegion[key]['Center'] = np.mean(data[np.where(clustering.labels_ == c)[0],:],axis=0)
		vRegion[key]['PatternIndex'] = np.where(clustering.labels_ == c)[0]
	return vRegion	

def createOutputByClustering(data,nCenter,nItr,algo='kMean'):
	if algo.upper()=='KMEAN':
		vRegion=KMean(data,nCenter,nItr)
	else:
		vRegion=Agglomerative(data,nCenter,nItr,algo)
	trInput=[]
	trOutput=[]
	for k in vRegion.keys():
		if 'Region' in k:
			d=data[vRegion[k]['PatternIndex'],:]
			trInput.append(d)
			trOutput.append(np.tile(vRegion[k]['Center'],(len(d),1)))
	return np.vstack((trInput)),np.vstack((trOutput))
			
def PCA(data,p):
	#data = [m x n] array where m: no of samples and n: no of features.
	#p: no. of principle components for dim. reduction
	#pdb.set_trace()
	data = data.astype('float64')
	data = data - np.mean(data,axis=0) # substruct the mean.
	cov = np.dot(data.T,data)
	#pdb.set_trace()
	#eVals,eVecs = np.linalg.eig(cov)
	#eVals = np.real(eVals)
	#eVecs = np.real(eVecs)

	eVals,eVecs = np.linalg.eigh(cov)

	#the eigen values are returned in ascending order. Need to flip the eigen values and eigrn vectors
	eVecs_flip = np.flip(eVecs,axis=1)
	eVals_flip = np.flip(eVals)

	#Now project the data on the first m eigen vector
	projectedData = np.dot(data,eVecs_flip[:,0:p])
	return projectedData,eVals_flip,eVecs_flip
	
	
def splitData(labeled_data,split_ratio):
	# This method will split the dataset into training and test set based on the split ratio.
	# Training and test set will have data from each class according to the split ratio.
	# First hold the data of different classes in different variables.
	train_set =[]
	test_set =[]
	no_data = len(labeled_data)	
	sorted_data = labeled_data[np.argsort(labeled_data[:,-1])]#sorting based on the numeric label.
	first_time = 'Y'
	for classes in np.unique(sorted_data[:,-1]):
		temp_class = np.array([sorted_data[i] for i in range(no_data) if sorted_data[i,-1] == classes])
		np.random.shuffle(temp_class)#Shuffle the data so that we'll get variation in each run
		tr_samples = np.floor(len(temp_class)*split_ratio)
		tst_samples = len(temp_class) - tr_samples
		if(first_time == 'Y'):
			train_set = temp_class[:int(tr_samples),]
			test_set = temp_class[-int(tst_samples):,]
			first_time = 'N'
		else:
			train_set = np.vstack((train_set,temp_class[:int(tr_samples),]))
			test_set = np.vstack((test_set,temp_class[-int(tst_samples):,]))
	
	#no_of_trn_samples = int(np.ceil(split_ratio*len(labeled_data)))
	#no_of_tst_samples = int(len(labeled_data) - no_of_trn_samples)
	#return labeled_data[:no_of_trn_samples,:],labeled_data[-no_of_tst_samples:,:]
	return train_set,test_set
