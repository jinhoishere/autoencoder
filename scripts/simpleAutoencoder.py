import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # An encoder with ReLU activation function
        # 784 ==> 2
        self.encoder = torch.nn.Sequential(
            # Question 1 - 2D
            # torch.nn.Linear(28 * 28, 2)

            # Question 1 - 3D
            # torch.nn.Linear(28 * 28, 3)
            
            # Question 2 - model 1
            # torch.nn.Linear(28 * 28, 50),
            # torch.nn.Tanh(),
            # torch.nn.Linear(50, 2)

            # Question 2 - model 2
            # torch.nn.Linear(28 * 28, 100),
            # torch.nn.Tanh(),
            # torch.nn.Linear(100, 50),
            # torch.nn.Tanh(),
            # torch.nn.Linear(50, 2)

            # Question 2 - model 3
            # torch.nn.Linear(28 * 28, 200),
            # torch.nn.Tanh(),
            # torch.nn.Linear(200, 100),
            # torch.nn.Tanh(),
            # torch.nn.Linear(100, 50),
            # torch.nn.Tanh(),
            # torch.nn.Linear(50, 2)
        )


        # A decoder with ReLU to reconstruct the data
        # outputs the value between 0 and 1
        # 2 ==> 784
        self.decoder = torch.nn.Sequential(
            # Question 1 - 2D
            # torch.nn.Linear(2, 28 * 28), 
            # torch.nn.Sigmoid()

            # Question 1 - 3D
            # torch.nn.Linear(3, 28 * 28), 
            # torch.nn.Tanh()

            # Question 2 - model 1
            # torch.nn.Linear(2, 50),
            # torch.nn.Tanh(),
            # torch.nn.Linear(50, 28 * 28),
            # torch.nn.Sigmoid()

            # Question 2 - model 2
            # torch.nn.Linear(2, 50),
            # torch.nn.Tanh(),
            # torch.nn.Linear(50, 100),
            # torch.nn.Tanh(),
            # torch.nn.Linear(100, 28 * 28),
            # torch.nn.Sigmoid()

            # Question 2 - model 3
            # torch.nn.Linear(2, 50),
            # torch.nn.Tanh(),
            # torch.nn.Linear(50, 100),
            # torch.nn.Tanh(),
            # torch.nn.Linear(100, 200),
            # torch.nn.Tanh(),
            # torch.nn.Linear(200, 28 * 28),
            # torch.nn.Sigmoid()
        )
        
        #store loss over epoch
        self.losses=[]
    
    def forward(self, x):
        # return self.decoder(self.encoder(x))
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, trData, lrnRate, nEpochs, miniBatchSize, cudaDeviceId):
    
        # set device
        # self.device = torch.device('cuda:'+str(cudaDeviceId))
        
        # Load the model to device
        self.device = torch.device('cpu')
        self.to(self.device)
        
        #Prepare data for torch
        trDataTorch = Data.TensorDataset(torch.from_numpy(trData).float())
        dataLoader = Data.DataLoader(dataset=trDataTorch,batch_size=miniBatchSize,shuffle=True)
    
        # Using MSE Loss function
        loss_function = torch.nn.MSELoss()
 
        # Using an Adam Optimizer with lr = 0.1
        optimizer = torch.optim.Adam(self.parameters(),lr = lrnRate)
        
        for epoch in range(nEpochs):
            error=[]
            for i, Input in enumerate(dataLoader):
                #pdb.set_trace()
                Input = Input[0].to(self.device)
                # Forward pass
                recons = self.forward(Input)
                loss = loss_function(recons, Input)
                
                error.append(loss.item())
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
            # Storing the losses in a list for plotting
            self.losses.append(np.mean(error))
            #outputs.append((epochs, image, reconstructed))
            if ((epoch+1) % (nEpochs*0.1)) == 0:
                print ('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, nEpochs, self.losses[-1]))

    
    # A method to get the bottleneck space
    def get_bottleneck(self, x):
        with torch.no_grad():
            bottleneck = self.encoder(x)
        return bottleneck
