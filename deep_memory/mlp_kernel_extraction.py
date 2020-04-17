import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import torch #install pytorch first!! pip3 install torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict



class MultilayerPerceptron():
    """Multilayer Perceptron Memory Kernel Extraction. A fully-connected neural network with one hidden layer. The whole backpropagation step is provided by modules from the library pytorch
    
    Parameters:
    -----------
    n_hidden: int
        The number of processing nodes (neurons) in the hidden layer. 
    n_epochs: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    fe: array
        array of v-U-correlation function
    dt: float
        time-step of input array 
    optimizer: string
        name of optimizer that will be used for backpropagation step (Adam, SGD or Rprop)

    """
    def __init__(self, n_hidden, n_epochs=3000, learning_rate=0.01, dt = 1, optimizer = 'Rprop', fe = None):
        self.n_hidden = n_hidden #number neuron in 
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.fe = fe 
        self.dt = dt 
        self.optimizer = optimizer
    
    # takes in the network module and applies the specified weight initialization
    def weights_init_uniform_rule(self,m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0/np.sqrt(n)
            #m.weight.data.uniform_(-y, y)
            m.weight.data.uniform_(0, y)
            m.bias.data.fill_(0)


    def RMSELoss(self,yhat,y):
        return torch.sqrt(torch.mean((yhat-y)**2))
    
    #creating and fitting the neural network
    def fit(self, X, y,early_stop = 4e-06, hybrid =None, accept_bias = True, gpu = False, initial = None):        
        t = np.array([np.arange(0, len(X)*self.dt, self.dt)]).T
        
        feature, label = Variable(torch.FloatTensor([t]), requires_grad=True), Variable(torch.FloatTensor([y]), requires_grad=False) #input (time array) and desired output
        if self.fe is None:
            self.fe = np.array(np.zeros(len(feature[0]))).T
            self.fe = torch.FloatTensor([self.fe])
        else:
            self.fe = torch.FloatTensor([self.fe])
            
        #create Model
        activation_function = MemInt(dt = self.dt, fe = self.fe, h = torch.FloatTensor([X]))
        self.model = nn.Sequential(OrderedDict([('fc1', nn.Linear(1, self.n_hidden)),
                                          ('GLE', activation_function),
                                          ('fc2', nn.Linear(self.n_hidden, 1))]))  

        #choose Loss function
        #criterion = RMSELoss
        criterion = nn.MSELoss(reduction = 'mean')


        #choose optimizer
        if self.optimizer == 'Rprop':
            
            optimizer = optim.Rprop(self.model.parameters(), lr=self.learning_rate)
        
        elif self.optimizer == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        
        elif self.optimizer == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate)
            
        else:
            print('Choose an optimzer (Adam or Rprop!)')
            
            
        
        #initialize weights
        self.model.apply(self.weights_init_uniform_rule)
        self.model.fc2.weight.data.uniform_(1, 1)
        
        if not initial is None:
            n_weights = self.n_hidden
            
            for i in range (n_weights):
                #self.model.fc1.weight.data[i] = (initial)**(1/3)
                
                self.model.fc1.weight.data[i] = (initial[i][0])
                self.model.fc1.weight.data[i] /= n_weights
                
                self.model.fc1.bias.data[i] = np.log(initial[i][1])
                
                self.model.fc2.weight.data[i] = (initial[i][0])
                self.model.fc2.weight.data[i] /= n_weights    
                
                
        #create list of losses for every epoch     
        losses = []  
        
        #start training
        for e in range(0,self.n_epochs):

                # Training pass
            optimizer.zero_grad()
            with torch.no_grad():
                if accept_bias == False:
                    self.model.fc1.bias.zero_()
                    self.model.fc2.bias.zero_()
                    self.model.fc2.weight.data.uniform_(1, 1)
            #print(self.model.fc1.weight.data)
            #print(self.model.fc1.bias.data)
            #model.fc1.weight = torch.nn.Parameter

           
            output = self.model(feature)
            
            loss = criterion(output, label)
            losses.append(loss.detach().numpy())
            print('loss in epoch ' + str(e+1) + ' : ' + str(loss.detach().numpy()))
            
            if loss.detach().numpy() < early_stop :
                print('Minimal loss reached! early stop of training!')
                break
                
            if not hybrid is None: #Change to SGD if training is trapped in local minimum
                if loss.detach().numpy() < hybrid: 
                    optimizer  = optim.SGD(self.model.parameters(), lr = self.learning_rate)
                  
            loss.backward(retain_graph=True)
            optimizer.step()
            
        return losses, self.model
                      
                      
    def compute_kernel(self,time, verbose_plot = False):


        weights = (np.absolute(self.model.fc1.weight.data.numpy()))
        biases = self.model.fc1.bias.data.numpy()
        amplitudes = (np.absolute(self.model.fc2.weight.data.numpy())).T

        #biases = np.zeros(len(biases))
        self.kernel = np.zeros(len(time))
        n_weights = self.n_hidden


        for i in range(n_weights):
            #k = (amplitudes[i])**3 * np.exp(biases[i]**2)*np.exp(-weights[i]*time)
            k = (amplitudes[i]) * np.exp(biases[i])*np.exp(-weights[i]*time)
            #k = weights[i] * np.exp(biases[i])*np.exp(-weights[i]*time)
            self.kernel += k

        if verbose_plot:
                plt.plot(time,self.kernel, label = 'extracted')
                plt.show

        return self.kernel
    
    def predict(self, X):
        t = np.array([np.arange(0, len(X)*self.dt, self.dt)]).T
        
        feature = Variable(torch.FloatTensor([t]), requires_grad=True)
        output = self.model(feature)
        output = output[0].detach().numpy()
        
        return output
    
class MemInt(nn.Module): #class-activation for nn.Module, dt has to be modified before fitting!!
    def __init__(self, fe, h, dt = 0.01):
        super().__init__()
        self.dt = dt
        self.fe = fe
        self.h = h #input of the NN
    
    def memint(self,x): #GLE activation function in hidden layer
    
        output = torch.zeros(x.shape)

        for i in range(x.shape[2]): #calculating output for every node in hidden layer
            output[0][0][i] = 0

            dt = self.dt
            
            t = torch.arange(0,len(output[0])*dt,dt)

            #h = torch.exp(-t)
            kernel = torch.zeros(len(t))
            #print(kernel.shape,self.h[0].view(len(kernel)).shape)
            h = self.h[0].view(len(kernel))
            for j in range(0,len(output[0])): 
                inv_idx = torch.arange(h[:j].size(0)-1, -1, -1).long() #instead of h[:i][::-1], not supported in pytorch
                inv_tensor = h[:j][inv_idx]

                kernel[j] = torch.exp(-x[0][j][i])
                
                output[0][j][i] = - 0.5*dt*kernel[0]*h[j] - 0.5*dt*kernel[j]*h[0] - dt*torch.sum(kernel[1:j+1]*inv_tensor) + self.fe[0][j]
                
        #output = output + self.fe
        #plt.plot(output[0].detach().numpy())
        #plt.show()
        return output #every output is summarized to one array at output layer

        

    def forward(self, x):
        return self.memint(x)