from __future__ import print_function, division
import numpy as np
import math

import matplotlib.pyplot as plt

from deep_memory import *

class MultilayerPerceptron():
    """Multilayer Perceptron Memory Kernel Extraction. A fully-connected neural network with one hidden layer.
    Unrolled to display the whole forward and backward pass.
    Parameters:
    -----------
    n_hidden: int:
        The number of processing nodes (neurons) in the hidden layer. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.01, dt = 1, optimizer = False, loss = 'mse', fe = None):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lr_cache = learning_rate
        #self.hidden_activation = Sigmoid()
        self.hidden_activation = MemInt()
        self.output_activation = Linear()
        self.fe = fe #no deterministic force in GLE
        if loss == 'mse':
            self.loss = SquareLoss()
        elif loss == 'mae':
            self.loss = Loss()
       
        self.stop = self.n_iterations
        self.optimizer = optimizer
        if self.optimizer == 'rprop' :
            self.rprop = Rprop(learning_rate = self.learning_rate, etapos = 1.3, etaneg = 0.6)
        if self.optimizer == 'adam' :
            self.adam = Adam(learning_rate = self.learning_rate)
        self.dt = dt
    def _initialize_weights(self, X, y):
        n_features = 1 #only one trajectory
        
        n_samples, n_features = X.shape
        _, n_outputs = y.shape
        
        # Hidden layer
        limit   = 1 / math.sqrt(n_features)

        self.W  = np.random.uniform(0, limit, (n_features, self.n_hidden))
        
        self.w0 = np.zeros((1, self.n_hidden))
        # Output layer
        limit = 1 
        self.V  = np.random.uniform(limit, limit, (self.n_hidden, n_outputs))
        #self.V = np.array([1])
        self.v0 = np.zeros((1, n_outputs))

    def fit(self, X, y, early_stop = 1e-06, accept_bias = False, check_loss = 5, initial = None):
        
        self._initialize_weights(X, y)
        if not initial is None:
            n_weights = self.n_hidden
            
            for i in range (n_weights):
                self.W[0][i] = (initial/np.exp(self.w0[0][i]))**(1/3)
                self.W[0][i] /= n_weights       
    
        old_grad =  0
        
        self.losses = []
        best_weights = self.W
        best_biases = self.w0
        
        if self.fe is None:
            self.fe = np.zeros(len(X))
         
        for i in range(self.n_iterations):

            # ..............
            #  Forward Pass
            # ..............

            # HIDDEN LAYER
            
            inp = np.array([np.arange(0,len(X)*self.dt, self.dt)]).T
            
   
            hidden_output = np.zeros((self.n_hidden,  len(inp)))
            
            for j in range(self.n_hidden):
                hidden_input = -1*(inp*self.W[0][j] + self.w0[0][j])
                hidden_output[j] = -1*self.hidden_activation(hidden_input,X, dt = self.dt) + self.fe
                
            hidden_output = hidden_output.T 
            
           
            # OUTPUT LAYER
            
           
            output_layer_input = hidden_output.dot(self.V) + self.v0
            
            
            y_pred = self.output_activation(output_layer_input)
            
            # ...............
            #  Backward Pass
            # ...............
            
            loss = np.mean(self.loss.loss(y,y_pred))
            print('loss in epoch ' + str(i+1) + ' : ' + str(loss))
            self.losses.append(loss)
            np.nan_to_num(self.losses, nan = np.inf)
            
            if i > check_loss:
                if self.losses[i] < np.min(self.losses[:i]):
                   
                    best_weights = self.W
                    best_biases = self.w0
            if loss < early_stop :
                print('Minimal loss reached! early stop of training!')
                self.stop = i
                break
                
            # OUTPUT LAYER
            # Grad. w.r.t input of output layer (have  no weights)
            grad_wrt_out_l_input = np.mean(self.loss.gradient(y, y_pred)) * self.output_activation.gradient(output_layer_input)
            
            #therefore grad_v is zero
            #grad_v = hidden_output.T.dot(grad_wrt_out_l_input)
            grad_v = 0
            grad_v0 = 0
            #grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)
        
            # HIDDEN LAYER
            # Grad. w.r.t input of hidden layer
            
            grad_hidden = np.zeros((self.n_hidden,  len(inp)))
            
            for j in range(self.n_hidden):
                hidden_input = -1*(inp*self.W[0][j] + self.w0[0][j])
                grad_hidden[j] = self.hidden_activation.gradient(hidden_input,X,self.dt)
            grad_wrt_hidden_l_input = -1*grad_wrt_out_l_input*grad_hidden.T
            grad_w = X.T.dot(grad_wrt_hidden_l_input)
            
            grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)
            
            # Update weights (by gradient descent)
            # Move against the gradient to minimize loss
            
            if i > check_loss:
                    
                loss_change = self.losses[i] - np.mean(self.losses[-check_loss:])
                 
                if (loss_change > 0):
                    
                    #self.learning_rate = self.lr_cache*1.2
                    self.W = best_weights
                    self.w0 = best_biases 
            if self.optimizer == False:
                self.W  -= self.learning_rate * grad_w
                
            else :
                    
                if self.optimizer == 'adam' :
                    self.W = self.adam.update(self.W, grad_w)
                    #self.V = self.rprop.update(self.V, grad_v)
                elif self.optimizer == 'rprop':

                    eta_new, self.W = self.rprop.update(self.W, grad_w,old_grad, self.learning_rate)
                    self.learning_rate =  eta_new
                    print('new learning rate = ' + str(self.learning_rate))
                    old_grad = grad_w

                else:
                    print('please use an optimizer (rprop or adam) or set optimizer to FALSE')

            
            if accept_bias:
                
                #self.V  -= self.learning_rate * grad_v
                self.V -= 0
                #self.v0 -= self.learning_rate * grad_v0
                self.v0 -= 0
                self.w0 -= self.learning_rate * grad_w0
            else:
                
                self.V -= 0
                self.v0 -= 0
                self.w0 -= 0
            

            #print(self.W, self.w0)
        return self.V, self.v0, self.W, self.w0
    # Use the trained model to predict labels of X
    def predict(self, X):
        inp = np.array([np.arange(0,len(X)*self.dt, self.dt)]).T
            
   
        hidden_output = np.zeros((self.n_hidden,  len(inp)))

        for j in range(self.n_hidden):
            hidden_input = -1*(inp*self.W[0][j] + self.w0[0][j])
            hidden_output[j] = -1*self.hidden_activation(hidden_input,X, dt = self.dt) + self.fe

        hidden_output = hidden_output.T

        output_layer_input = hidden_output.dot(self.V) + self.v0


        y_pred = self.output_activation(output_layer_input) 
            
        return y_pred
    
    def learning_curve(self):
        epochs = np.arange(1,self.stop+1)
        learning_curve = self.losses[:len(epochs)]
        return epochs, learning_curve
    
    
    def compute_kernel(self, time, verbose_plot = False, initial = None):
        weights = (np.absolute(self.W))
        biases = (np.absolute(self.w0))
        self.kernel = np.zeros(len(time))
        n_weights = self.n_hidden
        
        if initial == None:
            
            for i in range(n_weights):
           # k = (weights[0][i])**3 * np.exp(biases[0][i]**2)*np.exp(-weights[0][i]**2*time)
                k = (weights[0][i])**3 * np.exp(biases[0][i])*np.exp(-weights[0][i]*time)
                self.kernel += k
            
        else:
            k0 = np.zeros(weights.shape)
            for i in range (n_weights):
                k0[0][i] = ((initial/np.exp(biases[0][i]))/n_weights)**(1/3)
    
                k = (k0[0][i])**3 * np.exp(biases[0][i])*np.exp(-k0[0][i]*time)
                self.kernel += k
        
        
    
        if verbose_plot:
            plt.plot(t,self.kernel, label = 'extracted')
            plt.show
            
        return self.kernel
