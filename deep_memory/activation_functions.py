import numpy as np

# Collection of activation functions
# Reference: https://en.wikipedia.org/wiki/Activation_function


class Linear():
    def __call__(self,x):
        return x

    def gradient(self,x):
        return 1
    
class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))
    
class FDT():
    def __call__(self,x):
        return np.exp(-x)
    
    def gradient(self,x):
        return self.__call__(x)*-1
    
class MemInt():
    def __call__(self, inp,x, dt = 1):
        output = np.zeros(len(x))
        output[0] = 0
        
        for i in range(0,len(output)):
                #output[i] = 0.5*dt*inp[0]*x[i] + 0.5*dt*inp[i]*x[0] + dt*np.sum(inp[1:i+1]*x[:i][::-1])
                
                output[i] = 0.5*dt*np.exp(inp[0])*x[i] + 0.5*dt*np.exp(inp[i])*x[0] + dt*np.sum(np.exp(inp[1:i+1])*x[:i][::-1])
        return output
    
    def gradient(self, inp, x, dt =1, y0 = 0):
        output = np.zeros(len(x))
        output[0] = 0
        
        
        for i in range(0,len(output)):
                output[i] = 0.5*dt*x[i]*np.exp(inp[0])*inp[0] + 0.5*dt*x[0]*np.exp(inp[i])*inp[i] + dt*np.sum(np.exp(inp[1:i+1])*inp[1:i+1]*x[:i][::-1])
        return output
    
    
class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class TanH():
    def __call__(self, x):
        return 2 / (1 + np.exp(-2*x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)

class ReLU():
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)

class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)

class ELU():
    def __init__(self, alpha=0.1):
        self.alpha = alpha 

    def __call__(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return np.where(x >= 0.0, 1, self.__call__(x) + self.alpha)

class SELU():
    # Reference : https://arxiv.org/abs/1706.02515,
    # https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_MLP_MNIST.ipynb
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946 

    def __call__(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha*(np.exp(x)-1))

    def gradient(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))

class SoftPlus():
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1 / (1 + np.exp(-x))

