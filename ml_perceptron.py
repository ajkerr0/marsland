"""


@author: Alex Kerr
"""

import numpy as np
import matplotlib.pyplot as plt

class MLP:
    """Multi-layer Perceptron of one hidden layer.  Hidden layer sigmoid threshold function.
    
    Arguments:
        training (array-like): 2D array where rows correspond to the points.
        target (array-like): 2D array of the target values, indexed like training data.
        nh (int): Number of nodes in the hidden layer.
        
    Keywords:
        beta (float): Parameter for the sigmoid threshold functions.
        p (float): Momentum rate in training.
        tbias (int): Number of bias nodes of -1 in the input layer. Default is 1.
        hbias (int): Number of bias nodes of -1 in the hidden layer.  Default is 1.
        outtype (str): Threshold function of the output node(s).  Choices are 'linear',
            'logistic', or 'softmax'.  Default is 'linear'."""
    
    def __init__(self, training, target, nh, beta=1., p=.9, tbias=1, hbias=1, outtype="linear"):
        self.training = np.concatenate((training, -np.ones((len(training), tbias))), axis=1)
        self.target = np.array(target)
        self.beta = beta
        self.p = p
        self.tbias = tbias
        self.hbias = hbias
        self.w1 = np.random.randn(self.training.shape[1], nh)/10.
        self.w2 = np.random.randn(nh+self.hbias, self.target.shape[1])/10.
        if outtype == 'linear':
            def threshold(outputs):
                return outputs
            def deltao(outputs):
                return (outputs-self.target)/self.target.shape[0]
        elif outtype == 'logistic':
            def threshold(outputs):
                return 1./(1. + np.exp(-self.beta*outputs))
            def deltao(outputs):
                return self.beta*(outputs-self.target)*outputs*(1.-outputs)
        self.threshold = threshold
        self.deltao = deltao
            
    def forward(self, inputs):
        """Return the forward progress of the neural net."""
        
        #apply first set of weights
        hidden = np.dot(inputs, self.w1)
        #apply threshold function
        hidden = 1./(1. + np.exp(-self.beta*hidden))
        #prep the inputs to the output nodes
        self.hidden = np.concatenate((hidden, -np.ones((inputs.shape[0], self.hbias))),axis=1)
        
        return self.threshold(np.dot(self.hidden, self.w2))
        
            
    def train(self, lrate=.25, niter=100):
        """Train the neural net."""
        
        #starting momentum
        dw1 = np.zeros((self.w1.shape))
        dw2 = np.zeros((self.w2.shape))
        
        for n in range(niter):
            
            outputs = self.forward(self.training)
            
            error = .5*np.sum((outputs-self.target)**2)
            if n%100 == 0:
                print("Iteration: {0}".format(n))
                print("Error: {0}".format(error))
                
            deltao = self.deltao(outputs)
            deltah = self.hidden*self.beta*(1.-self.hidden)*(np.dot(deltao, np.transpose(self.w2)))
            
            dw1 = lrate*np.dot(np.transpose(self.training), deltah[:,:-1]) + self.p*dw1
            dw2 = lrate*np.dot(np.transpose(self.hidden), deltao) + self.p*dw2
            
            self.w1 += -dw1
            self.w2 += -dw2
            
            
#testing out       
if __name__ == "__main__":
    
    x = np.ones((1,40))*np.linspace(0,1,40)
    t = np.sin(2*np.pi*x) + np.cos(4*np.pi*x) + np.random.randn(40)*.2
    x = x.T
    t = t.T
    
    #plt.plot(x,t)
    
    train, traint = x[0::2,:], t[0::2,:]
    test, testt = x[1::4,:], t[1::4,:]
    valid, validt = x[3::4,:], t[3::4,:]
    
    net = MLP(train, traint, 3)
    net.train(niter=101)            
        
        