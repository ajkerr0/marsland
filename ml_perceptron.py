"""

Multi-layer Perceptron code ala Chapter 4 of Marsland's book
Borrowed from Marsland's code

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
        self.outtype = outtype  #for train_seq
            
    def forward(self, inputs):
        """Return the forward progress of the neural net."""
        
        #apply first set of weights
        hidden = np.dot(inputs, self.w1)
        #apply threshold function
        hidden = 1./(1. + np.exp(-self.beta*hidden))
        #prep the inputs to the output nodes
        self.hidden = np.concatenate((hidden, -np.ones((inputs.shape[0], self.hbias))),axis=1)
        
        return self.threshold(np.dot(self.hidden, self.w2))
        
            
    def train(self, lrate=.25, niter=100, print_err=True):
        """Train the neural net."""
        
        #starting momentum
        dw1 = np.zeros((self.w1.shape))
        dw2 = np.zeros((self.w2.shape))
        
        for n in range(niter):
            
            outputs = self.forward(self.training)
            
            error = .5*np.sum((outputs-self.target)**2)
            if n%100 == 0 and print_err:
                print("Iteration: {0}".format(n))
                print("Error: {0}".format(error))
                
            deltao = self.deltao(outputs)
            deltah = self.hidden*self.beta*(1.-self.hidden)*(np.dot(deltao, np.transpose(self.w2)))
            
            dw1 = lrate*np.dot(np.transpose(self.training), deltah[:,:-1]) + self.p*dw1
            dw2 = lrate*np.dot(np.transpose(self.hidden), deltao) + self.p*dw2
            
            self.w1 += -dw1
            self.w2 += -dw2
            
    def train_seq(self, lrate=.25, niter=100):
        """Train the neural net sequential (vs batch).
        Doesn't fit rest of class, for Problem 4.5."""
        
        inputs = self.training
        targets = self.target
        counts = np.arange(self.training.shape[0])
        change = np.copy(counts)
        
        for n in range(niter):
            
            
            if self.outtype == 'linear':
                def deltao_seq(output, t):
                    return output-t
            elif self.outtype == 'logistic':
                def deltao_seq(output, t):
                    return self.beta(output-t)*output*(1.-output) 
            
            for count, point, target in zip(counts, inputs, targets):
                
                hidden = np.dot(point, self.w1)
                hidden = 1./(1. + np.exp(-self.beta*hidden))
                hidden = np.append(hidden, [-1]*self.hbias)
                output = self.threshold(np.dot(hidden, self.w2))
                deltao = deltao_seq(output, target)
                deltah = hidden*self.beta*(1. - hidden)*(np.dot(deltao,np.transpose(self.w2)))
                
                self.w1 += -lrate*np.dot(np.transpose(point).reshape((len(np.transpose(point)), 1)), 
                                         deltah[:-1].reshape((1,len(deltah)-1)))
                self.w2 += -lrate*np.dot(np.transpose(hidden).reshape((len(np.transpose(hidden)),1)), 
                                         deltao.reshape((1,len(deltao))))
                
            np.random.shuffle(change)
            inputs = inputs[change,:]
            targets = targets[change,:]
                
            
    def estop(self, v, vtarget, lrate=.25, niter=100):
        """Train the neural net but stop when the validation set's error stops to decrease."""
        
        v = np.concatenate((v, -np.ones((len(v), self.tbias))), axis=1)
        vtarget = np.array(vtarget)
        
        #start with dummy errors
        old_verr1 = 1e5 + 2.
        old_verr2 = 1e5 + 1.
        new_verr = 1e5
        
        count = 0
        while ((old_verr1 - new_verr) > 1e-3 or (old_verr2 - old_verr1) > 1e-3):
            self.train(lrate, niter, print_err=False)
            old_verr2, old_verr1 = old_verr1, new_verr
            vout = self.forward(v)
            new_verr = .5*np.sum((vtarget-vout)**2)
            count += 1
            
        print("Stopped")
        print(new_verr)
        print(old_verr1)
        print(old_verr2)
            
#testing out       
if __name__ == "__main__":
    
    x = np.ones((1,40))*np.linspace(0,1,40)
    t = np.sin(2*np.pi*x) + np.cos(4*np.pi*x) + np.random.randn(40)*.2
    x = x.T
    t = t.T
    
    x = (x - x.mean(axis=0))/x.var(axis=0)
    t = (t - t.mean(axis=0))/t.var(axis=0)
    
#    plt.plot(x,t)
#    plt.show()
    
    train, traint = x[0::2,:], t[0::2,:]
    test, testt = x[1::4,:], t[1::4,:]
    valid, validt = x[3::4,:], t[3::4,:]
    
    print("Naive training")
    net = MLP(train, traint, 3)
#    net.train()
    net.train_seq()

    print("Early stopping")
    net = MLP(train, traint, 3)
    net.estop(valid, validt)       
        
        