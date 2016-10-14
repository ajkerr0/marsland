"""


@author: Alex Kerr
"""

import numpy as np

class Perceptron:
    """Babby's first neural network
    
    Arguments:
        training (array-like): 2D array where rows correspond to the points.
        target (array-like): 2D array of the target values, indexed like training data.
        
    Keywords:
        lrate (float): Learning rate of the network, ideally 0.1 <= lrate <= 0.4.
            Default is 0.25
        bias (int): Number of bias inputs of -1.  Default is 1.
        
    Methods:
        train: Train the neural net, as in find weights that generate a correct output
            from the training data.
        output: Return the final output of the network given an input."""
    
    def __init__(self, training, target, lrate=.25, bias=1):
        self.training = np.concatenate((training, -np.ones((len(training), bias))), axis=1)
        self.target = np.array(target)
        self.w = np.random.randn(self.training.shape[1], self.target.shape[1])/10.
        self.lrate = lrate
        self.bias = bias
        
    def recall(self, input_):
        return np.where(np.dot(input_, self.w)>0.,1,0)
        
    def train(self, n=10):
        """Train the neural net for 'n' iterations or until the output is perfect.
        
        Keywords:
            n (int): Iteration counter.  Default is 10."""
        
        for count in range(n):
            for point, t in zip(self.training, self.target):
                act = self.recall(point)
                self.w += -self.lrate*np.dot(np.transpose(point)[:,None], (act-t)[:,None])
            output = self.recall(self.training)
            print('Iteration: {0}'.format(count))
            print(self.w)
            print(output)
            
            if np.array_equal(output, self.target):
                print("########")
                print("Success!")
                print("########")
                return self.w, output
                
        print("####################")
        print("Iterations Exhausted")
        print("####################")
        return self.w, output
        
    def output(self, input_):
        return self.recall(np.concatenate(input_,-np.ones((len(input_), self.bias))))

#training logic gates as an example
#of course it doesn't make sense
inputs = [[0,0], [1,0], [0,1], [1,1]]
ORtargets = [[0],[1],[1],[1]]
ANDtargets = [[0],[0],[0],[1]]
XORtargets = [[0],[1],[1],[0]]

lor = Perceptron(inputs, ORtargets)
lor.train()

land = Perceptron(inputs, ANDtargets)
land.train()

#has difficulty with XOR logic...now we know why
lxor = Perceptron(inputs, XORtargets)
lxor.train()
