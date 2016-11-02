"""


"""

import numpy as np

class Evolution:
    """A genetic algorithm class, using binary strings as individuals.
    
    Keywords:
        isize (int): Size of the individual strings.
        psize (int): Size of the population.
        fitfunc (function): Fitness function; returns high values for high fitness.
        ne (int): Number of epochs."""
    
    def __init__(self, isize, psize, fitfunc, ne):
        self.pop = np.where(np.random.rand(psize,isize),0,1)
        self.fitfunc = fitfunc
        self.ne = ne
        
    def select_parents(self):
        pass
    
def fourpeaks(pop, T=.10):
    
    T = int(pop.shape[0]*T)
    
    fitness = np.zeros(pop.shape[0])
    
    for i in range(pop.shape[0]):
        zeros = np.where(pop[i,:]==0)
        ones =  np.where(pop[i,:]==1)
        
        if ones.size > 0:
            consec0 = ones[0][0]
        else:
            consec0 = 0
            
        if zeros.size > 0:
            consec1 = pop.shape[1] - zeros[-1][-1] - 1
        else:
            consec1 = 0
            
        if consec0 > T and consec1 > T:
            fitness[i] = np.maximum(consec0, consec1)+100
        else:
            fitness[i] = np.maximum(consec0, consec1)
            
    return fitness
    
Evolution(3,3,fourpeaks,3)
    
    