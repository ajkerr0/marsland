"""


"""

import numpy as np
import matplotlib.pyplot as plt

class Evolution:
    """A genetic algorithm class, using binary strings as individuals.
    
    Keywords:
        isize (int): Size of the individual strings.
        psize (int): Size of the population.
        fitfunc (function): Fitness function; returns high values for high fitness.
        ne (int): Number of epochs."""
    
    def __init__(self, isize, psize, fitfunc, nepoch=50, nelite=3, tour=True):
        self.fitfunc = fitfunc
        self.nepoch = nepoch
        self.nelite = nelite
        self.tour = tour
        if psize%2 == 1:
            psize += 1
        self.pop = np.where(np.random.rand(psize,isize)<.5,0,1)
        self.mrate = 1/isize
        
    @staticmethod
    def select_parents(pop, fitness):
        """Return the parents of the next generation
        using fitness proportional selection."""
        
        #scale the fitnesses such that the highest value is 10
        #ignore individuals with new fitness < 1 as parents for new generation
        #add number of copies of the individuals based on their new fitness to be randomly selected
        
        fitness = 10*fitness/fitness.max()
        
        #build pool of fit parents
        #initialize pool with a dummy array
        newpop = np.zeros((1, pop.shape[1]))
        
        for i in range(pop.shape[0]):
            if np.round(fitness[i]) >= 1:
                newpop = np.concatenate((newpop, np.kron(np.ones((np.round(fitness[i]),1)), pop[i,:])), axis=0)
        
        newpop = np.delete(newpop, 0, 0)
                
        newpop = newpop.astype(int)
        
        indices = np.arange(newpop.shape[0])
        np.random.shuffle(indices)
        
        return newpop[indices[:pop.shape[0]]]
    
    @staticmethod
    def crossover(pop):
        """Return offspring of input population by
        performing single point crossover."""
        
        newpop = np.zeros(pop.shape, dtype=int)
        cross_point = np.random.randint(0, pop.shape[1], pop.shape[0])
        
        for i in range(0, pop.shape[0], 2):
            newpop[i  , :cross_point[i]] = pop[i  , :cross_point[i]]
            newpop[i  , cross_point[i]:] = pop[i+1, cross_point[i]:]
            newpop[i+1, :cross_point[i]] = pop[i+1, :cross_point[i]]
            newpop[i+1, cross_point[i]:] = pop[i  , cross_point[i]:]
        
        return newpop
        
    def mutate(self, pop):
        """Return mutated population"""
        
        whereMu = np.random.rand(pop.shape[0], pop.shape[1])
        pop[np.where(whereMu < self.mrate)] = 1 - pop[np.where(whereMu < self.mrate)]
        return pop
     
    def elitism(self, newpop, fitness):
        """Return the population with random individuals replaced by the nelite
        individuals from the previous generation."""
        
        best = np.argsort(fitness)
        best = self.pop[best[-self.nelite:]]
        indices = np.arange(newpop.shape[0])
        np.random.shuffle(indices)
        newpop = newpop[indices]
        newpop[:self.nelite] = best
        return newpop
        
    def tournament(self, newpop, fitness):
        """Return the results of a tournament between the new generation and
        its preceeding one."""
        return newpop
        
        
    def evolve(self):
        """Run the genetic algorithm"""
        
        bestfit = np.zeros(self.nepoch)
        
        for i in range(self.nepoch):
            
            #get the fitness of the current population
            fitness = self.fitfunc(self.pop)
            
            #select the parents of the next generation
            newpop = self.select_parents(self.pop, fitness)
            
            #perform crossover, mutation
            newpop = self.crossover(newpop)
            newpop = self.mutate(newpop)
            
            #apply elitism and host tournaments
            if self.nelite > 0:
                newpop = self.elitism(newpop, fitness)
            if self.tour:
                newpop = self.tournament(newpop, fitness)
                
            self.pop = newpop
            bestfit[i] = np.max(fitness)
            
        plt.plot(np.arange(1,self.nepoch+1), bestfit, '-kx')
    
def fourpeaks(pop, T=.20):
    
    T = np.ceil(pop.shape[0]*T)
    
    fitness = np.zeros(pop.shape[0])
    
    for i in range(pop.shape[0]):
        zeros = np.where(pop[i,:]==0)[0]
        ones =  np.where(pop[i,:]==1)[0]
        
        if ones.size > 0:
            consec0 = ones[0]
        else:
            consec0 = 0
            
        if zeros.size > 0:
            consec1 = pop.shape[1] - zeros[-1] - 1
        else:
            consec1 = 0
            
        if consec0 > T and consec1 > T:
            fitness[i] = np.maximum(consec0, consec1)+100
        else:
            fitness[i] = np.maximum(consec0, consec1)
            
    return fitness
    
a = Evolution(20,20,fourpeaks)

a.evolve()
    
    