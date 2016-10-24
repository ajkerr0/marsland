"""
Linear discriminant analysis

"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def lda(data, labels, redDim):
    
    data = np.array(data)
#    data -= data.mean(axis=0)
    
    #covariance matrix
    #rows corresponds to variable, columns to data points for numpy function
    cov = np.cov(np.transpose(data))
    
    #initialize within-class scatter
    sw = np.zeros((data.shape[1], data.shape[1]))
    #get the unique labels, which are the classes
    classes = np.unique(labels)
    for class_ in classes:
        #first pick out the data points
        indices = np.where(labels==class_)[0]
        d = data[indices,:]
        #contribute to within-class scatter
        #probability of being in that class times class covariance
        sw += (d.shape[0]/data.shape[0])*np.cov(np.transpose(d))
        
    sb = cov - sw
    
    val, vec = scipy.linalg.eig(sw, sb)
    indices = np.argsort(val)
    indices = indices[::-1]
    vec = vec[:,indices]
    val = val[indices]
    w = vec[:,:redDim]
    newData = np.dot(data,w)
    return newData,w
    
iris = np.loadtxt("./iris.data", delimiter=',')
data, labels = iris[:,:-1], iris[:,-1].astype(int)

#print(data, labels)
    
#data = np.array([[0.1,0.1],[0.2,0.2],[0.3,0.3],[0.35,0.3],[0.4,0.4],[0.6,0.4],[0.7,0.45],[0.75,0.4],[0.8,0.35]])
#labels = np.array([0,0,0,0,0,1,1,1,1])
newData,w = lda(data,labels,3)
print(w)
plt.plot(data[:,0],data[:,1],'o',newData[:,0],newData[:,0],'.')
plt.show()

        
    
        