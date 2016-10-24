"""
Linear discriminant analysis

"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def lda(data, labels, redDim):
    
    data = np.array(data)
    
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

#plot vanilla iris data
iris = np.loadtxt("./iris.data", delimiter=',')
data, labels = iris[:,:-1], iris[:,-1].astype(int)

ind1, ind2, ind3 = np.where(labels==0)[0], np.where(labels==1)[0], np.where(labels==2)[0]

plt.figure()
plt.plot(data[ind1,0], data[ind1,1], 'bo',
         data[ind2,0], data[ind2,1], 'r^',
         data[ind3,0], data[ind3,1], 'gv')
         
plt.show()
         
#perform lda on iris data
newData,w = lda(data, labels,2)

plt.figure()
plt.plot(newData[ind1,0], newData[ind1,1], 'bo',
         newData[ind2,0], newData[ind2,1], 'r^',
         newData[ind3,0], newData[ind3,1], 'gv')

plt.show()

        
    
        