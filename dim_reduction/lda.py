"""
Linear discriminant analysis

"""

import numpy as np

def lda(data, labels):
    
    data = np.array(data)
    
    #covariance matrix
    #rows corresponds to variable, columns to data points for numpy function
    C = np.cov(np.transpose(data))
    
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
        
    
        