# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""
import os
import pickle
os.chdir('/')
os.chdir('F:/NN_vin/MSMS_tool_after/SITE_must/optimaly_tilted/size_200_new_prop/results')
y_train=pickle.load(open("y_train.p", "rb"))
train_probabilities=pickle.load(open("train_probabilities.p","rb"))  
#pickle.dump(y_test, open("y_train.p", "wb"))  
#pickle.dump(pdbs_train, open("pdbs_train.p", "wb"))  
#pickle.dump(train_probabilities, open("train_probabilities.p", "wb"))  
#%%
import numpy as np
y_test=pickle.load(open("y_test.p", "rb"))
test_probabilities=np.load('test_prob.npy')    
classes= np.argmax(test_probabilities, axis = 1)
def softmax(x):

    """Compute softmax values for each sets of scores in x."""

    return np.exp(x) / np.sum(np.exp(x), axis=0)
test_probabilities_2=softmax(np.load('test_prob.npy'))   
classes_1= np.argmax(test_probabilities_2, axis = -1)
print(sum(y_test==classes_1)/30)