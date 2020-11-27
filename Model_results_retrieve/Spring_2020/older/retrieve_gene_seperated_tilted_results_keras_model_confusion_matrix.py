# -*- coding: utf-8 -*-
"""
Created on %31-Jan-2020 at 7.47pm

@author: %A.Nishanth C00294860
"""
#%%
import os
import numpy as np
from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
model_name="brain_inception_residual"
main_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_gene_seperated/"
#working_dir_NN_prob = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_NN_results"])
#unigene_test_dir = "".join([working_dir_part,"a_Results_for_model_train/Tier_",str(Tier),"/clean"])
loading_dir =''.join([main_dir,'Finalised_results/',model_name])
#%%
os.chdir('/')
os.chdir(loading_dir)
OG_test_probs_numpy = np.load('OG_test_probs.npy')
TSG_test_probs_numpy = np.load('TSG_test_probs.npy')
Fusion_test_probs_numpy =np.load('Fusion_test_probs.npy')
#OG_test_probs_numpy = np.load('OG_train_probs.npy')
#TSG_test_probs_numpy = np.load('TSG_train_probs.npy')
#Fusion_test_probs_numpy =np.load('Fusion_train_probs.npy')
#OG_test_probs_numpy = np.load('OG_valid_probs.npy')
#TSG_test_probs_numpy = np.load('TSG_valid_probs.npy')
#Fusion_test_probs_numpy =np.load('Fusion_valid_probs.npy')
''' confusion matrix'''
def confusion_matrix_row(list_1):
    count_OG=0
    count_TSG=0
    count_Fusion=0
    for i in range(0,len(list_1)):
        if list_1[i][0]>=list_1[i][1] and list_1[i][0]>=list_1[i][2]:
            count_OG=count_OG+1
        elif list_1[i][1]>=list_1[i][0] and list_1[i][1]>=list_1[i][2]:
            count_TSG=count_TSG+1
        elif list_1[i][2]>=list_1[i][0] and list_1[i][2]>=list_1[i][1]:
            count_Fusion=count_Fusion+1
    return count_OG,count_TSG,count_Fusion

ONGO_count_OG,ONGO_count_TSG,ONGO_count_Fusion=confusion_matrix_row(OG_test_probs_numpy)
Fusion_count_OG,Fusion_count_TSG,Fusion_count_Fusion=confusion_matrix_row(Fusion_test_probs_numpy)
TSG_count_OG,TSG_count_TSG,TSG_count_Fusion=confusion_matrix_row(TSG_test_probs_numpy)
confusion_matrix=np.empty((3,3))
confusion_matrix[0,:]=[ONGO_count_OG,ONGO_count_TSG,ONGO_count_Fusion]
confusion_matrix[1,:]=[TSG_count_OG,TSG_count_TSG,TSG_count_Fusion]
confusion_matrix[2,:]=[Fusion_count_OG,Fusion_count_TSG,Fusion_count_Fusion]

#%%
def get_scores_and_assign_n_to_2(probabilities,class_given,class_sel_1,class_sel_2):
    '''
    Mainly used ONGO vs TSG
    ploting ROC curve slected classes
    Eg: If ONGO Vs TSG just leaving Fusion
            then ONGO's class_given=0 & TSG's class_given=1
        class_sel_1: index of probability of class ONGO
        class_sel_2: index of probability of class TSG
    '''
    y_score=np.zeros((len(probabilities),2)) 
    y_test=np.zeros((len(probabilities),2),int) 
    correct=0
    for i in range(0,len(probabilities)):
        if sum(probabilities[i])==0:
            y_score[i,:]=[0.5,0.5]
        else:
            prob_1_normal = probabilities[i][class_sel_1]/sum(probabilities[i])
            prob_2_normal = probabilities[i][class_sel_2]/sum(probabilities[i])
            if (prob_1_normal+prob_2_normal)==0:
                y_score[i,:]=[0.5,0.5]
            else:
                prob_2_normal = probabilities[i][class_sel_2]/sum(probabilities[i])
            y_score[i,:]=[prob_1_normal/(prob_1_normal+prob_2_normal),prob_2_normal/(prob_1_normal+prob_2_normal)]
            if class_given==0 and (prob_1_normal/(prob_1_normal+prob_2_normal))>=(prob_2_normal/(prob_1_normal+prob_2_normal)):
                correct=correct+1
            elif class_given==1 and (prob_1_normal/(prob_1_normal+prob_2_normal))<(prob_2_normal/(prob_1_normal+prob_2_normal)):
                correct=correct+1
        y_test[i][class_given]=1
    return y_score, y_test,correct
OG_test_probs=OG_test_probs_numpy
TSG_test_probs=TSG_test_probs_numpy
Fusion_test_probs=Fusion_test_probs_numpy

'''ONGO VS TSG only'''
ONGO_y_score,ONGO_y_test,OG_correct = get_scores_and_assign_n_to_2(OG_test_probs,0,0,1)
TSG_y_score,TSG_y_test,TSG_correct = get_scores_and_assign_n_to_2(TSG_test_probs,1,0,1)
y_test = np.concatenate((ONGO_y_test, TSG_y_test), axis=0)
y_score = np.concatenate((ONGO_y_score, TSG_y_test), axis=0)
'''ONGO VS Fusion only'''
ONGO_y_score,ONGO_y_test = get_scores_and_assign_n_to_2(OG_test_probs,0,0,1)
Fusion_y_score,Fusion_y_test = get_scores_and_assign_n_to_2(Fusion_test_probs,1,0,2)
y_test = np.concatenate((ONGO_y_test, Fusion_y_test), axis=0)
y_score = np.concatenate((ONGO_y_score, Fusion_y_test), axis=0)
#TSG_y_score,TSG_y_test = get_scores_and_assign_n_to_2(TSG_test_probs,0,0,1)
#Fusion_y_score,Fusion_y_test = get_scores_and_assign_n_to_2(Fusion_test_probs,1,0,2)
#y_test = np.concatenate((TSG_y_test, Fusion_y_test), axis=0)
#y_score = np.concatenate((TSG_y_score, Fusion_y_test), axis=0)
n_classes=2
print("Only ONGO vs TSG over all accuracy: ",100*(OG_correct+TSG_correct)/(len(OG_test_probs)+len(TSG_test_probs)),'%')
#%%
'''
since none of the Fusion classes are classified so avoided to check
Plotting one class Vs all 
    like:ONGO VS TSG and Fusion  '''
n_classes=3
y_score=np.zeros((len(OG_test_probs)+len(TSG_test_probs)+len(Fusion_test_probs),3)) 
y_test=np.zeros((len(OG_test_probs)+len(TSG_test_probs)+len(Fusion_test_probs),3),int) 
k=0
for i in range(0,len(OG_test_probs)):
    y_score[k,:]=OG_test_probs[i]
    y_test[k,0]=1
    k=k+1
for i in range(0,len(TSG_test_probs)):
    y_score[k,:]=TSG_test_probs[i]
    y_test[k,1]=1
    k=k+1
for i in range(0,len(Fusion_test_probs)):
    y_score[k,:]=Fusion_test_probs[i]
    y_test[k,2]=1
    k=k+1
'''Since some PDBs belongs to fusion totally classified as ONGO we can't plot Fusion against TSG'''
#% Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve, auc

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw = 2

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# %then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.3f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.3f})'
               ''.format(roc_auc["macro"]),
         color='red', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    if i==0:
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of ONGO (area = {1:0.3f})'
                 ''.format(i, roc_auc[i]))
    elif i==1:
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of TSG (area = {1:0.3f})'
                 ''.format(i, roc_auc[i]))
    elif i==2:
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of Fusion (area = {1:0.3f})'
                 ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic to ONCO Vs TSG Vs -class')
plt.legend(loc="lower right")
plt.show()





