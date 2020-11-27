# -*- coding: utf-8 -*-
"""
Created on %30-Jan-2020 at 11.00A.m

@author: %A.Nishanth C00294860
"""
import os
import numpy as np
import csv
from copy import deepcopy
import pickle

# make sure the 
#from class_clean_probabilty_weigtht_cal_results_from_NN_test  import clean_probability_weight_cal_results_from_NN
#%%
minimum_prob_decide = 0.25 
#% First change the results given by Torch to only the pdb_id formating
SITE_MUST = True
clean = True
Tier = 1
"""FOR clean new_ran_clean"""
#model_name="Brain_inception_only_results"
#model_name="Brain_inception_residual_results/again"
#model_name="Brain_inception_residual_results/reran"
model_name="Brain_inception_residual_results/fixed"
nn_probability=''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/',model_name])
main_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results"
saving_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Finalised_results/',model_name])
#%%
os.chdir('/')
os.chdir(main_dir)
#for organise the results or performance
OG_train_PDBs=  pickle.load(open("OG_train_PDBs.p", "rb"))
TSG_train_PDBs=  pickle.load(open("TSG_train_PDBs.p", "rb"))    
Fusion_train_PDBs=  pickle.load(open("Fusion_train_PDBs.p", "rb"))

overlapped_PDBs_ONGO= pickle.load(open("overlapped_PDBs_ONGO.p", "rb"))
overlapped_PDBs_TSG= pickle.load(open("overlapped_PDBs_TSG.p", "rb"))
test_labels_dic= pickle.load(open("train_labels_dic.p", "rb"))



def prob_fix_func(prob,round_fac=2):
    '''
    This function fix the probability added to one and round up the probability
    '''
    prob = np.round(prob,2)
    if sum(prob)==1:
        return prob
    else:
        all_sum=sum(prob)
        prob[0] = prob[0]/all_sum
        prob[1] = prob[1]/all_sum
        prob[2] = prob[2]/all_sum
        return np.round(prob,2)
#%%
os.chdir(nn_probability)

splited_clean_ONGO =  pickle.load(open("10_splited_clean_ONGO.p", "rb"))
splited_clean_TSG =  pickle.load(open("10_splited_clean_TSG.p", "rb"))
splited_clean_Fusion =  pickle.load(open("10_splited_clean_Fusion.p", "rb"))

#%%
print("")
print("")
print("")

print("start from 0 and end at 10")
print("")
print("")
OG_test_probs=[]
OG_test_PDBs=[]
TSG_test_probs=[]
TSG_test_PDBs=[]
Fusion_test_probs=[]
Fusion_test_PDBs=[]
test_OG_len=0
test_TSG_len=0
test_Fusion_len=0
for nth_fold in range(0,10):
#for nth_fold in range(2,7):
    os.chdir('/')
    os.chdir(nn_probability)
    
    #splited_clean_Fusion =  pickle.load(open("10_splited_clean_Fusion.p", "rb"))
    #splited_clean_ONGO =  pickle.load(open("10_splited_clean_ONGO.p", "rb"))
    #splited_clean_TSG =  pickle.load(open("10_splited_clean_TSG.p", "rb"))

    test_PDBs= pickle.load(open(''.join(["pdbs_valid_fold_",str(nth_fold),".p"]), "rb"))
    test_probabilities = pickle.load(open(''.join(["valid_probabilities_fold_",str(nth_fold),".p"]), "rb"))
    
    #% Finalise the Test results
    #sum(a, [])#to make as one list
    '''Extract the test results'''
    OG_test_probs_t=[]
    OG_test_PDBs_t=[]
    TSG_test_probs_t=[]
    TSG_test_PDBs_t=[]
    Fusion_test_probs_t=[]
    Fusion_test_PDBs_t=[]
    for i in range(0,len(test_PDBs)):
        for j in range(0,len(test_PDBs[i])):
            if test_labels_dic[test_PDBs[i][j]]==0 and (test_PDBs[i][j] not in overlapped_PDBs_ONGO):
                OG_test_probs_t.append(prob_fix_func(deepcopy(test_probabilities[i][j,:])))
                OG_test_PDBs_t.append(test_PDBs[i][j])
            elif test_labels_dic[test_PDBs[i][j]]==1 and (test_PDBs[i][j] not in overlapped_PDBs_TSG):
                TSG_test_probs_t.append(prob_fix_func(deepcopy(test_probabilities[i][j,:])))
                TSG_test_PDBs_t.append(test_PDBs[i][j])
            elif test_labels_dic[test_PDBs[i][j]]==2:
                Fusion_test_probs_t.append(prob_fix_func(deepcopy(test_probabilities[i][j,:])))
                Fusion_test_PDBs_t.append(test_PDBs[i][j])
            else:
                raise("Some thing wrong")
                
    test_OG_len=test_OG_len + len(OG_test_probs_t)
    test_TSG_len=test_TSG_len + len(TSG_test_probs_t)
    test_Fusion_len=test_Fusion_len + len(Fusion_test_probs_t)
       
    OG_test_probs.append(deepcopy(OG_test_probs_t))
    OG_test_PDBs.append(deepcopy(OG_test_PDBs_t))
    TSG_test_probs.append(deepcopy(TSG_test_probs_t))
    TSG_test_PDBs.append(deepcopy(TSG_test_PDBs_t))
    Fusion_test_probs.append(deepcopy(Fusion_test_probs_t))
    Fusion_test_PDBs.append(deepcopy(Fusion_test_PDBs_t))
    del test_probabilities
    del test_PDBs
#%%

OG_PDBs_validated=[]
OG_test_probs_numpy=np.empty((test_OG_len,3))
OG_validation_correct=0
k=0
for j in range(0,len(OG_test_probs)):
    OG_test_probs_t = deepcopy(OG_test_probs[j])
    OG_test_PDBs_t = deepcopy(OG_test_PDBs[j])
    for i in range(0,len(OG_test_probs_t)):
        OG_test_probs_numpy[k+i,:]=deepcopy(OG_test_probs_t[i])
        if OG_test_probs_t[i][0]>= OG_test_probs_t[i][1] and OG_test_probs_t[i][0]>= OG_test_probs_t[i][2]:
            OG_validation_correct=OG_validation_correct+1
        OG_PDBs_validated.append(OG_test_PDBs_t[i])
    k=k+ len(OG_test_probs_t)
print("Overall ONGO 10-fold cross validation accuracy: ",round(100*OG_validation_correct/test_OG_len,2),' %')

TSG_PDBs_validated=[]
TSG_test_probs_numpy=np.empty((test_TSG_len,3))
TSG_validation_correct=0
k=0
for j in range(0,len(TSG_test_probs)):
    TSG_test_probs_t = deepcopy(TSG_test_probs[j])
    TSG_test_PDBs_t = deepcopy(TSG_test_PDBs[j])
    for i in range(0,len(TSG_test_probs_t)):
        TSG_test_probs_numpy[k+i,:]=deepcopy(TSG_test_probs_t[i])
        if TSG_test_probs_t[i][0]<= TSG_test_probs_t[i][1] and TSG_test_probs_t[i][1]>= TSG_test_probs_t[i][2]:
            TSG_validation_correct=TSG_validation_correct+1
        TSG_PDBs_validated.append(TSG_test_PDBs_t[i])
    k=k+ len(TSG_test_probs_t)
print("Overall TSG 10-fold cross validation accuracy: ",round(100*TSG_validation_correct/test_TSG_len,2),' %')

Fusion_PDBs_validated=[]
Fusion_test_probs_numpy=np.empty((test_Fusion_len,3))
Fusion_validation_correct=0
k=0
for j in range(0,len(Fusion_test_probs)):
    Fusion_test_probs_t = deepcopy(Fusion_test_probs[j])
    Fusion_test_PDBs_t = deepcopy(Fusion_test_PDBs[j])
    for i in range(0,len(Fusion_test_probs_t)):
        Fusion_test_probs_numpy[k+i,:]=deepcopy(Fusion_test_probs_t[i])
        if Fusion_test_probs_t[i][2]>= Fusion_test_probs_t[i][1] and Fusion_test_probs_t[i][0]<= Fusion_test_probs_t[i][2]:
            Fusion_validation_correct=Fusion_validation_correct+1
        Fusion_PDBs_validated.append(Fusion_test_PDBs_t[i])
    k=k+ len(Fusion_test_probs_t)
print("Overall Fusion 10-fold cross validation accuracy: ",round(100*Fusion_validation_correct/test_Fusion_len,2),' %')
print("Overall combined 10-fold cross validation accuracy: ",round(100*(OG_validation_correct+TSG_validation_correct+Fusion_validation_correct)/(test_OG_len+test_TSG_len+test_Fusion_len),2),' %')

#%%
os.chdir('/')
if not os.path.isdir(saving_dir):
    os.makedirs(saving_dir)
os.chdir('/')
os.chdir(saving_dir)

np.save('OG_10_fold_probs.npy', OG_test_probs_numpy)
np.save('TSG_10_fold_probs.npy', TSG_test_probs_numpy)
np.save('Fusion_10_fold_probs.npy', Fusion_test_probs_numpy)

pickle.dump(OG_PDBs_validated, open("OG_10_fold_PDBs.p", "wb"))  
pickle.dump(TSG_PDBs_validated, open("TSG_10_fold_PDBs.p", "wb")) 
pickle.dump(Fusion_PDBs_validated, open("Fusion_10_fold_PDBs.p", "wb"))  

#%% Find the confusion matrix
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
confusion_matrix[:,0]=[ONGO_count_OG,ONGO_count_TSG,ONGO_count_Fusion]
confusion_matrix[:,1]=[TSG_count_OG,TSG_count_TSG,TSG_count_Fusion]
confusion_matrix[:,2]=[Fusion_count_OG,Fusion_count_TSG,Fusion_count_Fusion]

#%%
import os
import numpy as np
from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
#model_name="Brain_inception_residual_results/again"
model_name="Brain_inception_residual_results/reran"

loading_dir = ''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results/2020_k_fold_results/Finalised_results/',model_name])
os.chdir('/')
os.chdir(loading_dir)
OG_test_probs = np.load('OG_10_fold_probs.npy')
TSG_test_probs = np.load('TSG_10_fold_probs.npy')
Fusion_test_probs =np.load('Fusion_10_fold_probs.npy')
#
#OG_PDBs_validated= pickle.load(open("OG_10_fold_PDBs.p", "rb"))  
#TSG_PDBs_validated= pickle.load(open("TSG_10_fold_PDBs.p", "rb")) 
#Fusion_PDBs_validated= pickle.load(open("Fusion_10_fold_PDBs.p", "rb")) 
#%%

'''ploting ROC curve slected classes
Like ONGO Vs TSG & Fusion
     TSG Vs TSG & Fusion
     Fusion Vs TSG & ONGO
'''
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
        if (probabilities[i][class_sel_1]+ probabilities[i][class_sel_2])==0:
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

#'''ONGO VS TSG only'''
#ONGO_y_score,ONGO_y_test,OG_correct = get_scores_and_assign_n_to_2(OG_test_probs,0,0,1)
#TSG_y_score,TSG_y_test,TSG_correct = get_scores_and_assign_n_to_2(TSG_test_probs,1,0,1)
#y_test = np.concatenate((ONGO_y_test, TSG_y_test), axis=0)
#y_score = np.concatenate((ONGO_y_score, TSG_y_test), axis=0)
#confusion_matrix_binary=np.empty((2,2))
#confusion_matrix_binary[0,:]=[OG_correct,len(OG_test_probs)-OG_correct]
#confusion_matrix_binary[1,:]=[len(TSG_test_probs)-TSG_correct,TSG_correct]
#print("Only ONGO vs TSG over all accuracy: ",round(100*(OG_correct+TSG_correct)/(len(OG_test_probs)+len(TSG_test_probs)),2),'%')

#'''ONGO VS Fusion only'''
#ONGO_y_score,ONGO_y_test,OG_correct = get_scores_and_assign_n_to_2(OG_test_probs,0,0,2)
#Fusion_y_score,Fusion_y_test, Fusion_correct = get_scores_and_assign_n_to_2(Fusion_test_probs,1,0,2)
#y_test = np.concatenate((ONGO_y_test, Fusion_y_test), axis=0)
#y_score = np.concatenate((ONGO_y_score, Fusion_y_test), axis=0)
#print("Only ONGO vs Fusion over all accuracy: ",round(100*(OG_correct+Fusion_correct)/(len(OG_test_probs)+len(Fusion_test_probs)),2),'%')
#confusion_matrix_binary=np.empty((2,2))
#confusion_matrix_binary[0,:]=[OG_correct,len(OG_test_probs)-OG_correct]
#confusion_matrix_binary[1,:]=[len(Fusion_test_probs)-Fusion_correct,Fusion_correct]

#'''TSG VS Fusion only'''
TSG_y_score,TSG_y_test,TSG_correct = get_scores_and_assign_n_to_2(TSG_test_probs,0,1,2)
Fusion_y_score,Fusion_y_test,Fusion_correct = get_scores_and_assign_n_to_2(Fusion_test_probs,1,1,2)
y_test = np.concatenate((TSG_y_test, Fusion_y_test), axis=0)
y_score = np.concatenate((TSG_y_score, Fusion_y_test), axis=0)
print("Only TSG vs Fusion over all accuracy: ",round(100*(TSG_correct+Fusion_correct)/(len(TSG_test_probs)+len(Fusion_test_probs)),2),'%')
confusion_matrix_binary=np.empty((2,2))
confusion_matrix_binary[0,:]=[TSG_correct,len(TSG_test_probs)-TSG_correct]
confusion_matrix_binary[1,:]=[len(Fusion_test_probs)-Fusion_correct,Fusion_correct]


n_classes=2
#%%
'''Plotting one class Vs all 
    like:ONGO VS TSG and Fusion  '''
#n_classes=3
#y_score=np.zeros((len(OG_test_probs)+len(TSG_test_probs)+len(Fusion_test_probs),3)) 
#y_test=np.zeros((len(OG_test_probs)+len(TSG_test_probs)+len(Fusion_test_probs),3),int) 
#k=0
#for i in range(0,len(OG_test_probs)):
#    y_score[k,:]=OG_test_probs[i]
#    y_test[k,0]=1
#    k=k+1
#for i in range(0,len(TSG_test_probs)):
#    y_score[k,:]=TSG_test_probs[i]
#    y_test[k,1]=1
#    k=k+1
#for i in range(0,len(Fusion_test_probs)):
#    y_score[k,:]=Fusion_test_probs[i]
#    y_test[k,2]=1
#    k=k+1
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
#    if i==0:
#        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                 label='ROC curve of ONGO (area = {1:0.3f})'
#                 ''.format(i, roc_auc[i]))
    if i==0:
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of TSG (area = {1:0.3f})'
                 ''.format(i, roc_auc[i]))
    elif i==1:
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
