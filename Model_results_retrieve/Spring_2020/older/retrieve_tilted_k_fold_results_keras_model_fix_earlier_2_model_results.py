# -*- coding: utf-8 -*-
"""
Created on %31-Jan-2020 at 11.00A.m

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
model_name="Brain_inception_only_results"
#model_name="Brain_inception_residual_results/again"
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
#        if len(test_probabilities[i])<len(test_PDBs[i]):
#            test_PDBs[i] = 
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
        if OG_test_probs_t[i][0]> OG_test_probs_t[i][1] and OG_test_probs_t[i][0]> OG_test_probs_t[i][2]:
            OG_validation_correct=OG_validation_correct+1
        OG_PDBs_validated.append(OG_test_PDBs_t[i])
    k=k+ len(OG_test_probs_t)
print("Over all OG 10fold cross validation accuracy: ",round(100*OG_validation_correct/test_OG_len,2),' %')

TSG_PDBs_validated=[]
TSG_test_probs_numpy=np.empty((test_TSG_len,3))
TSG_validation_correct=0
k=0
for j in range(0,len(TSG_test_probs)):
    TSG_test_probs_t = deepcopy(TSG_test_probs[j])
    TSG_test_PDBs_t = deepcopy(TSG_test_PDBs[j])
    for i in range(0,len(TSG_test_probs_t)):
        TSG_test_probs_numpy[k+i,:]=deepcopy(TSG_test_probs_t[i])
        if TSG_test_probs_t[i][0]< TSG_test_probs_t[i][1] and TSG_test_probs_t[i][1]> TSG_test_probs_t[i][2]:
            TSG_validation_correct=TSG_validation_correct+1
        TSG_PDBs_validated.append(TSG_test_PDBs_t[i])
    k=k+ len(TSG_test_probs_t)
print("Over all TSG 10fold cross validation accuracy: ",round(100*TSG_validation_correct/test_TSG_len,2),' %')

Fusion_PDBs_validated=[]
Fusion_test_probs_numpy=np.empty((test_Fusion_len,3))
Fusion_validation_correct=0
k=0
for j in range(0,len(Fusion_test_probs)):
    Fusion_test_probs_t = deepcopy(Fusion_test_probs[j])
    Fusion_test_PDBs_t = deepcopy(Fusion_test_PDBs[j])
    for i in range(0,len(Fusion_test_probs_t)):
        Fusion_test_probs_numpy[k+i,:]=deepcopy(Fusion_test_probs_t[i])
        if Fusion_test_probs_t[i][2]> Fusion_test_probs_t[i][1] and Fusion_test_probs_t[i][0]< Fusion_test_probs_t[i][2]:
            Fusion_validation_correct=Fusion_validation_correct+1
        Fusion_PDBs_validated.append(Fusion_test_PDBs_t[i])
    k=k+ len(Fusion_test_probs_t)
print("Over all Fusion 10fold cross validation accuracy: ",round(100*Fusion_validation_correct/test_Fusion_len,2),' %')
print("Over all All 10fold cross validation accuracy: ",round(100*(OG_validation_correct+TSG_validation_correct+Fusion_validation_correct)/(test_OG_len+test_TSG_len+test_Fusion_len),2),' %')

#%%
os.chdir('/')
os.chdir(saving_dir)

np.save('OG_10_fold_probs.npy', OG_test_probs_numpy)
np.save('TSG_10_fold_probs.npy', TSG_test_probs_numpy)
np.save('Fusion_10_fold_probs.npy', Fusion_test_probs_numpy)

pickle.dump(OG_PDBs_validated, open("OG_10_fold_PDBs.p", "wb"))  
pickle.dump(TSG_PDBs_validated, open("TSG_10_fold_PDBs.p", "wb")) 
pickle.dump(Fusion_PDBs_validated, open("Fusion_10_fold_PDBs.p", "wb"))  

#%%

''' check the resy'''
#%%
def gene_type_extractor(chk_name):
    '''
    Just helper function check the portion of last name and decide type
    '''
    if chk_name=='on':
        type_g = 'Fusion'
    elif chk_name=='OG':
        type_g = 'ONGO'
    elif chk_name=='SG':
        type_g = 'TSG'
    else:
        print('Some thing went wrong: fix the file name format')
        raise ValueError
    return type_g


def file_r_stack(name):
    ''' This function read the .txt file and return the rows as lists'''
    f = open(name)
    csv_f = csv.reader(f)
    
    stack=[]
    for row in csv_f:
        stack.append(row)
    r_stack=[]    
    for ele in stack:
        pdb_name = ele[0][-8:-4]
    #        print(pdb_name)
        temp = [pdb_name]
        for i in range(1,len(ele)):
            temp.append(ele[i])
        r_stack.append(temp)
    return r_stack

#%%
'''average the probabilities and check the results'''
def r_stact_to_prob(r_stack):
    '''
    This funstion extract the Probabilities from the list
    '''
    pdbs=[]
    probabilities=np.empty((len(r_stack)-1,3))
    for k in range(1,len(r_stack)):
        pdbs.append(r_stack[k][0])
        probabilities[k-1,:]=deepcopy(np.float_(r_stack[k][1:4]))
    return pdbs,probabilities

# to form the list of PDBs first time
chk_OG =True
chk_TSG=True
chk_Fusion=True

os.chdir('/')
os.chdir(nn_probability)

list_files=os.listdir()
#%%
# for checking the results without mean and std deviation
for i in range(0,len(list_files)):
    if list_files[i][0] == 'w':
        name = list_files[i]
        file_type_g  = gene_type_extractor(name[-8:-6])
        r_stack=file_r_stack(name)      
    
        if file_type_g=='ONGO':
            if chk_OG:
                OG_combined_probabilities=np.empty((6,len(r_stack)-1,3))
                OG_pdbs, probabilities = r_stact_to_prob(r_stack)
                OG_combined_probabilities[int(name[-5])-1,:,:] =deepcopy(probabilities)
                chk_OG=False
            else:
                _, probabilities = r_stact_to_prob(r_stack)
                OG_combined_probabilities[int(name[-5])-1,:,:] =deepcopy(probabilities)
            del r_stack
            del probabilities
        elif file_type_g=='TSG':
            if chk_TSG:
                TSG_combined_probabilities=np.empty((6,len(r_stack)-1,3))
                TSG_pdbs, probabilities = r_stact_to_prob(r_stack)
                TSG_combined_probabilities[int(name[-5])-1,:,:] =deepcopy(probabilities)
                chk_TSG=False
            else:
                _, probabilities = r_stact_to_prob(r_stack)
                TSG_combined_probabilities[int(name[-5])-1,:,:] =deepcopy(probabilities)
            del r_stack
            del probabilities
        elif file_type_g=='Fusion':
            if chk_Fusion:
                Fusion_combined_probabilities=np.empty((6,len(r_stack)-1,3))
                Fusion_pdbs, probabilities = r_stact_to_prob(r_stack)
                Fusion_combined_probabilities[int(name[-5])-1,:,:] =deepcopy(probabilities)
                chk_Fusion=False
            else:
                _, probabilities = r_stact_to_prob(r_stack)
                Fusion_combined_probabilities[int(name[-5])-1,:,:] =deepcopy(probabilities)
            del r_stack
            del probabilities
#%% Then check the average probabilities
def probailities_to_predicted_class(combined_probabilities,file_type_g):
    '''This function retrive the average probabilities'''
    
    probabilities = np.mean(combined_probabilities, axis=0)   
    class_list=['ONGO','TSG','Fusion']
    assigned_class=[]
    count=0
    for k in range(0,len(probabilities)):
        assigned_class.append(class_list[np.argmax(probabilities[k,:])])
        if file_type_g==class_list[np.argmax(probabilities[k,:])]:
            count=count+1
    print('Accuracy of ',file_type_g, ': ',100*count/len(probabilities))
    return assigned_class

ONGO_assigned_classes = probailities_to_predicted_class(OG_combined_probabilities,'ONGO')
TSG_assigned_classes = probailities_to_predicted_class(TSG_combined_probabilities,'TSG')
Fusion_assigned_classes = probailities_to_predicted_class(Fusion_combined_probabilities,'Fusion')

#%%
'''
To calculate the confusion matrix for each direction
'''
#%count the numberof predicted outcomes for each gene type
def count_classified_type(r_stack):
    '''count the numberof predicted outcomes for each gene type'''
    O_count=0#number of PDBs classified as ONGO
    T_count=0#number of PDBs classified as TSG
    F_count=0#number of PDBs classified as Fusion
    for j in range(1,len(r_stack)):
        if gene_type_extractor(r_stack[j][4][-2:])=='ONGO':
            O_count=O_count+1
        elif gene_type_extractor(r_stack[j][4][-2:])=='TSG':
            T_count=T_count+1
        elif gene_type_extractor(r_stack[j][4][-2:])=='Fusion':
            F_count=F_count+1
        else:
            print('Some thing went wrong: fix the file name format')
            raise ValueError
#    print('number of ONGO classified PDBs: ',O_count)
#    print('number of TSG classified PDBs: ',T_count)
#    print('number of Fusion classified PDBs: ',F_count)
    return np.array([O_count,T_count,F_count],dtype=int)
#%
os.chdir('/')
os.chdir(nn_probability)
confusion_matrix=np.empty((6,3,3),dtype=int)

list_files=os.listdir()
# for checking the results without mean and std deviation
for i in range(0,len(list_files)):
    if list_files[i][0] == 'w':
        name = list_files[i]
        #print(file_name)
        # to get the titled axis direction
#        print(name[-5])
        # to get the type
#        print(name[-8:-6])
        file_type_g  = gene_type_extractor(name[-8:-6])
        r_stack=file_r_stack(name)    

        if file_type_g=='ONGO':
            confusion_matrix[int(name[-5])-1,0,:] = deepcopy(count_classified_type(r_stack))
            del r_stack
        elif file_type_g=='TSG':
            confusion_matrix[int(name[-5])-1,1,:] = deepcopy(count_classified_type(r_stack))
        elif file_type_g=='Fusion':
            confusion_matrix[int(name[-5])-1,2,:] = deepcopy(count_classified_type(r_stack))
            del r_stack

#%
diagonal=0
diagonal_max=0
selected_axis=7
for i in range(0,6):
    diagonal=   confusion_matrix[i,0,0]+confusion_matrix[i,1,1]+confusion_matrix[i,2,2]
    if diagonal_max<diagonal:
        selected_axis=i+1
        diagonal_max=diagonal
        total=sum(sum(confusion_matrix[i,:,:]))
print('Axis selected: ',selected_axis)
print('Total: ',total)
print('diagonal_max: ',diagonal_max)
print('total: ',total)
print('Accuracy: ',100*diagonal_max/total)