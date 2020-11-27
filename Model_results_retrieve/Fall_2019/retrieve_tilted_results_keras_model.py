# -*- coding: utf-8 -*-
"""
Created on %29-Dec-2019 at 3.30pm

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
nn_probability='F:/Suceed_BIBM_1/BIBM_81_1084/rest_pikles'
#nn_probability='F:/PDBs_tilted/tilted/test_tilted_results'
#working_dir_NN_prob = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_NN_results"])
#unigene_test_dir = "".join([working_dir_part,"a_Results_for_model_train/Tier_",str(Tier),"/clean"])
saving_dir='F:/Suceed_BIBM_1/BIBM_81_1084/extract_results'
#%%
os.chdir('/')
os.chdir(nn_probability)
train_probabilities = pickle.load(open("train_test_probabilities.p", "rb"))
train_PDBs= pickle.load(open("pdbs_train_test.p", "rb"))
test_probabilities = pickle.load(open("test_probabilities.p", "rb"))
test_PDBs= pickle.load(open("pdbs_test.p", "rb"))

#for organise the results or performance
OG_train_PDBs=  pickle.load(open("OG_train_PDBs.p", "rb"))
TSG_train_PDBs=  pickle.load(open("TSG_train_PDBs.p", "rb"))    
Fusion_train_PDBs=  pickle.load(open("Fusion_train_PDBs.p", "rb"))

overlapped_PDBs_ONGO= pickle.load(open("overlapped_PDBs_ONGO.p", "rb"))
overlapped_PDBs_TSG= pickle.load(open("overlapped_PDBs_TSG.p", "rb"))
valid_ids= pickle.load(open("valid_ids.p", "rb"))

test_labels_dic= pickle.load(open("test_labels_dic.p", "rb"))

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
#% Finalise the Test results
#sum(a, [])#to make as one list
'''Extract the test results'''
OG_test_probs=[]
OG_test_PDBs=[]
TSG_test_probs=[]
TSG_test_PDBs=[]
Fusion_test_probs=[]
Fusion_test_PDBs=[]
for i in range(0,len(test_PDBs)):
    for j in range(0,len(test_PDBs[i])):
        if test_labels_dic[test_PDBs[i][j]]==0 and (test_PDBs[i][j] not in overlapped_PDBs_ONGO):
            OG_test_probs.append(prob_fix_func(deepcopy(test_probabilities[i][j,:])))
            OG_test_PDBs.append(test_PDBs[i][j])
        elif test_labels_dic[test_PDBs[i][j]]==1 and (test_PDBs[i][j] not in overlapped_PDBs_TSG):
            TSG_test_probs.append(prob_fix_func(deepcopy(test_probabilities[i][j,:])))
            TSG_test_PDBs.append(test_PDBs[i][j])
        elif test_labels_dic[test_PDBs[i][j]]==2:
            Fusion_test_probs.append(prob_fix_func(deepcopy(test_probabilities[i][j,:])))
            Fusion_test_PDBs.append(test_PDBs[i][j])
        else:
            raise("Some thing wrong")
del test_probabilities
OG_test_probs_numpy=np.empty((len(OG_test_probs),3))
for i in range(0,len(OG_test_probs)):
    OG_test_probs_numpy[i,:]=deepcopy(OG_test_probs[i])
del OG_test_probs

TSG_test_probs_numpy=np.empty((len(TSG_test_probs),3))
for i in range(0,len(TSG_test_probs)):
    TSG_test_probs_numpy[i,:]=deepcopy(TSG_test_probs[i])
del TSG_test_probs

Fusion_test_probs_numpy=np.empty((len(Fusion_test_probs),3))
for i in range(0,len(Fusion_test_probs)):
    Fusion_test_probs_numpy[i,:]=deepcopy(Fusion_test_probs[i])
del Fusion_test_probs
    
#%%
os.chdir('/')
os.chdir(saving_dir)

np.save('OG_test_probs.npy', OG_test_probs_numpy)
np.save('TSG_test_probs.npy', TSG_test_probs_numpy)
np.save('Fusion_test_probs.npy', Fusion_test_probs_numpy)

pickle.dump(OG_test_PDBs, open("OG_test_PDBs.p", "wb"))  
pickle.dump(TSG_test_PDBs, open("TSG_test_PDBs.p", "wb")) 
pickle.dump(Fusion_test_PDBs, open("Fusion_test_PDBs.p", "wb"))  
#%%
'''Retrieve the results of Training set'''
OG_train_probs=[]
OG_train_PDBs_fin=[]
TSG_train_probs=[]
TSG_train_PDBs_fin=[]
Fusion_train_probs=[]
Fusion_train_PDBs_fin=[]
OG_Fusion_overlap_train_PDBs=[]
TSG_Fusion_overlap_train_PDBs=[]
OG_Fusion_overlap_train_prob=[]
TSG_Fusion_overlap_train_prob=[]

OG_valid_probs=[]
OG_valid_PDBs_fin=[]
TSG_valid_probs=[]
TSG_valid_PDBs_fin=[]
Fusion_valid_probs=[]
Fusion_valid_PDBs_fin=[]
OG_Fusion_overlap_valid_PDBs=[]
TSG_Fusion_overlap_valid_PDBs=[]
OG_Fusion_overlap_valid_prob=[]
TSG_Fusion_overlap_valid_prob=[]

for i in range(0,len(train_PDBs)):
    for j in range(0,len(train_PDBs[i])):
        if train_PDBs[i][j] in valid_ids:
            if (train_PDBs[i][j] in OG_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_ONGO):
                OG_valid_probs.append(prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                OG_valid_PDBs_fin.append(train_PDBs[i][j])
            elif (train_PDBs[i][j] in TSG_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_TSG):
                TSG_valid_probs.append(prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                TSG_valid_PDBs_fin.append(train_PDBs[i][j])
            elif  (train_PDBs[i][j] in Fusion_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_ONGO) and (train_PDBs[i][j] not in overlapped_PDBs_TSG) :
                Fusion_valid_probs.append(prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                Fusion_valid_PDBs_fin.append(train_PDBs[i][j])
            elif (train_PDBs[i][j] in overlapped_PDBs_ONGO):
                OG_Fusion_overlap_valid_prob.append(prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                OG_Fusion_overlap_valid_PDBs.append(train_PDBs[i][j])
            elif (train_PDBs[i][j] in overlapped_PDBs_TSG):
                TSG_Fusion_overlap_valid_prob.append(prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                TSG_Fusion_overlap_valid_PDBs.append(train_PDBs[i][j])
            else:
                print(train_PDBs[i][j])
        else:
            if (train_PDBs[i][j] in OG_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_ONGO) and (train_PDBs[i][j] not in OG_train_PDBs_fin):
                OG_train_probs.append(prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                OG_train_PDBs_fin.append(train_PDBs[i][j])
            elif (train_PDBs[i][j] in TSG_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_TSG) and (train_PDBs[i][j] not in TSG_train_PDBs_fin):
                TSG_train_probs.append(prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                TSG_train_PDBs_fin.append(train_PDBs[i][j])
            elif  (train_PDBs[i][j] in Fusion_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_ONGO) and (train_PDBs[i][j] not in overlapped_PDBs_TSG) and (train_PDBs[i][j] not in Fusion_train_PDBs_fin) :
                Fusion_train_probs.append(prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                Fusion_train_PDBs_fin.append(train_PDBs[i][j])
            elif (train_PDBs[i][j] in overlapped_PDBs_ONGO) and (train_PDBs[i][j] not in OG_Fusion_overlap_train_PDBs):
                OG_Fusion_overlap_train_prob.append(prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                OG_Fusion_overlap_train_PDBs.append(train_PDBs[i][j])
            elif (train_PDBs[i][j] in overlapped_PDBs_TSG) and (train_PDBs[i][j] not in TSG_Fusion_overlap_train_PDBs):
                TSG_Fusion_overlap_train_prob.append(prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                TSG_Fusion_overlap_train_PDBs.append(train_PDBs[i][j])
            else:
                print('----------------------')
                print(train_PDBs[i][j])
            
del train_probabilities
OG_train_probs_numpy=np.empty((len(OG_train_probs),3))
for i in range(0,len(OG_train_probs)):
    OG_train_probs_numpy[i,:]=deepcopy(OG_train_probs[i])
del OG_train_probs

TSG_train_probs_numpy=np.empty((len(TSG_train_probs),3))
for i in range(0,len(TSG_train_probs)):
    TSG_train_probs_numpy[i,:]=deepcopy(TSG_train_probs[i])
del TSG_train_probs

Fusion_train_probs_numpy=np.empty((len(Fusion_train_probs),3))
for i in range(0,len(Fusion_train_probs)):
    Fusion_train_probs_numpy[i,:]=deepcopy(Fusion_train_probs[i])
del Fusion_train_probs

OG_Fusion_overlap_train_probs_numpy=np.empty((len(OG_Fusion_overlap_train_prob),3))
for i in range(0,len(OG_Fusion_overlap_train_prob)):
    OG_Fusion_overlap_train_probs_numpy[i,:]=deepcopy(OG_Fusion_overlap_train_prob[i])
del OG_Fusion_overlap_train_prob

TSG_Fusion_overlap_train_probs_numpy=np.empty((len(TSG_Fusion_overlap_train_prob),3))
for i in range(0,len(TSG_Fusion_overlap_train_prob)):
    TSG_Fusion_overlap_train_probs_numpy[i,:]=deepcopy(TSG_Fusion_overlap_train_prob[i])
del TSG_Fusion_overlap_train_prob
#%%
OG_valid_probs_numpy=np.empty((len(OG_valid_probs),3))
for i in range(0,len(OG_valid_probs)):
    OG_valid_probs_numpy[i,:]=deepcopy(OG_valid_probs[i])
del OG_valid_probs

TSG_valid_probs_numpy=np.empty((len(TSG_valid_probs),3))
for i in range(0,len(TSG_valid_probs)):
    TSG_valid_probs_numpy[i,:]=deepcopy(TSG_valid_probs[i])
del TSG_valid_probs

Fusion_valid_probs_numpy=np.empty((len(Fusion_valid_probs),3))
for i in range(0,len(Fusion_valid_probs)):
    Fusion_valid_probs_numpy[i,:]=deepcopy(Fusion_valid_probs[i])
del Fusion_valid_probs

OG_Fusion_overlap_valid_probs_numpy=np.empty((len(OG_Fusion_overlap_valid_prob),3))
for i in range(0,len(OG_Fusion_overlap_valid_prob)):
    OG_Fusion_overlap_valid_probs_numpy[i,:]=deepcopy(OG_Fusion_overlap_valid_prob[i])
del OG_Fusion_overlap_valid_prob

TSG_Fusion_overlap_valid_probs_numpy=np.empty((len(TSG_Fusion_overlap_valid_prob),3))
for i in range(0,len(TSG_Fusion_overlap_valid_prob)):
    TSG_Fusion_overlap_valid_probs_numpy[i,:]=deepcopy(TSG_Fusion_overlap_valid_prob[i])
del TSG_Fusion_overlap_valid_prob
#%%
np.save('OG_valid_probs.npy', OG_valid_probs_numpy)
np.save('TSG_valid_probs.npy', TSG_valid_probs_numpy)
np.save('Fusion_valid_probs.npy', Fusion_valid_probs_numpy)
np.save('OG_Fusion_overlap_valid_probs.npy', OG_Fusion_overlap_valid_probs_numpy)
np.save('TSG_Fusion_overlap_valid_probs.npy', TSG_Fusion_overlap_valid_probs_numpy)
pickle.dump(OG_valid_PDBs_fin, open("OG_valid_PDBs.p", "wb"))  
pickle.dump(TSG_valid_PDBs_fin, open("TSG_valid_PDBs.p", "wb")) 
pickle.dump(Fusion_valid_PDBs_fin, open("Fusion_valid_PDBs.p", "wb"))  
pickle.dump(OG_Fusion_overlap_valid_PDBs, open("OG_Fusion_overlap_valid_PDBs.p", "wb")) 
pickle.dump(TSG_Fusion_overlap_valid_PDBs, open("TSG_Fusion_overlap_valid_PDBs.p", "wb"))  

np.save('OG_train_probs.npy', OG_train_probs_numpy)
np.save('TSG_train_probs.npy', TSG_train_probs_numpy)
np.save('Fusion_train_probs.npy', Fusion_train_probs_numpy)
np.save('OG_Fusion_overlap_train_probs.npy', OG_Fusion_overlap_train_probs_numpy)
np.save('TSG_Fusion_overlap_train_probs.npy', TSG_Fusion_overlap_train_probs_numpy)

pickle.dump(OG_train_PDBs_fin, open("OG_train_PDBs.p", "wb"))  
pickle.dump(TSG_train_PDBs_fin, open("TSG_train_PDBs.p", "wb")) 
pickle.dump(Fusion_train_PDBs_fin, open("Fusion_train_PDBs.p", "wb"))  
pickle.dump(OG_Fusion_overlap_train_PDBs, open("OG_Fusion_overlap_train_PDBs.p", "wb")) 
pickle.dump(TSG_Fusion_overlap_train_PDBs, open("TSG_Fusion_overlap_train_PDBs.p", "wb"))  

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