# -*- coding: utf-8 -*-
"""
Created on %15-Nov-2019 at 3.40pm

@author: %A.Nishanth C00294860
"""
import os
import numpy as np
import csv
from copy import deepcopy
# make sure the 
#from class_clean_probabilty_weigtht_cal_results_from_NN_test  import clean_probability_weight_cal_results_from_NN
#%%
minimum_prob_decide = 0.25 
#% First change the results given by Torch to only the pdb_id formating
#working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/"
#working_dir_part = "C:/Users/c00294860/Documents/Projects_UL/Continues BIBM/"
SITE_MUST = True
clean = True

Tier = 1
"""FOR clean new_ran_clean"""
nn_probability='F:/PDBs_tilted/optimaly_tilted/test_results'
#nn_probability='F:/PDBs_tilted/tilted/test_tilted_results'
#working_dir_NN_prob = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_NN_results"])
#unigene_test_dir = "".join([working_dir_part,"a_Results_for_model_train/Tier_",str(Tier),"/clean"])
#saving_dir_part = "".join([working_dir_part, "a_Results_for_model_train/Tier_",str(Tier),"/clean_Finalised_probabilities_gene/"])
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