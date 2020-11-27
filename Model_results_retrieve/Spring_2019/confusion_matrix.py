# -*- coding: utf-8 -*-
"""
Created on %3-Mar-2019(9.27A.m)

@author: %A.Nishanth C00294860
"""
import os
import numpy as np
import csv
#%%

def confusion_mat_creator(confusion_matrix,model_part,name):
    file_name=''.join([model_part,name,'_train.csv'])
    os.chdir('/')
    os.chdir(checking_dir)
    f = open(file_name)
    csv_f = csv.reader(f)
    
    stack=[]
    for row in csv_f:
        stack.append(row)
    
    if name=='ONGO':
        k=0
    elif name=='TSG':
        k=1
    elif name=='Fusion':
        k=2
    else:
        print('name should be ONGO or TSG or Fusion')
        
    for i in range(1,len(stack)):
        if stack[i][4]=="OG":
            confusion_matrix[k][0]=confusion_matrix[k][0]+1
        elif stack[i][4]== "TSG":
            confusion_matrix[k][1]=confusion_matrix[k][1]+1
        elif stack[i][4]== "Fusion":
            confusion_matrix[k][2]=confusion_matrix[k][2]+1
    return confusion_matrix

def confusion_mat_creator_NOF(confusion_matrix,model_part,name):
    file_name=''.join([model_part,name,'.csv'])
    os.chdir('/')
    os.chdir(checking_dir)
    f = open(file_name)
    csv_f = csv.reader(f)
    
    stack=[]
    for row in csv_f:
        stack.append(row)
    
    if name=='ONGO':
        k=0
    elif name=='TSG':
        k=1
    elif name=='Fusion':
        k=2
    else:
        print('name should be ONGO or TSG')
        
    for i in range(1,len(stack)):
        if stack[i][3]=="OG":
            confusion_matrix[k][0]=confusion_matrix[k][0]+1
        elif stack[i][3]== "TSG":
            confusion_matrix[k][1]=confusion_matrix[k][1]+1
    return confusion_matrix
##%%
#
#checking_dir= "E:/NN_vin/MSMS_tool_after/all_results/mixed_pdbs/results/ensamble_NOF"
#
##checking_dir='E:/NN_vin/MSMS_tool_after/all_results/mixed_pdbs/results/NOF/k_5'
##checking_dir='E:/NN_vin/MSMS_tool_after/all_results/mixed_pdbs/results/NOF'
##model_part='all_mix_OT_k_7_ac82_'
#model_part='T_1_'
#confusion_matrix = np.zeros((3,2))
#name='ONGO'
#confusion_matrix = confusion_mat_creator_NOF(confusion_matrix,model_part,name)
#name='TSG'
#confusion_matrix = confusion_mat_creator_NOF(confusion_matrix,model_part,name)
##name='Fusion'
#confusion_matrix = confusion_mat_creator_NOF(confusion_matrix,model_part,name)
#%%
#checking_dir='E:/NN_vin/MSMS_tool_after/SITE_must/unigene_results/results'
#model_part='mix_k_3_005_'
#model_part='mix_k_5_005_'
#model_part='mix_k_7_005_'
#model_part='T_1_'
#model_part='all_v2_mix_OT_k_5_ac80_Fusion'
#checking_dir='E:/NN_vin/MSMS_tool_after/all_results/mixed_pdbs/results'
#checking_dir='E:/NN_vin/MSMS_tool_after/all_results/unigene/results'
#model_part='all_uni_k_7_ac80_'
checking_dir= 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean_NN_results'
model_part='wo_mean_S_V2_uni_k_3_ac71_'

confusion_matrix = np.zeros((3,3))
name='ONGO'
confusion_matrix = confusion_mat_creator(confusion_matrix,model_part,name)
name='TSG'
confusion_matrix = confusion_mat_creator(confusion_matrix,model_part,name)
name='Fusion'
confusion_matrix = confusion_mat_creator(confusion_matrix,model_part,name)