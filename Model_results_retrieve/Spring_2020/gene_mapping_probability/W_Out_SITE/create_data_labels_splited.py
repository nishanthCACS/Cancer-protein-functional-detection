#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:17:22 2019

@author: c00294860
"""
import os
from copy import deepcopy
import pickle
'''This python script is created to make balanced dataset'''
#%%
#main_dir = '/scratch/optimaly_tilted_21'
#main_dir = '/media/c00294860/Nishanth/scrach_cacs_1083/21_aminoacids'
#main_dir = '/media/c00294860/Nishanth/scrach_cacs_1083/neg_14_21_aminoacids'
#main_dir = '/media/c00294860/Nishanth/scrach_cacs_1083/w_OUT_Tilt/21_aminoacids_NO_TILT_thrsh_new_SITE_MUST'
#main_dir = '/media/c00294860/Nishanth/scrach_cacs_1083//w_OUT_Tilt/SITE_MUST_17'
#main_dir = '/media/c00294860/Nishanth/scrach_cacs_1083/Before_thresh_changed/optimaly_tilted_21_quarter'
main_dir = 'F:/ULL_Project/optimally_tilted/quater_models/W_OUT_SITE/all_c_alpha/20_amino_acid'
#main_dir = 'F:/ULL_Project/optimally_tilted/quater_models/W_OUT_SITE/before_thresh/21_amino_acid'
#main_dir = 'F:/ULL_Project/optimally_tilted/quater_models/W_OUT_SITE/New_thresh/20_amino_acid'
#main_dir = 'F:/ULL_Project/optimally_tilted/quater_models/all_c_alpha_SITE/21_amino_acid'
main_dir=''.join([main_dir,'/K_fold'])
os.chdir('/')
os.chdir(main_dir)
train_list_IDs = pickle.load(open("train_list_ids.p", "rb"))
train_labels = pickle.load(open("train_labels.p", "rb"))
train_labels_dic = pickle.load(open("train_labels_dic.p", "rb"))  

#%%     check howmany ONGO and TSG overlap with Fusion
ongo_chk=0
tsg_chk=0
overlapped_PDBs_ONGO=[]
overlapped_PDBs_TSG=[]
overlapped_PDBs=[]
for i in range(0,len(train_labels)):
    if train_labels[i]!=train_labels_dic[train_list_IDs[i]]:
        overlapped_PDBs.append(train_list_IDs[i])
        if train_labels[i]==0:
            overlapped_PDBs_ONGO.append(train_list_IDs[i])
            ongo_chk=ongo_chk+1
        elif train_labels[i]==1:
            overlapped_PDBs_TSG.append(train_list_IDs[i])
            tsg_chk=tsg_chk+1 
print('ONGO overlapped Fusion: ',ongo_chk)
print('TSG overlapped Fusion: ',tsg_chk)
# create seperate dictinary that only have no overlapping PDBs
clean_train_labels_dic={}
for i in range(0,len(train_labels)):
    if train_list_IDs[i] not in overlapped_PDBs:
        clean_train_labels_dic.update({train_list_IDs[i]: train_labels[i]})
#%%
''' oneway is only using clean no overlapping with Fusion at all'''
OG_train_PDBs=[]
TSG_train_PDBs=[]
Fusion_train_PDBs=[]
for i in range(0,len(train_labels)):
    if (train_list_IDs[i] not in overlapped_PDBs) and train_labels[i]==0:
        OG_train_PDBs.append(train_list_IDs[i])
    elif  (train_list_IDs[i] not in overlapped_PDBs) and train_labels[i]==1:
        TSG_train_PDBs.append(train_list_IDs[i])
    elif  (train_list_IDs[i] not in overlapped_PDBs) and train_labels[i]==2:
        Fusion_train_PDBs.append(train_list_IDs[i])
#%% First split the PDB ids into groups  
'''Here the overlap only contains the PDBs with Fusion'''
pickle.dump(OG_train_PDBs, open("OG_train_PDBs.p", "wb"))  
pickle.dump(TSG_train_PDBs, open("TSG_train_PDBs.p", "wb"))      
pickle.dump(Fusion_train_PDBs, open("Fusion_train_PDBs.p", "wb"))  
pickle.dump(overlapped_PDBs_ONGO, open("overlapped_PDBs_ONGO.p", "wb"))      
pickle.dump(overlapped_PDBs_TSG, open("overlapped_PDBs_TSG.p", "wb"))      
#
##%%
#'''This part is done for data-generation check still we give higher prority to Fusion
#'''
## shuflle them seperately
#import numpy as np      
##def on_epoch_end(self):
##  'Updates indexes after each epoch'
###  self.indexes = np.arange(len(self.list_IDs))
##  index_ongo= np.arange(len(OG_train_PDBs))
##  if self.shuffle == True:
##      np.random.shuffle(self.indexes)
#
#
##%%
#
#def index_stack_chk(index_ongo_chk,index_tsg_chk,index_fusion_chk):
#    '''
#    This function check the stack if any of them is empty then creat a new stack
#    This way always change the train data when it cycles
#    '''
#    shuffle =True
#    if index_ongo_chk:
#        print("ONGO idexes reset")
#        index_ongo= np.arange(len(OG_train_PDBs))
#        if shuffle:
#            np.random.shuffle(index_ongo)
#        return list(index_ongo)
#    if index_tsg_chk:
#        print("TSG idexes reset")
#        index_tsg = np.arange(len(TSG_train_PDBs))
#        if  shuffle:
#            np.random.shuffle(index_tsg)
#        return list(index_tsg)
#    if index_fusion_chk:
#        print("Fusion idexes reset")
#        index_fusion = np.arange(len(Fusion_train_PDBs))
#        if  shuffle:
#            np.random.shuffle(index_fusion)
#        return list(index_fusion)
#
#def adding(Fusion_chk,OG_train_PDBs,TSG_train_PDBs,Fusion_train_PDBs,index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes):
#    '''
#    This function add the elements of PDBs from the stack
#    Fusion_chk: True;means add the Fusion PDB in the list
#    '''
#    # adding the ONGO elements
#    if len(index_ongo)>0: 
#        clean_list_of_ids.append(OG_train_PDBs[index_ongo.pop()])
#        clean_list_of_id_classes.append(0)
#    else:
#        print("ONGO idexes reset")
#        index_ongo = index_stack_chk(index_ongo_chk=True,index_tsg_chk=False,index_fusion_chk=False)
#        clean_list_of_ids.append(OG_train_PDBs[index_ongo.pop()])
#        clean_list_of_id_classes.append(0)
#    
#    if Fusion_chk:
#        # adding the Fusion elements
##        print("Fusion element added")
#        if len(index_fusion)>0:
#            clean_list_of_ids.append(Fusion_train_PDBs[index_fusion.pop()])
#            clean_list_of_id_classes.append(2)
#        else:
#            print("Fusion idexes reset")
#            index_fusion = index_stack_chk(index_ongo_chk=False,index_tsg_chk=False,index_fusion_chk=True)
#            clean_list_of_ids.append(Fusion_train_PDBs[index_fusion.pop()])
#            clean_list_of_id_classes.append(2)
#    # adding the TSG elements
#    if len(index_tsg)>0:
#        clean_list_of_ids.append(TSG_train_PDBs[index_tsg.pop()])
#        clean_list_of_id_classes.append(1)
#    else:
#        print("TSG idexes reset")
#        index_tsg = index_stack_chk(index_ongo_chk=False,index_tsg_chk=True,index_fusion_chk=False)
#        clean_list_of_ids.append(TSG_train_PDBs[index_tsg.pop()])
#        clean_list_of_id_classes.append(1)
#        
#    return index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes
##to creat the indexes easy
#index_ongo=[]
#index_tsg=[]
#index_fusion=[]
#'''use indexes as stack'''
##%%
#'''reuse the class '''
#clean_list_of_id_classes=[]
#clean_list_of_ids=[]
#iter_train=199
## since the Fusion indexes runing out early 
#''' reuse the Fusion class again or combined class'''
## since the TSG has higher number of labels where iter means the number of irteration each training
## and since the batch size is 32 and data taken by model is 16 thus the iter is factored by 2
#for i in range(0,(iter_train)): 
#    #choose 6 from ONGO 6 from TSG and 4 From Fusion to make a batch that distribute the all most all PDBs
#    #Fusion_chk=True
#    index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes=adding(True,OG_train_PDBs,TSG_train_PDBs,Fusion_train_PDBs,index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes)
#    index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes=adding(True,OG_train_PDBs,TSG_train_PDBs,Fusion_train_PDBs,index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes)
#    index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes=adding(False,OG_train_PDBs,TSG_train_PDBs,Fusion_train_PDBs,index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes)
#    index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes=adding(True,OG_train_PDBs,TSG_train_PDBs,Fusion_train_PDBs,index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes)
#    index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes=adding(True,OG_train_PDBs,TSG_train_PDBs,Fusion_train_PDBs,index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes)
#    index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes=adding(False,OG_train_PDBs,TSG_train_PDBs,Fusion_train_PDBs,index_ongo,index_tsg,index_fusion,clean_list_of_ids,clean_list_of_id_classes)
#pickle.dump(clean_list_of_ids, open("clean_list_of_ids.p", "wb"))  
#pickle.dump(clean_list_of_id_classes, open("clean_list_of_id_classes.p", "wb"))    
##%%
##if len(index_fusion)//4==0:
##    print("Fusion is factor of 4")
##    print("IDs in clean Fusion class : 352, and got ", len(Fusion_train_PDBs))
##else:
##     raise Exception('Fusion PDBs list is not factor of 4 check it')
#     
##k_og_st=6*((i-(((i*6)//len(index_ongo))*(len(index_ongo)//6))))-((len(index_ongo)%6)*((i*6)//len(index_ongo)))
##for j in index_ongo[i*6:((i+1)*6)]:  
##    OG_temp.append(OG_train_PDBs[j])
##for j in index_tsg[i*6:((i+1)*6)]:  
##    TSG_temp.append(TSG_train_PDBs[j])
###to check thenumber of PDBs in thegroup finished or not    
##
##kf=i-(((i*4)//len(index_fusion))*(len(index_fusion)//4))# to reuse the data since the Fusion length is factor of 4 no issues
##for j in index_fusion[kf*4:((kf*4)+4)]:  
##    Fusion_temp.append(Fusion_train_PDBs[j])
##    #%%
##i=99
##k_og_st=6*((i-(((i*6)//len(index_ongo))*(len(index_ongo)//6))))-((len(index_ongo)%6)*((i*6)//len(index_ongo)))
##print(k_og_st)
##if k_og_st>(len(index_ongo)-6):
##    k_og_st=k_og_st-len(index_ongo)
# 
#
#      #%%
## first choose the PDBs in the group of ONGO and TSG and Fusion sperately
#batch_size=32
##y = np.empty((self.batch_size), dtype=int)
##create indexes like 8 ONGO and 8 TSG and 8 Fusion
##%% fi
##%%
''''using not only clean means create like 0.5,0.5,0 like numpy array categorical for output for'''
 