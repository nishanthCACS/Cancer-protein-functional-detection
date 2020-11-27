# -*- coding: utf-8 -*-
"""
Created on %10-May-2020 at 3.31Am

@author: %A.Nishanth C00294860

"""
import os
import pickle
'''
To check MSMS needed PDBs left out before weight assignment
'''
##%% first load the PDB in the pickle and findout PDBs went through the model
#SITE_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020'
## to load the needed PDBs
#os.chdir('/')
#os.chdir(SITE_dir)
#SITE_satisfied_all=list(set(pickle.load(open('SITE_MSMS_quater_needed.p', "rb"))))     
#PDB_directory='F:/scrach_cacs_1083/optimaly_tilted_21_quarter/T_1_T_2_rest/before_thresh'
##PDB_directory='F:/scrach_cacs_1083/optimaly_tilted_17_quarter/T_1_T_2_rest/before_thresh'
#pdbs=os.listdir(PDB_directory)
#not_satisfied=[]
#for i in range(0,len(SITE_satisfied_all)):
##    print(i)
#    if ''.join([SITE_satisfied_all[i],'.npy']) not in pdbs:
#        not_satisfied.append(SITE_satisfied_all[i])
#pickle.dump(not_satisfied, open( 'not_satisfied_model.p', "wb" ) ) 
#
#        

#%%

from class_probability_weight_initialize_gene_from_PDB_seq_length_C_alpha_sat import probability_weight_initialize_gene_from_PDB_seq_length_C_alpha_sat

working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/"
#% Gothrough the T_1 and create the results
for Tier in [1,2]:
    for amino_acids in [20,21]:
        names_temp_list=os.listdir(''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/pdb_details_length_source_2020_V_91/T_',str(Tier)]))
        for names_temp in names_temp_list:
            name=names_temp[4:-4]
            print("creating the results of Tier: ",Tier, " of amino_acids: ",amino_acids," name ",name)

            working_dir=''.join([working_dir_part,'Tier_',str(Tier),'_pdb_pikles_length_gene/'])
            saving_dir=''.join([working_dir_part,'weights_assign_methods/Overlapped_allowed_all/T_',str(Tier),'/SITE/before_thresh/',str(amino_acids),'_aminoacid/'])
            os.chdir('/')
            if not os.path.exists(saving_dir):
               os.makedirs(saving_dir)
            obj = probability_weight_initialize_gene_from_PDB_seq_length_C_alpha_sat(working_dir,saving_dir,name,Tier,amino_acids)
            obj.method_whole_finalize()
            del obj
#       #%%
#import os
#import pickle
#name='ONGO'
#Tier=1
#amino_acids=21
#working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/"
#
#working_dir=''.join([working_dir_part,'Tier_',str(Tier),'_pdb_pikles_length_gene/'])
#
#
#train_test_site_sat_dir=''.join(['C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/SITE_must_all_calpha_Train_test_details/',str(amino_acids),'_aminoacid'])
#
##Inorder to use the training and testing data from 2018 census to model
#os.chdir('/')
#os.chdir(train_test_site_sat_dir)
#temp_1_pds = pickle.load(open('train_list_ids.p', "rb" ) )
#temp_2_pds = pickle.load(open('test_list_ids.p', "rb" ) )
#
#
#SITE_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020'
#os.chdir('/')
#os.chdir(SITE_dir)
#SITE_satisfied_temp= pickle.load(open('SITE_MSMS_quater_needed.p', "rb"))  
## since both 21 amino acids and 20 amino acids not-satisfied in have same PDBs
#not_satisfied_model=pickle.load(open('not_satisfied_model.p', "rb"))  #the PDB data unable to convert to supported quarter model formation 
#
#SITE_satisfied_all=list(set(temp_1_pds+temp_2_pds+SITE_satisfied_temp))
# 
## first load the files needed
#os.chdir('/')
#os.chdir(working_dir)
#whole_PDBs = pickle.load(open(''.join([name,"_PDBs.p"]), "rb" ) )
#print(len(whole_PDBs))      
#print(len(SITE_satisfied_all))
#
#site_sat_flatten=[]
#if Tier==1:
#    if name=='ONGO' or name=='TSG' or name=='Fusion':
#        overlapped_PDBs_ONGO = pickle.load(open('overlapped_PDBs_ONGO.p', "rb"))
#        overlapped_PDBs_TSG = pickle.load(open('overlapped_PDBs_TSG.p', "rb"))
#        #to avoid not staified or overlap
#        print("HEer")
#        overlap=list(set(overlapped_PDBs_ONGO+overlapped_PDBs_TSG+not_satisfied_model))
#        print('overlap: ',len(overlap))
#
#        for pdbs in whole_PDBs:
#            if pdbs in SITE_satisfied_all:
#                if pdbs not in overlap:
#                    site_sat_flatten.append(pdbs) 
#                    
                    #%%
