# -*- coding: utf-8 -*-
"""
Created on %21-May-2020 at 4.41pm

@author: %A.Nishanth C00294860
"""

import os
import csv
import pickle
#%% go through the pickles and select the UniGene details
def ens_id_retrieve(name):
    """
    from the pickle files retrieve the UniGened details
    """
    gene_direct_prob = pickle.load( open(''.join([name,"_gene_direct_prob.p"]), "rb" ))
    Ensamble_gene_prob = pickle.load( open(''.join([name,"_Ensamble_gene_prob.p"]), "rb" ))
    uni_gene=[]
    for id_s in gene_direct_prob:
        uni_gene.append(id_s[0])
    for id_s in Ensamble_gene_prob:
        uni_gene.append(id_s[0])
    return uni_gene

def fin_selected_unigene_G_ID(working_dir,train=True):   
    #% First change the results given by Torch to only the pdb_id formating
    if train:
        files_train=["ONGO_T_1_train","TSG_T_1_train","Fusion_T_1_train"]
    else:
        files_train= ["ONGO_T_1_test","TSG_T_1_test","Fusion_T_1_test"]

    uni_gene_all=[]
    os.chdir('/')
    os.chdir(working_dir)
    for name in files_train:
        uni_gene_all.append(ens_id_retrieve(name))
    uni_gene_all=sum(uni_gene_all,[])
    
    
    os.chdir('/')
    os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success')
     
    f = open('gene_symbol.txt')
    csv_f = csv.reader(f)
    
    stack=[]
    for row in csv_f:
        stack.append(row)
    gene_names=sum(stack,[])
    
    if train:
        f = open('uniprot-train_unigene_edited.csv')
    else:
        f = open('uniprot-test_unigene_edited.csv')

    csv_f = csv.reader(f)
    stack=[]
    for row in csv_f:
        stack.append(row)
    
    selected=[]
    for row in stack:
        if not len(row[0])==0:
            selected.append(row)
    
    fin_selec=[]
    unigene_sat=[]
    if train:
        for row in selected:
            for i in range(1,7):
                if row[i] in uni_gene_all:
                    if len(fin_selec)>0:
                        if row[0].split()[0] != fin_selec[-1][0]:
                            if row[0].split()[0] in gene_names:
                                fin_selec.append([row[0].split()[0],row[i]])
                                unigene_sat.append(row[i])
                    else:
                        if row[0].split()[0] in gene_names:
                            fin_selec.append([row[0].split()[0],row[i]])
                            unigene_sat.append(row[i])
        fin_selec.append(['HIST1H3B','Hs.626666'])   
        fin_selec.append(['HIST1H4I','Hs.745457'])   
    else:
        for row in selected:
            for i in range(1,3):
                if row[i] in uni_gene_all:
                    if len(fin_selec)>0:
                        if row[0].split()[0] != fin_selec[-1][0]:
                            if row[0].split()[0] in gene_names:
                                fin_selec.append([row[0].split()[0],row[i]])
                                unigene_sat.append(row[i])
                    else:
                        if row[0].split()[0] in gene_names:
                            fin_selec.append([row[0].split()[0],row[i]])
                            unigene_sat.append(row[i])
        fin_selec.append(['EGFR','Hs.731652'])   
        fin_selec.append(['RABEP1','Hs.732542'])                           
    return fin_selec

working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/clean_Finalised_probabilities_gene/train_test"
fin_selec=fin_selected_unigene_G_ID(working_dir)
fin_selec_test=fin_selected_unigene_G_ID(working_dir,train=False)

#%%
def map_unigene_sym(unigenes,fin_selec,test_og=False):
    selected_sym=[]
    prob_sym=[]
    for ug in unigenes:
        selec_not_sat=True
        for i in range(0,len(fin_selec)):
            if fin_selec[i][1]==ug:
                if test_og:
                    if fin_selec[i][0]=='KRAS':
                        print('LEFT KRAS: ',ug)
                    else:
                        selected_sym.append(fin_selec[i][0])
                else:
                    selected_sym.append(fin_selec[i][0])
                selec_not_sat=False
        if selec_not_sat:
            prob_sym.append(ug)
            print(ug)
    return selected_sym,prob_sym

def load_unigene_train_test(name,fin_selec,fin_selec_test):    
                
    unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Tier_1/clean/21_amino_acids_fully_clean_SITE'
    os.chdir('/')
    os.chdir(unigene_load_dir)
    train_gene_list = pickle.load(open(''.join([name,"_train_genes_21.p"]), "rb")) 
    test_gene_list = pickle.load(open(''.join([name,"_test_genes_21.p"]), "rb")) 
    
    train_new_gene_list,prob_sym_train=map_unigene_sym(train_gene_list,fin_selec)
    if name=="ONGO":
        test_new_gene_list,prob_sym_test=map_unigene_sym(test_gene_list,fin_selec_test,test_og=True)
    else:
        test_new_gene_list,prob_sym_test=map_unigene_sym(test_gene_list,fin_selec_test)

    #    return train_new_gene_list,test_new_gene_list,prob_sym_train,prob_sym_test
    train_new_gene_list= list(set(train_new_gene_list))
    test_new_gene_list=list(set(test_new_gene_list))
    if len(train_new_gene_list)==len(train_gene_list):
          if  len(test_new_gene_list)==len(test_gene_list):
              return train_new_gene_list,test_new_gene_list
          else:
              raise('Prob in test')
    else:
        raise('Prob in train')
#ONGO_train_unigene,ONGO_test_unigene,ONGO_prob_sym_train,ONGO_prob_sym_test= load_unigene_train_test('ONGO',fin_selec,fin_selec_test)
#TSG_train_unigene,TSG_test_unigene,TSG_prob_sym_train,TSG_prob_sym_test= load_unigene_train_test('TSG',fin_selec,fin_selec_test)
#Fusion_train_unigene,Fusion_test_unigene,Fusion_prob_sym_train,Fusion_prob_sym_test= load_unigene_train_test('Fusion',fin_selec,fin_selec_test)
ONGO_train_unigene,ONGO_test_unigene= load_unigene_train_test('ONGO',fin_selec,fin_selec_test)
TSG_train_unigene,TSG_test_unigene= load_unigene_train_test('TSG',fin_selec,fin_selec_test)
Fusion_train_unigene,Fusion_test_unigene= load_unigene_train_test('Fusion',fin_selec,fin_selec_test)

#%%

def symbol_gene_id(unigenes):
    working_dir='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Documentation/census_updates/2018_Vs_2020'
    os.chdir('/')
    os.chdir(working_dir)
    f = open('symbol_to_gene_ID.csv')
    csv_f = csv.reader(f)
    
    avoid_first=True
    stack=[]
    for row in csv_f:
        if avoid_first:
            avoid_first=False
        else:
            stack.append(row)
    
    selected_gene_id=[]
    prob_sym=[]
    for gs in unigenes:  
        selec_not_sat=True
        for i in range(0,len(stack)):
            if gs==stack[i][0]:
#                if gs=='KRAS':
#                    print('KRAS: ',stack[i][1])
                selected_gene_id.append(stack[i][1])
                selec_not_sat=False
        if selec_not_sat:
            prob_sym.append(gs)
            print(gs)
    
    return selected_gene_id,prob_sym

ONGO_train_gene_id,ONGO_prob_sym_train=symbol_gene_id(ONGO_train_unigene)
TSG_train_gene_id,TSG_prob_sym_train=symbol_gene_id(TSG_train_unigene)
Fusion_train_gene_id,Fusion_prob_sym_train=symbol_gene_id(Fusion_train_unigene)

ONGO_test_gene_id,ONGO_prob_sym_test=symbol_gene_id(ONGO_test_unigene)
TSG_test_gene_id,TSG_prob_sym_test=symbol_gene_id(TSG_test_unigene)
Fusion_test_gene_id,Fusion_prob_sym_test=symbol_gene_id(Fusion_test_unigene)
#%% 
'''
Go through the newly formed dtaset and include the newly created genes in the train test list
'''
def updte_test_ids(name,train_gene_id,test_gene_id):
    unigene_load_dir =     'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/Tier_1_pdb_pikles_length_gene'
    os.chdir('/')
    os.chdir(unigene_load_dir)
    '''
    Sine this dataset is only formed for SITE staisfied PDBs
    '''
    gene_list_all_temp= pickle.load(open(''.join([name,"_Gene_ID.p"]), "rb"))
    SITE_satisfied= pickle.load(open(''.join([name,"_SITE_satisfied.p"]), "rb"))
    gene_list_all=[]
    for i in range(0,len(SITE_satisfied)):
        if len(SITE_satisfied[i])>0:
            gene_list_all.append(gene_list_all_temp[i])
    train_test_there=train_gene_id+test_gene_id
    missed_ids=[]
    if name=="ONGO" or name=="TSG":
        unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Spring_2020_V_91_train_test_T_1/SITE_MUST_clean'
        os.chdir('/')
        os.chdir(unigene_load_dir)
        overlapped_genes= pickle.load(open(''.join(['problemteric_',name,'_genes.p']), "rb"))
        for gene_id in gene_list_all:
            if (gene_id not in train_test_there) and (gene_id not in overlapped_genes):
                missed_ids.append(gene_id)
    else:
        for gene_id in gene_list_all:
            if gene_id not in train_test_there:
                missed_ids.append(gene_id)   
    for i in range(0,len(missed_ids)):
        if i<len(missed_ids)*0.5:       
            train_gene_id.append(missed_ids[i])
        else:
            test_gene_id.append(missed_ids[i])
    return  train_gene_id,test_gene_id

ONGO_train_gene_id,ONGO_test_gene_id=updte_test_ids("ONGO",ONGO_train_gene_id,ONGO_test_gene_id)
TSG_train_gene_id,TSG_test_gene_id=updte_test_ids("TSG",TSG_train_gene_id,TSG_test_gene_id)
Fusion_train_gene_id,Fusion_test_gene_id=updte_test_ids("Fusion",Fusion_train_gene_id,Fusion_test_gene_id)

#%%
unigene_load_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_for_model_train/Spring_2020_V_91_train_test_T_1/SITE_MUST_clean'
os.chdir('/')
os.chdir(unigene_load_dir)
pickle.dump(ONGO_train_gene_id, open("ONGO_train_gene_id.p", "wb"))  
pickle.dump(ONGO_test_gene_id, open("ONGO_test_gene_id.p", "wb"))  
pickle.dump(TSG_train_gene_id, open("TSG_train_gene_id.p", "wb"))  
pickle.dump(TSG_test_gene_id, open("TSG_test_gene_id.p", "wb"))  
pickle.dump(Fusion_train_gene_id, open("Fusion_train_gene_id.p", "wb"))  
pickle.dump(Fusion_test_gene_id, open("Fusion_test_gene_id.p", "wb"))  