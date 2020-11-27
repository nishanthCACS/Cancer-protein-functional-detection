# -*- coding: utf-8 -*-
"""
Created on %19-April-2019(1.55p.m)

@author: %A.Nishanth C00294860
"""
import pickle
import os
import csv

minimum_prob_decide =0.25
# make sure the 
#%%
#% First change the results given by Torch to only the pdb_id formating
working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/clean_Finalised_probabilities_gene"
naming_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_details_length_source"
os.chdir('/')
os.chdir(naming_dir)
files=os.listdir()
#%%
def decider_prob(ids, minimum_prob_decide,i):
    """
    Helper function for documentation for decide calss
    """ 
    if float(ids[1]) > minimum_prob_decide and float(ids[2]) > minimum_prob_decide and float(ids[3]) > minimum_prob_decide:
       return "ONGO_TSG_Fusion"
    elif float(ids[1]) > minimum_prob_decide and float(ids[2]) > minimum_prob_decide and float(ids[3]) < minimum_prob_decide:
       return "ONGO_TSG"
    elif float(ids[1]) > minimum_prob_decide and float(ids[2]) < minimum_prob_decide and float(ids[3]) > minimum_prob_decide:
       return  "ONGO_Fusion"
    elif float(ids[1]) < minimum_prob_decide and float(ids[2]) > minimum_prob_decide and float(ids[3]) > minimum_prob_decide:
       return "TSG_Fusion"
    elif float(ids[1]) > minimum_prob_decide*2:
       return "ONGO" 
    elif float(ids[2]) > minimum_prob_decide*2:
       return "TSG"
    elif float(ids[3]) > minimum_prob_decide*2:
       return "Fusion"
    else:
       raise ValueError("You needed to change the minimum_prob_decide because it doesn't satisfy all cases")


#%% to load the SITE_satisfied gene with PDB details
def pdb_details_fun_train(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
    thres_sat_gene_list = pickle.load(open( ''.join([name,"_train_unigene_details.p"]), "rb" ))
    pdb_details =pickle.load(open( ''.join([name,"_train_pdb_ids_details.p"]), "rb" ))
    return pdb_details,thres_sat_gene_list

def pdb_details_fun_test(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
    thres_sat_gene_list = pickle.load(open( ''.join([name,"_test_uni_gene.p"]), "rb" ))
    pdb_details =pickle.load(open( ''.join([name,"_test_pdb_ids_details.p"]), "rb" ))
    return pdb_details,thres_sat_gene_list

def pdb_to_probability(pdb_id,stack):
    for row in stack:
        if row[0]==pdb_id:
            return row


#%% to sort the trian pdbs results
checking_pdb_pool=[]
problemtric_pdb=[]
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/k_3_86/train_results/"
working_dir_NN_prob_train = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/k_3_86/train_results/clean_NN_results"

os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success')
f= open('T1_train_pdbs_finalised.csv',"w+")
f.write( 'Tier,\t'+'Uni_gene_ID,\t' +'PDB_id,\t'+ 'OG,'+'\t'+ 'TSG,'+ '\t'+ 'Fusion,\t'+'Class_classified' + '\t, Assigned class for gene census\n')


os.chdir('/')
os.chdir(working_dir_NN_prob_train)
files_NN = os.listdir() 
for name_NN in files_NN:
    os.chdir('/')
    os.chdir(working_dir_NN_prob_train)
    if 'T_1_' in name_NN:
        print(name_NN[4:-4])
        name=name_NN[4:-4]

        f_2 = open(name_NN)
        csv_f = csv.reader(f_2)
        stack=[]
        for row in csv_f:
            stack.append(row)
            
        pdb_details,thres_sat_gene_list = pdb_details_fun_train(name,loading_dir)
        for i in range(0,len(pdb_details)):
            if len(pdb_details[i])>0:
                for pdb_id in pdb_details[i]:
                    if pdb_id !='4MDQ':
                        if pdb_id not in checking_pdb_pool:#to check the same pdb belong to more classes
                            row = pdb_to_probability(pdb_id,stack)
                            predicted=decider_prob(row, minimum_prob_decide,i)
                            f.write('1,\t'+thres_sat_gene_list[i] +',\t'  +pdb_id+',\t'+ str(round(float(row[1]),2))  +',\t' + str(round(float(row[2]),2))+',\t' + str(round(float(row[3]),2))+',\t'+predicted+',\t' + name+'\n' )
                        else:
                            problemtric_pdb.append([pdb_id,name])
f.close() 


#%% to do the test
checking_pdb_pool=[]
problemtric_pdb=[]
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/k_3_86/train_results/"
working_dir_NN_prob_test = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/k_3_86/test_results/clean_NN_results"

os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success')
f= open('T1_test_pdbs_finalised.csv',"w+")
f.write( 'Tier,\t'+'Uni_gene_ID,\t' +'PDB_id,\t'+ 'OG,'+'\t'+ 'TSG,'+ '\t'+ 'Fusion,\t'+'Class_classified' + '\t, Assigned class for gene census\n')


os.chdir('/')
os.chdir(working_dir_NN_prob_test)
files_NN = os.listdir() 
for name_NN in files_NN:
    os.chdir('/')
    os.chdir(working_dir_NN_prob_test)
    if 'T_1_' in name_NN:
        print(name_NN[4:-4])
        name=name_NN[4:-4]

        f_2 = open(name_NN)
        csv_f = csv.reader(f_2)
        stack=[]
        for row in csv_f:
            stack.append(row)
            
        pdb_details,thres_sat_gene_list = pdb_details_fun_test(name,loading_dir)
        for i in range(0,len(pdb_details)):
            if len(pdb_details[i])>0:
                for pdb_id in pdb_details[i]:
                    if pdb_id !='721P':
                        if pdb_id not in checking_pdb_pool:#to check the same pdb belong to more classes
                            row = pdb_to_probability(pdb_id,stack)
                            predicted=decider_prob(row, minimum_prob_decide,i)
                            f.write('1,\t'+thres_sat_gene_list[i] +',\t'  +pdb_id+',\t'+ str(round(float(row[1]),2))  +',\t' + str(round(float(row[2]),2))+',\t' + str(round(float(row[3]),2))+',\t'+predicted+',\t' + name+'\n' )
                        else:
                            problemtric_pdb.append([pdb_id,name])
f.close() 
#%% sorting the problemetric PDBs
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/k_3_86/train_results/"
pdb_details,thres_sat_gene_list = pdb_details_fun_train('ONGO',loading_dir)
for i in range(0,len(pdb_details)):
    if len(pdb_details[i])>0:
        if '4MDQ' in pdb_details[i]:
            print('problemetric Uni_gene: ',thres_sat_gene_list[i])
            print(i)