# -*- coding: utf-8 -*-
"""
Created on %10-April-2019(08.55p.m)

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
working_dir_NN_prob = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/clean_NN_results"
os.chdir('/')
os.chdir(working_dir_NN_prob)
files_NN = os.listdir() 
name_NN = files_NN[1]
f = open(name_NN)
csv_f = csv.reader(f)
stack=[]
for row in csv_f:
    stack.append(row)
#%% to load the SITE_satisfied gene with PDB details
def pdb_details_fun(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
    thres_sat_gene_list = pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    return pdb_details,thres_sat_gene_list

#%%
def pdb_to_probability(pdb_id,stack):
    for row in stack:
        if row[0]==pdb_id:
            return row
#%%
checking_pdb_pool=[]
problemtric_pdb=[]
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked"
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success')
f= open('pdbs_finalised.csv',"w+")
f.write( 'Tier,\t'+'Uni_gene_ID,\t' +'PDB_id,\t'+ 'OG,'+'\t'+ 'TSG,'+ '\t'+ 'Fusion,\t'+'Class_classified' + '\t, Assigned class for gene census\n')
for names in files:
        name = names[8:-4]
        pdb_details,thres_sat_gene_list = pdb_details_fun(name,loading_dir)
        for i in range(0,len(pdb_details)):
            if len(pdb_details[i])>0:
                for pdb_id in pdb_details[i]:
                    if pdb_id not in checking_pdb_pool:#to check the same pdb belong to more classes
                        row = pdb_to_probability(pdb_id,stack)
                        predicted=decider_prob(row, minimum_prob_decide,i)
                        f.write(name[-1] +',\t'+thres_sat_gene_list[i] +',\t'  +pdb_id+',\t'+ str(round(float(row[1]),2))  +',\t' + str(round(float(row[2]),2))+',\t' + str(round(float(row[3]),2))+',\t'+predicted+',\t' + name[0:-4]+'\n' )
                    else:
                        problemtric_pdb.append([pdb_id,name])
f.close() 

#%%
def method(i):
    os.chdir('/')
    os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success')
    f= open(''.join(['method',str(i),'.csv']),"w+")
    f.write( 'Tier,\t'+'Uni_gene_ID,\t' +'PDB_id,\t'+ 'Method_number,\t'+ 'OG,'+'\t'+ 'TSG,'+ '\t'+ 'Fusion,\t'+'Class_classified' + '\t, Class_census\n')
    for names in files:
        name = names[8:-4]
        gene_direct_prob, Ensamble_gene_prob, fin_gene_prob=pickle_retrieve(name)
        
        for ids in fin_gene_prob:
            #insert the group details
            predicted = decider_prob(f, ids, minimum_prob_decide,i)
            if  ids[0] not in problemetric:
                f.write(name[-1] +',\t'+ uni_gene_to_id_gene_name(fin_selec,ids[0])+',\t' +  ids[0] +',\t'  +str(i)+',\t'+ str(round(ids[1][0],2))  +',\t' + str(round(ids[1][1],2))+',\t' + str(round(ids[1][2],2))+',\t'+predicted+',\t' + name[0:-4]+'\n' )
    f.close() 
    

#%%
def pickle_retrieve(name):
    os.chdir('/')
    os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/clean_Finalised_probabilities_gene')
    gene_direct_prob = pickle.load( open(''.join([name,"_gene_direct_prob.p"]), "rb" ))
    Ensamble_gene_prob = pickle.load( open(''.join([name,"_Ensamble_gene_prob.p"]), "rb" ))
    fin_gene_prob = pickle.load( open(''.join([name,"_fin_gene_prob.p"]), "rb" ))
    return gene_direct_prob, Ensamble_gene_prob, fin_gene_prob

gene_direct_prob, Ensamble_gene_prob, fin_gene_prob=pickle_retrieve(name)

#%%
uni_gene_all=[]
os.chdir('/')
os.chdir(working_dir)
for names in files:
    name = names[8:-4]
    uni_gene_all.append(ens_id_retrieve(name))
uni_gene_all=sum(uni_gene_all,[])
# to load the gene name

gene_names=sum(stack,[])



