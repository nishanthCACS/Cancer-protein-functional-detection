# -*- coding: utf-8 -*-
"""
Created on %19-April-2019(3.22p.m)

@author: %A.Nishanth C00294860
"""
import pickle
import os
import csv
# make sure the 
#%%
minimum_prob_decide = 0.25 
#% First change the results given by Torch to only the pdb_id formating
working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/clean_Finalised_probabilities_gene/train_test"

files_train=["ONGO_T_1_train","TSG_T_1_train","Fusion_T_1_train"]
#%%
def decider_prob(f,ids, minimum_prob_decide,i):
    """
    Helper function for documentation for decide calss
    """ 
#    print(ids[i],'----')
    if ids[i][0] > minimum_prob_decide and ids[i][1] > minimum_prob_decide and ids[i][2] > minimum_prob_decide:
       return "ONGO_TSG_Fusion"
    elif ids[i][0] > minimum_prob_decide and ids[i][1] > minimum_prob_decide and ids[i][2] < minimum_prob_decide:
       return "ONGO_TSG"
    elif ids[i][0] > minimum_prob_decide and ids[i][1] < minimum_prob_decide and ids[i][2] > minimum_prob_decide:
       return  "ONGO_Fusion"
    elif ids[i][0] < minimum_prob_decide and ids[i][1] > minimum_prob_decide and ids[i][2] > minimum_prob_decide:
       return "TSG_Fusion"
    elif ids[i][0] > minimum_prob_decide*2:
       return "ONGO" 
    elif ids[i][1] > minimum_prob_decide*2:
       return "TSG"
    elif ids[i][2] > minimum_prob_decide*2:
       return "Fusion"
    else:
       raise ValueError("You needed to change the minimum_prob_decide because it doesn't satisfy all cases")

#%%
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success')
f= open('Finalize_all_methods_T1_train.csv',"w+")
f.write( 'Tier,\t'+'Gene_name,\t'+'Uni_gene_ID,\t' + 'Direct or Ensamble,\t'+ 'OG,'+'\t'+ 'TSG,'+ '\t'+ ', Fusion,\t'+'Class_classified' + '\t, Class_census\n')
for names in files_train:
    name = names[0:-10]
    gene_direct_prob, Ensamble_gene_prob, fin_gene_prob=pickle_retrieve(names)
    print(name)
    for ids in gene_direct_prob:
        #insert the group details
        predicted = decider_prob(f, ids, minimum_prob_decide,1)
#        if  ids[0] not in problemetric:
        f.write('1,\t'+ uni_gene_to_id_gene_name(fin_selec,ids[0])+',\t' +  ids[0] +',\t'  +'Direct,\t'+ str(round(ids[1][0],2))  +',\t' + str(round(ids[1][1],2))+',\t' + str(round(ids[1][2],2))+',\t'+predicted+',\t' + name+'\n' )
        
    for ids in Ensamble_gene_prob:
        #insert the group details
        predicted = decider_prob(f, ids, minimum_prob_decide,1)
#        if  ids[0] not in problemetric:
        f.write('1,\t'+ uni_gene_to_id_gene_name(fin_selec,ids[0])+',\t' +  ids[0] +',\t'  +'Ensamble,\t'+ str(round(ids[1][0],2))  +',\t' + str(round(ids[1][1],2))+',\t' + str(round(ids[1][2],2))+',\t'+predicted+',\t' + name+'\n' )
f.close() 
#%%
def method(i):
    os.chdir('/')
    os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success')
    f= open(''.join(['method',str(i),'_T1_train.csv']),"w+")
    f.write( 'Tier,\t'+'Gene_name,\t'+'Uni_gene_ID,\t' + 'Method_number,\t'+ 'OG,'+'\t'+ 'TSG,\t'+ 'Fusion,\t'+'Class_classified\t, Class_census\n')
    for names in files_train:
        name = names[0:-10]
        gene_direct_prob, Ensamble_gene_prob, fin_gene_prob=pickle_retrieve(names)
        
        for ids in fin_gene_prob:
            #insert the group details
            predicted = decider_prob(f, ids, minimum_prob_decide,i)
            if  ids[0] not in problemetric:
                f.write('1,\t'+ uni_gene_to_id_gene_name(fin_selec,ids[0])+',\t' +  ids[0] +',\t'  +str(i)+',\t'+ str(round(ids[1][0],2))  +',\t' + str(round(ids[1][1],2))+',\t' + str(round(ids[1][2],2))+',\t'+predicted+',\t' + name+'\n' )
    f.close() 
    
method(1)
method(2)
method(3)

#%%
def pickle_retrieve(name):
    os.chdir('/')
    os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/clean_Finalised_probabilities_gene/train_test')
    gene_direct_prob = pickle.load( open(''.join([name,"_gene_direct_prob.p"]), "rb" ))
    Ensamble_gene_prob = pickle.load( open(''.join([name,"_Ensamble_gene_prob.p"]), "rb" ))
    fin_gene_prob = pickle.load( open(''.join([name,"_fin_gene_prob.p"]), "rb" ))
    return gene_direct_prob, Ensamble_gene_prob, fin_gene_prob
#name=files_train[0]
#gene_direct_prob, Ensamble_gene_prob, fin_gene_prob=pickle_retrieve(name)

"""
To find the ENs ids to gene name cards
"""
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
    
#%%
uni_gene_all=[]
os.chdir('/')
os.chdir(working_dir)
for name in files_train:
    uni_gene_all.append(ens_id_retrieve(name))
uni_gene_all=sum(uni_gene_all,[])
#%% to load the gene name
import csv
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success')
 
f = open('gene_symbol.txt')
csv_f = csv.reader(f)

stack=[]
for row in csv_f:
    stack.append(row)
gene_names=sum(stack,[])

f = open('uniprot-train_unigene_edited.csv')
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

occur_twice=[]
problemetric=[]
for i in range(0,len(fin_selec)):
    for j in range(i+1,len(fin_selec)):
        if fin_selec[i][1] == fin_selec[j][1]:
            occur_twice.append(j)
            print(fin_selec[i], ' ',fin_selec[j])
            problemetric.append(fin_selec[j][1])

problemetric = list(set(problemetric))
problemtric_all=[]
for ids in problemetric:
    temp=[ids]
    for i in range(0,len(fin_selec)):
        if fin_selec[i][1] == ids:
            temp.append(fin_selec[i][0])
    problemtric_all.append(temp)
#
for ids in uni_gene_all:
    if ids not in unigene_sat:
        print("missed: ",ids)
def uni_gene_to_id_gene_name(fin_selec,chk_uni_gene):
    for ids in fin_selec:
        if ids[1]==chk_uni_gene:
            return ids[0]
#%% after manualy checked then add it the two genes
#for ids in problemetric:
#    for i in range(0,len(fin_selec)):
#        if fin_selec[i][1] == ids:
#            fin_selec[i][0]= ''.join([problemtric_all[0][1],' or ',problemtric_all[0][2]])
          
fin_selec.append(['HIST1H3B','Hs.626666'])   
fin_selec.append(['HIST1H4I','Hs.745457'])         
## from the geneID confiremed
##Hs.626666: HIST1H3B#ONGO
##Hs.745457: HIST1H4I#Fusion