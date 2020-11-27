# -*- coding: utf-8 -*-
"""
Created on %(02-Sep-2018) 16.00Pm

@author: %A.Nishanth C00294860
"""
import os
import csv
import pickle
#%% first load the wanted results files from the directory
working_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping"
working_dir_NN_prob = "".join([working_dir_part, "/Tier_1_NN_source"])
"BLOCK 6 out of 7"
#% creating the model for ONGO_TSG
name = "ONGO_TSG"
working_dir_fraction = "".join([working_dir_part, "/Tier_1_pdb_pikles_length_gene/",name])
saving_dir =  "".join([working_dir_part, "/Tier_1_Finalised_probabilities_gene/",name])
#os.mkdir(saving_dir)
#%% then load the probility values of NN
os.chdir('/')
os.chdir(working_dir_NN_prob)

f = open("".join(["T_1_",name,".csv"]))
csv_f = csv.reader(f)

PDB_id = []
probabilities =[]
cond = True
for row in csv_f:
    if cond:
        #to avoid the heading
        cond = False
    else:
        PDB_id.append(row[0])
        #order is OG, TSG, Fusion
        probabilities.append([float(row[1]),float(row[2]),float(row[3])])
        
#%% then read the fraction values for that class
os.chdir('/')
os.chdir(working_dir_fraction)
# first choose the direct hit for the Genes with PDB's and findout the probabilities
weight_initialize_gene_from_PDB_seq_length = os.listdir()
direct_pikles = []
method_1_pikles = []
method_2_pikles = []
method_3_pikles = []
for n in weight_initialize_gene_from_PDB_seq_length: 
    if n[len(name)+1:len(name)+7] == 'direct':
        direct_pikles.append(n)
    elif n[len(name)+1:len(name)+9] == 'method_1':
        method_1_pikles.append(n)
    elif n[len(name)+1:len(name)+9] == 'method_2':
        method_2_pikles.append(n)
    elif n[len(name)+1:len(name)+9] == 'method_3':
        method_3_pikles.append(n)
#%% calculate the probability for the genes only has one PDBs
gene_direct_prob = []
for i in range(0,len(direct_pikles)):
    dir_1 = pickle.load(open(direct_pikles[i], "rb" ))
    for j in range(0,len(PDB_id)):
        if dir_1[1][0] == PDB_id[j]:
            gene_direct_prob.append([dir_1[0], probabilities[j]])
#%% calculate the probability for the genes only has maore than one PDBs

def probability_from_method(method_name, PDB_id, probabilities):
    """
    This function caculate the final probability for give pikle set
    """
    whole = pickle.load(open(method_name,"rb"))
    # go through all available coverages
    # and calculate the overall fraction first for all PDBs
    pdb_temp = []
    sum_length = 0
    for j in range(0,len(whole[1])):
        sum_length = sum_length + whole[1][j]
        for k in range(0,len(whole[3])):
            #intially needed to append the fractions 
            if j==0:
                pdb_temp.append(whole[1][j]*whole[2][j][k])
            else:
                pdb_temp[k] = pdb_temp[k] + whole[1][j]*whole[2][j][k]
     
    for k in range(0,len(whole[3])):            
        pdb_temp[k] = pdb_temp[k]/sum_length
        
    # then gothrough probabilities and caculate the final
    p_OG = 0
    p_TSG = 0
    p_Fusion = 0
    for k in range(0,len(whole[3])): 
        if pdb_temp[k] > 0:
            for j in range(0,len(PDB_id)):
                if whole[3][k] == PDB_id[j]:
                    p_OG = p_OG + pdb_temp[k]*probabilities[j][0]
                    p_TSG = p_TSG + pdb_temp[k]*probabilities[j][1]
                    p_Fusion = p_Fusion + pdb_temp[k]*probabilities[j][2]
                    
    # finalize the probabilities
    prob = [p_OG/(p_OG + p_TSG + p_Fusion), p_TSG/(p_OG + p_TSG + p_Fusion),  p_Fusion/(p_OG + p_TSG + p_Fusion)]
    return  prob, whole[0]

# start with save the finalised details with gene name
fin_gene_prob = []
Ensamble_gene_prob = []
for i in range(0,len(method_1_pikles)):
#i = 6
    prob_1, gene_name_1 = probability_from_method(method_1_pikles[i], PDB_id, probabilities)
    prob_2, gene_name_2 = probability_from_method(method_2_pikles[i], PDB_id, probabilities)
    prob_3, gene_name_3 = probability_from_method(method_3_pikles[i], PDB_id, probabilities)
    prob_ens = [(prob_1[0] + prob_2[0] + prob_3[0])/3, (prob_1[1] + prob_2[1] + prob_3[1])/3, (prob_1[2] + prob_2[2] + prob_3[2])/3]
    if gene_name_1 == gene_name_2 == gene_name_3:
         fin_gene_prob.append([gene_name_1, prob_1, prob_2, prob_3])
         Ensamble_gene_prob.append([gene_name_1, prob_ens])
#     print("Done: ", i)
         
#%%
 """
 If you can place two decimals for float
 
 """
minimum_prob_decide= 0.25         

def decider_prob(ids, minimum_prob_decide,i):         
    if ids[i][0] > minimum_prob_decide and ids[i][1] > minimum_prob_decide and ids[i][2] > minimum_prob_decide:
       f.write( '\t' +', '+ "OG_TSG_Fusion" + '\n') 
    elif ids[i][0] > minimum_prob_decide and ids[i][1] > minimum_prob_decide and ids[i][2] < minimum_prob_decide:
       f.write( '\t' +', '+ "OG_TSG" + '\n') 
    elif ids[i][0] > minimum_prob_decide and ids[i][1] < minimum_prob_decide and ids[i][2] > minimum_prob_decide:
       f.write( '\t' +', '+ "OG_Fusion" + '\n') 
    elif ids[i][0] < minimum_prob_decide and ids[i][1] > minimum_prob_decide and ids[i][2] > minimum_prob_decide:
       f.write( '\t' +', '+ "TSG_Fusion" + '\n') 
    elif ids[i][0] > minimum_prob_decide*2:
       f.write( '\t' +', '+ "OG" + '\n') 
    elif ids[i][1] > minimum_prob_decide*2:
       f.write( '\t' +', '+ "TSG" + '\n')   
    elif ids[i][2] > minimum_prob_decide*2:
       f.write( '\t' +', '+ "Fusion" + '\n')  
    else:
       raise ValueError("You needed to change the minimum_prob_decide because it doesn't satisfy all cases")


         
os.chdir('/')
os.chdir(saving_dir)
f= open(''.join([name,'_results_gene_overlap_all_methods.txt']),"w+")
f.write('\t\t'+'Probabilities of direct pikles'+'\n')
f.write('Uni_gene_ID'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + 'Class' + '\n')

for ids in gene_direct_prob:
    #insert the group details
    f.write(ids[0] + '\t' +',' + str(round(ids[1][0],2))  + '\t' +',' + str(round(ids[1][1],2)) + '\t' +',' + str(round(ids[1][2],2)))
    decider_prob(ids, minimum_prob_decide,1)
f.write('\n')
f.write('\t--//////////-----------------------------------************-------------------------------------///////----------\t'+'\n')
f.write('\n')
f.write('\t\t'+'Probabilities from Ensamble'+'\n')
f.write('Uni_gene_ID'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + 'Class' + '\n')
for ids in Ensamble_gene_prob:
    #insert the group details
    f.write(ids[0] + '\t' +',' + str(round(ids[1][0],2))  + '\t' +',' + str(round(ids[1][1],2)) + '\t' +',' + str(round(ids[1][2],2)))
    decider_prob(ids, minimum_prob_decide,1)
#print("Done")
f.close() 
# for method_1
f= open(''.join([name,'_supplimentery_results_gene_overlap_method_1.txt']),"w+")
f.write('\t\t'+'Probabilities of Method_1 pikles'+'\n')
f.write('Uni_gene_ID'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + 'Class' + '\n')
for ids in fin_gene_prob:
    #insert the group details
    f.write(ids[0] + '\t' +',' + str(round(ids[1][0],2))  + '\t' +',' + str(round(ids[1][1],2)) + '\t' +',' + str(round(ids[1][2],2)))
    decider_prob(ids, minimum_prob_decide,1)
f.close() 

# for method_2
f= open(''.join([name,'_supplimentery_results_gene_overlap_method_2.txt']),"w+")
f.write('\t\t'+'Probabilities of Method_2 pikles'+'\n')
f.write('Uni_gene_ID'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + 'Class' + '\n')
for ids in fin_gene_prob:
    #insert the group details
    f.write(ids[0] + '\t' +',' + str(round(ids[2][0],2))  + '\t' +',' + str(round(ids[2][1],2)) + '\t' +',' + str(round(ids[2][2],2)))
    decider_prob(ids, minimum_prob_decide,2)
f.close() 

# for method_3
f= open(''.join([name,'_supplimentery_results_gene_overlap_method_3.txt']),"w+")
f.write('\t\t'+'Probabilities of Method_3 pikles'+'\n')
f.write('Uni_gene_ID'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + 'Class' + '\n')
for ids in fin_gene_prob:
    #insert the group details
    f.write(ids[0] + '\t' +',' + str(round(ids[3][0],2))  + '\t' +',' + str(round(ids[3][1],2)) + '\t' +',' + str(round(ids[3][2],2)))
    decider_prob(ids, minimum_prob_decide,3)
print("Done")
f.close() 

