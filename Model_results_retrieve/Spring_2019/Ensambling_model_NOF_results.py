# -*- coding: utf-8 -*-
"""
Created on %(05-Oct-2018) 16.38Pm

@author: %A.Nishanth C00294860

This method ensamble the results of Deeplearning models(PDBs and finalised the results)
"""
import os
import csv
import numpy as np

#%%
def retrieve_csv_names(files,Type):
    """
    This function get all the results files corresponds to the Type
    Type: ONGO, or TSG or Fusion
    """
    type_files = []
    for i in range(0,len(files)):
        if files[i].endswith(''.join([Type,'.csv'])):# choose the files are in .csv format
            type_files.append(files[i])
    return type_files

def retrieve_probabilty(type_files,i):
    """
    This function retrieve the results from the .csv file and 
    return the probabilities
    """   
    name = type_files[i]        
    stack=[]
    chk = False
    f = open(name)
    csv_f = csv.reader(f)
    if i == 0: 
        pdb_ids = []
        for row in csv_f:
            if chk == True:# to avoid the title
                stack.append(row)
                pdb_ids.append(row[0][-8:-4])
            chk =True
            
        probabilities = np.zeros((len(stack),2))
#        probabilities = np.zeros((len(stack),3))
        s = 0
        for row in stack:
            probabilities[s][0]=float(row[1])
            probabilities[s][1]=float(row[2])
#            probabilities[s][2]=float(row[3])
            s=s+1
        return pdb_ids, probabilities
    else:
        for row in csv_f:
            if chk == True:# to avoid the title
                stack.append(row)
            chk =True
            
#        probabilities = np.zeros((len(stack),3))
        probabilities = np.zeros((len(stack),2))
        s = 0
        for row in stack:
            probabilities[s][0]=float(row[1])
            probabilities[s][1]=float(row[2])
#            probabilities[s][2]=float(row[3])
            s=s+1
        return probabilities

def finalised_prob_type(files,Type,threshold_prob, main_factor):
    """
    This function go through all probability(.csv) files and calculate the finalised probability
    
    files           : list of files in the directory 
    Type            : ONGO, or TSG or Fusion
    threshold_prob  : threhold probability take the class
    main_factor     : The factor*threshold_prob used to take as main class
    """
    type_files = retrieve_csv_names(files,Type)
    pdb_ids, probabilities = retrieve_probabilty(type_files,0)
    for i in range(1,len(type_files)):
        probabilities_t = retrieve_probabilty(type_files,i)
        probabilities = probabilities + probabilities_t
    probabilities = probabilities/len(type_files)
    final_class = []
    for i in range(0,len(probabilities)):
        '''To place the highest probability directly'''
        if probabilities[i][0] > probabilities[i][1]:
           final_class.append("OG")
        elif probabilities[i][1] > probabilities[i][0]:
           final_class.append("TSG")    
#        elif probabilities[i][2] > probabilities[i][1] and probabilities[i][2] > probabilities[i][0]:
#           final_class.append("Fusion")  
#        if probabilities[i][0] >= main_factor*threshold_prob:
#           final_class.append("ONGO")
#        elif probabilities[i][1] >= main_factor*threshold_prob:
#           final_class.append("TSG")    
#        elif probabilities[i][2] >= main_factor*threshold_prob:
#           final_class.append("Fusion")    
#        elif probabilities[i][0] >= threshold_prob and probabilities[i][1] >= threshold_prob and  probabilities[i][1] >= threshold_prob:
#           final_class.append("ONGO_TSG_Fusion") 
#        elif probabilities[i][0] >= threshold_prob and probabilities[i][1] >= threshold_prob:
#           final_class.append("ONGO_TSG") 
#        elif probabilities[i][0] >= threshold_prob and probabilities[i][2] >= threshold_prob:
#           final_class.append("ONGO_Fusion") 
#        elif probabilities[i][1] >= threshold_prob and probabilities[i][2] >= threshold_prob:
#           final_class.append("TSG_Fusion") 
#        else:
#           raise Exception('change the main_factor or threshold_prob!') 
    accuracy = 0       
    for t in final_class:
        if t == Type:
            accuracy = accuracy+1
    accuracy = 100*accuracy/len(final_class)
    print(Type, " accuracy: ", round(accuracy,2),"%")
    return pdb_ids, probabilities, accuracy,final_class
#%% then retrieve the 
Tier=1
working_dir_part = "E:/NN_vin/MSMS_tool_after/all_results/mixed_pdbs/results/ensamble_NOF"

os.chdir("/")
os.chdir(working_dir_part)
files = os.listdir()
threshold_prob = 0.3
main_factor = 1.4
ONGO_pdb_ids, ONGO_probabilities, ONGO_accuracy, ONGO_final_class = finalised_prob_type(files,"ONGO",threshold_prob, main_factor)
TSG_pdb_ids, TSG_probabilities, TSG_accuracy, TSG_final_class = finalised_prob_type(files,"TSG",threshold_prob, main_factor)
#Fusion_pdb_ids, Fusion_probabilities, Fusion_accuracy, Fusion_final_class = finalised_prob_type(files,"Fusion",threshold_prob, main_factor)
#%% then build a ensambled .csv file
Tier = 1

def build_ensamble_csv(Tier,type_g,pdb_ids,probabilities,final_class):
    """
    build csv file
    """
    f= open(''.join(["T_",str(Tier),"_",type_g,'.csv']),"w+")
    f.write("Pro,OG,	TSG, Fusion,	Label" +'\n') 
    for i in range(0,len(pdb_ids)):
        f.write(pdb_ids[i]+',')
        for j in range(0,len(probabilities[i])):
            f.write(str(probabilities[i][j])+',')
        f.write(final_class[i]+'\n')  
    f.close() 
    
build_ensamble_csv(Tier,"ONGO",ONGO_pdb_ids,ONGO_probabilities,ONGO_final_class)    
build_ensamble_csv(Tier,"TSG",TSG_pdb_ids,TSG_probabilities,TSG_final_class)    
build_ensamble_csv(Tier,"Fusion",Fusion_pdb_ids,Fusion_probabilities,Fusion_final_class)    
