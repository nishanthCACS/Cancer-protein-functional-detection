# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""

import os
import pickle


#%%
def create_pikles_test(name):
    os.chdir('/')
    os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_pikles_length_gene/SITE_checked')
    SITE_satisfied = pickle.load( open(''.join([name,'_SITE_satisfied.p']), "rb" ))
    thres_sat_gene_list=pickle.load( open(''.join([name,'_thres_sat_gene_list.p']), "rb" ))
    
    test_pdb_ids_details=[]
    test_uni_gene=[]
    for i in range(0,len(SITE_satisfied)):
        if len(SITE_satisfied[i])>0:
            test_pdb_ids_details.append(SITE_satisfied[i])
            test_uni_gene.append(thres_sat_gene_list[i])
            
    os.chdir('/')
    os.chdir("C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/clean")
    pickle.dump(test_pdb_ids_details, open( ''.join([name,"_test_pdb_ids_details.p"]), "wb" ) )           
    pickle.dump(test_uni_gene, open( ''.join([name,"_test_uni_gene.p"]), "wb" ) )  
    

working_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2019/pdb_details_length_source"
os.chdir('/')
os.chdir(working_dir)
files=os.listdir()
for names in files:
    create_pikles_test(names[8:-4])
    print(names[8:-4]," done :)")
