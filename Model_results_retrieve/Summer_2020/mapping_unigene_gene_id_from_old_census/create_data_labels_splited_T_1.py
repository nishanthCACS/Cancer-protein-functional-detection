# -*- coding: utf-8 -*-
"""
Created on %10-May-2020 at 3.31Am

@author: %A.Nishanth C00294860
"""
import os
import pickle
from copy import deepcopy

#%%
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/Tier_1_pdb_pikles_length_gene')
ONGO_PDBs = pickle.load(open('ONGO_PDBs.p', "rb" ) )
TSG_PDBs = pickle.load(open("TSG_PDBs.p", "rb" ) )
Fusion_PDBs = pickle.load(open('Fusion_PDBs.p', "rb" ) )


def chk_over_lap(list_1,list_2,Fusion_list):
    overlapped=[]
    for ids in list_1:
        if ids in list_2:
            overlapped.append(ids)
        if ids in Fusion_list:
            overlapped.append(ids)
    return overlapped

overlapped_PDBs_ONGO=chk_over_lap(ONGO_PDBs,TSG_PDBs,Fusion_PDBs)
overlapped_PDBs_TSG=chk_over_lap(TSG_PDBs,ONGO_PDBs,Fusion_PDBs)

pickle.dump(overlapped_PDBs_ONGO, open('overlapped_PDBs_ONGO.p', "wb" ) ) 
pickle.dump(overlapped_PDBs_TSG, open('overlapped_PDBs_TSG.p', "wb" ) ) 
#%%
#working_dir='C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Spring_2020/Tier_1_pdb_pikles_length_gene'
if '5AY8' in Fusion_PDBs:
    print("there")
#%%
'''
Check the 5AY8 placed where in 2018 gene pdb mapping 

From the source 
C:\Users\nishy\Documents\Projects_UL\Continues BIBM\Uniprot_Mapping\Tier_1_pdb_details_length_source
in 2018 vertion its in ONGO and Fusion

It is not in ONGO in 2020
C:\Users\nishy\Documents\Projects_UL\Continues BIBM\Uniprot_Mapping\Spring_2020\pdb_details_length_source_2020_V_91\T_1

'''