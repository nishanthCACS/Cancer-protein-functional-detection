# -*- coding: utf-8 -*-
"""
Created on %19-Nov-2019(08.42 P.m)

@author: %A.Nishanth C00294860
"""
import pickle
import os
from  class_surface_msms_depth_tilted_tf_pikle import surface_msms_depth_MOTIF_class
#%%
def pdb_details_fun(name):
#    name='ONGO'    
    loading_dir ="C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/MOTIF_results/suceed_unigene"   
  
    os.chdir("/")
    os.chdir(loading_dir)
    test_pdb_ids_details  = pickle.load(open( ''.join([name,"_test_pdb_ids_details.p"]), "rb" ) )
    train_pdb_ids_details =  pickle.load(open( ''.join([name,"_train_pdb_ids_details.p"]), "rb" ))
    test_unigene_details =  pickle.load(open( ''.join([name,"_test_uni_gene.p"]), "rb" )) 
    return train_pdb_ids_details,test_pdb_ids_details,test_unigene_details
    
name = "ONGO"

#    pdb_source_dir_part = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/"
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
saving_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/MOTIF_results/unigene/optimaly_tilted"   
loading_pikle_dir = ''.join([loading_pikle_dir_part,name])

saving_dir_part_train=''.join([saving_dir_part,'/Train/',name])
train_pdb_ids_details_all=sum(train_pdb_ids_details,[])
print(name," training in progress")
#pdb_name= train_pdb_ids_details_all[0]
#problemtric PDB check
pdb_name='5WLB'
if pdb_name!='4MDQ':
    obj = surface_msms_depth_MOTIF_class(loading_pikle_dir, saving_dir_part_train, pdb_name)
    property_surface,MOTIF_prop,sur_res_cor = obj.results()
    del obj
    
    
#%%
import numpy as np
from copy import deepcopy 
MOTIF_prop=np.array(MOTIF_prop,dtype=float)
property_surface=np.array(property_surface)
whole_property_surface=np.column_stack((property_surface,MOTIF_prop))
#%%
coordinate_property=np.zeros((3,200,200,17))
sur_res_cor=np.array(sur_res_cor,dtype=int)
for i in range(0,len(sur_res_cor)):
    coordinate_property[0,sur_res_cor[i,0],sur_res_cor[i,1],:]=deepcopy(whole_property_surface[i,:])
    coordinate_property[1,sur_res_cor[i,1],sur_res_cor[i,2],:]=deepcopy(whole_property_surface[i,:])
    coordinate_property[2,sur_res_cor[i,0],sur_res_cor[i,2],:]=deepcopy(whole_property_surface[i,:])

#%%
import os
import numpy as np
os.chdir('/')
os.chdir('F:/codes/mapping_eigen')
#np.save(pdb_name, coordinate_property)
chk=np.load('1E3G.npy')