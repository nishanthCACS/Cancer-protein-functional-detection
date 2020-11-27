import os
import pickle
import numpy as np
def pdb_details_fun(name,loading_dir):
    os.chdir("/")
    os.chdir(loading_dir)
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    return pdb_details

loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

'''
From the results obtained above check the percentage of C_alpha carbon covered as surface
#his will check upto 25% of MOTIF atom loss while selecting surface atoms
'''
checking_thresh_ca=[7.2,6.9,6.7]
checking_thresh_res=[6.9,6.7,6.6]
loading_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"

def sur_per_saver(pdb_name,loading_pikle_dir,saving_dir):
    sat_ca_sur_thresh =np.zeros((3,3))
    
    os.chdir('/')
    os.chdir(loading_pikle_dir)
#    coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
    fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
    fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
    for i in range(0,len(checking_thresh_ca)):
        for j in range(0,len(checking_thresh_res)):
            for k in range(0,len(fin_ca_depth_all)):
                if fin_ca_depth_all[k]<=checking_thresh_ca[i]:
                    if fin_res_depth_all[k]<=checking_thresh_res[j]:
                        sat_ca_sur_thresh[i][j]=sat_ca_sur_thresh[i][j]+1
    for i in range(0,len(checking_thresh_ca)):
        for j in range(0,len(checking_thresh_res)):
            sat_ca_sur_thresh[i][j]=100*sat_ca_sur_thresh[i][j]/len(fin_ca_depth_all)
    
    os.chdir('/')
    os.chdir(saving_dir)
    np.save(''.join(['sat_ca_sur_thresh_',pdb_name,'.npy']), sat_ca_sur_thresh) 
#%%
name = 'Fusion'
loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
saving_dir=''.join([loading_pikle_dir,"/c_a_sur_per"])
loading_MOTIF_dir = ''.join([loading_pikle_dir_part,name,'/MOTIF_stats'])
pdb_details = pdb_details_fun(name,loading_dir)
for i in  range(0,len(pdb_details)):
    pdb_list = pdb_details[i]
    for pdb_name in pdb_list:
        print("PDB in progress: ", pdb_name)
        sur_per_saver(pdb_name,loading_pikle_dir,saving_dir)
