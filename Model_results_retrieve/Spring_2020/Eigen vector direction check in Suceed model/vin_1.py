# -*- coding: utf-8 -*-
"""
Created on Sat Jan 04 16:18 2020

@author: c00294860
"""
import numpy as np
from copy import deepcopy
import pickle
import os



def get_C_alpha_optimal_direc(coordinates):
    """
    then change the C-alpha coordinates to the optimal direction
    Inorder to do that calculate the eigen vector and 
    use the eigen vector to rotate the coordinates
    
    returns 
    coordinates             : cetralised coordinates
    finalised_cartesian     : Get the optimal direction rotated coordinates
    """
    cen_coordinates = deepcopy(coordinates)
    
    g_x = 0
    g_y = 0
    g_z = 0
    
    x =[]
    y=[]
    z=[]
    for i in range(0,len(coordinates)):
        #find the center of gravity
         g_x = g_x + coordinates[i][0]       
         g_y = g_y + coordinates[i][1]
         g_z = g_z + coordinates[i][2]    
         
         x.append(coordinates[i][0])
         y.append(coordinates[i][1])
         z.append(coordinates[i][2])
    
    #% then centralize the coordinates
    for i in range(0,len(coordinates)):
        #find the center of gravity
         cen_coordinates[i][0]  = coordinates[i][0] - g_x/len(coordinates)    
         cen_coordinates[i][1]  = coordinates[i][1] - g_y/len(coordinates)    
         cen_coordinates[i][2]  = coordinates[i][2] - g_z/len(coordinates)       
         
    cen_coordinates = np.array(cen_coordinates)
    #calculate the eigen values and vigen vectors
    cen_coordinates_cov=np.cov(cen_coordinates.transpose())
    #eigenvalues = np.linalg.eigvals(cen_coordinates_cov)
    #eigenvecs = np.linalg.eig(cen_coordinates_cov)
    w, v = np.linalg.eig(cen_coordinates_cov)
    return w,v

def get_sigen_vec(name,pdb_name):
    
    loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
    loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
    
     
    thresh_hold_ca=7.2# pickle.load(open("max_depths_ca_MOTIF.p", "rb"))  
    thresh_hold_res=6.7 #pickle.load(open("max_depths_res_MOTIF.p", "rb"))  
    
    os.chdir('/')
    os.chdir(loading_pikle_dir)
    coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
#    aminoacids = pickle.load( open( ''.join(["amino_acid_",pdb_name,".p"]), "rb" ))
    fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
    fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
    MOTIF_indexs_all = pickle.load( open( ''.join(["MOTIF_indexs_all_",pdb_name,".p"]), "rb" )) 
    
    c_alpha_indexes_MOTIF= sum(MOTIF_indexs_all, [])
    res_factor = 2.25 # see the documentation twhy 2.25 is chosen
    sur_res_cor_intial = []
    MOTIF_prop =[]
    sur_res_MOTIF=[]
    #% to find out the surface atoms residues 
    for i in range(0,len(fin_res_depth_all)):
        if fin_ca_depth_all[i] <= thresh_hold_ca:
            if fin_res_depth_all[i] <= thresh_hold_res:
                # multiply each coordinate by 2 (just for increasing the resolution) and then round them to decimal numbers.
                #sur_res_cor_intial_round.append([round(res_factor*coordinates[i][0]),round(res_factor*coordinates[i][1]),round(res_factor*coordinates[i][2])])
                sur_res_cor_intial.append([res_factor*coordinates[i][0],res_factor*coordinates[i][1],res_factor*coordinates[i][2]])        
                
                if i in c_alpha_indexes_MOTIF:
                    sur_res_MOTIF.append([res_factor*coordinates[i][0],res_factor*coordinates[i][1],res_factor*coordinates[i][2]])        
                else:
                    MOTIF_prop.append(0)
    if len(sur_res_cor_intial)==0:
        print(pdb_name," Has no atoms")
        eig_vec_Ca=np.zeros((3,3))
        e_val_Ca=np.zeros((3,1))
    else:         
        e_val_Ca, eig_vec_Ca =get_C_alpha_optimal_direc(sur_res_cor_intial)
    if len(sur_res_MOTIF)>0:
        e_val_MOTIF, eig_vec_MOTIF =get_C_alpha_optimal_direc(sur_res_MOTIF)
    else:   
        print(pdb_name," Has no MOTIF atoms")
        eig_vec_MOTIF=np.zeros((3,3))
        e_val_MOTIF=np.zeros((3,1))
    return eig_vec_Ca,e_val_Ca,eig_vec_MOTIF,e_val_MOTIF


#%% 
def ext_eig_fun(name):
#    name='ONGO'    
    loading_dir ="C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/MOTIF_results/suceed_unigene"   
  
    os.chdir("/")
    os.chdir(loading_dir)
#    test_pdb_ids_details  = pickle.load(open( ''.join([name,"_test_pdb_ids_details.p"]), "rb" ) )
    train_pdb_ids_details =  pickle.load(open( ''.join([name,"_train_pdb_ids_details.p"]), "rb" ))
    
    loading_dir ="F:/BIBM_suceed_model_eig_direc_chk"   
    os.chdir("/")
    os.chdir(loading_dir)
    exp_most_train_list_ids =  pickle.load(open('exp_most_train_list_ids.p', "rb" ))
    print('exp_most_train_list_ids: ',len(exp_most_train_list_ids))
    Eigen_detail_all=[]
    Eigen_detail_not_predicted_well=[]

    train_pdb_ids_details = sum(train_pdb_ids_details,[])
    for pdb_name in train_pdb_ids_details:
    #    print(pdb_name, ' in progress')
        if ''.join([pdb_name,'.npy']) in exp_most_train_list_ids:
            if '2J5E'!=pdb_name and '721P'!=pdb_name and '1A07'!=pdb_name and  '1ZS8' !=pdb_name and '4XXC'!=pdb_name and '5KNM'!=pdb_name and '2FBT'!=pdb_name:
                print(pdb_name,' already done')
#                eig_vec_Ca,e_val_Ca,eig_vec_MOTIF,e_val_MOTIF= get_sigen_vec(name,pdb_name)
#                Eigen_detail_all.append([pdb_name,deepcopy(eig_vec_Ca),deepcopy(e_val_Ca),deepcopy(eig_vec_MOTIF),deepcopy(e_val_MOTIF)])
        else:
            if '1PVH'!=pdb_name and '2GV2'!=pdb_name and '4MDQ'!=pdb_name and '1MHE'!=pdb_name:
                print(pdb_name,' working')
                eig_vec_Ca,e_val_Ca,eig_vec_MOTIF,e_val_MOTIF= get_sigen_vec(name,pdb_name)
                Eigen_detail_not_predicted_well.append([pdb_name,deepcopy(eig_vec_Ca),deepcopy(e_val_Ca),deepcopy(eig_vec_MOTIF),deepcopy(e_val_MOTIF)])

#    eig_vec_Ca_all=np.zeros((3,3))
#    eig_vec_MOTIF_all=np.zeros((3,3))
    
    eig_vec_Ca_nope=np.zeros((3,3))
    eig_vec_MOTIF_nope=np.zeros((3,3))
##    e_val_MOTIF=np.zeros((3,1))
#    for vec in Eigen_detail_all:
#        eig_vec_Ca_all=eig_vec_Ca_all + vec[1]
#        eig_vec_MOTIF_all=eig_vec_MOTIF_all+vec[3]
#    eig_vec_Ca_avg = eig_vec_Ca_all/len(Eigen_detail_all)
#    eig_vec_MOTIF_avg=eig_vec_MOTIF_all/len(Eigen_detail_all)
    
    for vec in Eigen_detail_not_predicted_well:
        eig_vec_Ca_nope=eig_vec_Ca_nope + vec[1]
        eig_vec_MOTIF_nope=eig_vec_MOTIF_nope+vec[3]
  
    eig_vec_Ca_nope_avg = eig_vec_Ca_nope/len(Eigen_detail_not_predicted_well)
    eig_vec_MOTIF_nope_avg = eig_vec_MOTIF_nope/len(Eigen_detail_not_predicted_well)
#    return eig_vec_Ca_avg,eig_vec_MOTIF_avg,eig_vec_Ca_nope_avg,eig_vec_MOTIF_nope_avg
    return eig_vec_Ca_nope_avg,eig_vec_MOTIF_nope_avg

#%%
name = "ONGO"
##Eigen_detail_all_ONGO = ext_eig_fun(name)
##OG_eig_vec_Ca_avg,OG_eig_vec_MOTIF_avg,
OG_eig_vec_Ca_nope_avg,OG_eig_vec_MOTIF_nope_avg=ext_eig_fun(name)

name = "TSG"
#Eigen_detail_all_TSG = ext_eig_fun(name)
#TSG_eig_vec_Ca_avg,TSG_eig_vec_MOTIF_avg,
TSG_eig_vec_Ca_nope_avg,TSG_eig_vec_MOTIF_nope_avg=ext_eig_fun(name)

name = "Fusion"
#Eigen_detail_all_Fusion = ext_eig_fun(name)
#Fusion_eig_vec_Ca_avg,Fusion_eig_vec_MOTIF_avg,
Fusion_eig_vec_Ca_nope_avg,Fusion_eig_vec_MOTIF_nope_avg=ext_eig_fun(name)
#%%
#differ_between_OG_TSG=OG_eig_vec_Ca_avg-TSG_eig_vec_Ca_avg
#whole_OG_TSG_avg= (OG_eig_vec_Ca_avg+TSG_eig_vec_Ca_avg)/2
#%%

loading_dir ="F:/BIBM_suceed_model_eig_direc_chk"   
os.chdir("/")
os.chdir(loading_dir)
#pickle.dump(OG_eig_vec_Ca_avg, open("OG_eig_vec_Ca_avg.p", "wb"))  
#pickle.dump(OG_eig_vec_MOTIF_avg, open("OG_eig_vec_MOTIF_avg.p", "wb"))  
#pickle.dump(TSG_eig_vec_Ca_avg, open("TSG_eig_vec_Ca_avg.p", "wb"))  
#pickle.dump(TSG_eig_vec_MOTIF_avg, open("TSG_eig_vec_MOTIF_avg.p", "wb"))  
#pickle.dump(Fusion_eig_vec_Ca_avg, open("Fusion_eig_vec_Ca_avg.p", "wb"))  
#pickle.dump(Fusion_eig_vec_MOTIF_avg, open("Fusion_eig_vec_MOTIF_avg.p", "wb"))  

pickle.dump(OG_eig_vec_Ca_nope_avg, open("OG_eig_vec_Ca_nope_avg.p", "wb"))  
pickle.dump(OG_eig_vec_MOTIF_nope_avg, open("OG_eig_vec_MOTIF_nope_avg.p", "wb"))  
pickle.dump(TSG_eig_vec_Ca_nope_avg, open("TSG_eig_vec_Ca_nope_avg.p", "wb"))  
pickle.dump(TSG_eig_vec_MOTIF_nope_avg, open("TSG_eig_vec_MOTIF_nope_avg.p", "wb"))  
pickle.dump(Fusion_eig_vec_Ca_nope_avg, open("Fusion_eig_vec_Ca_nope_avg.p", "wb"))  
pickle.dump(Fusion_eig_vec_MOTIF_nope_avg, open("Fusion_eig_vec_MOTIF_nope_avg.p", "wb"))  
#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#        print('xs: ',xs)
#        print('ys: ',ys)
#        print('zs: ',zs)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

## plotting in 3d
fig = plt.figure()
plt.figure(figsize=(5,5))

ax = fig.add_subplot(111, projection='3d')

centroid = np.array([0, 0,0])
ax.scatter(centroid[0], centroid[1], centroid[2], marker='o', color='r')
# labelling the axes
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")

ax.set_title('Eigen vector ')
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-0.2, 0.2)
ax.set_zlim(-0.2, 0.3)

##########################################################################
# plotting Eigen vectors
##########################################################################
for vec in OG_eig_vec_Ca_avg.T:  # fetching one vector from list of eigvecs
    # drawing the vec, basically drawing a arrow form centroid to the end
    # point of vec
    vec += centroid
    drawvec = Arrow3D([centroid[0], vec[0]], [centroid[1], vec[1]], [centroid[2], vec[2]],
                      mutation_scale=20, lw=3, arrowstyle="-|>", color='r')
    # adding the arrow to the plot
    ax.add_artist(drawvec)


for vec in TSG_eig_vec_Ca_avg.T:  # fetching one vector from list of eigvecs
    # drawing the vec, basically drawing a arrow form centroid to the end
    # point of vec
    vec += centroid
    drawvec = Arrow3D([centroid[0], vec[0]], [centroid[1], vec[1]], [centroid[2], vec[2]],
                      mutation_scale=20, lw=3, arrowstyle="-|>", color='b')
    # adding the arrow to the plot
    ax.add_artist(drawvec)
    
#for vec in Fusion_eig_vec_Ca_avg.T:  # fetching one vector from list of eigvecs
#    # drawing the vec, basically drawing a arrow form centroid to the end
#    # point of vec
#    vec += centroid
#    drawvec = Arrow3D([centroid[0], vec[0]], [centroid[1], vec[1]], [centroid[2], vec[2]],
#                      mutation_scale=1, lw=3, arrowstyle="-|>", color='g')
#    # adding the arrow to the plot
#    ax.add_artist(drawvec)
# plot show
plt.show()
print("Since the MOTIF eigen have imaginary part")
#%%
'''Detail checking'''
    
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
Eigen_detail_all=[]
train_pdb_ids_details,test_pdb_ids_details,test_unigene_details = pdb_details_fun(name)
test_pdb_ids_details = sum(test_pdb_ids_details,[])
for pdb_name in test_pdb_ids_details:
#    print(pdb_name, ' in progress')
    if '2J5E'!=pdb_name and '721P'!=pdb_name:
        eig_vec_Ca,e_val_Ca,eig_vec_MOTIF,e_val_MOTIF= get_sigen_vec(name,pdb_name)
        Eigen_detail_all.append([pdb_name,deepcopy(eig_vec_Ca),deepcopy(e_val_Ca),deepcopy(eig_vec_MOTIF),deepcopy(e_val_MOTIF)])
#name = "TSG"
#name = "Fusion"
#pdb_name='121p'
#%%
        
pdb_name = '721P'
loading_pikle_dir_part = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/msms_pikles/"
loading_pikle_dir = ''.join([loading_pikle_dir_part,name])
    
     
thresh_hold_ca=7.2# pickle.load(open("max_depths_ca_MOTIF.p", "rb"))  
thresh_hold_res=6.7 #pickle.load(open("max_depths_res_MOTIF.p", "rb"))  

os.chdir('/')
os.chdir(loading_pikle_dir)
coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
#    aminoacids = pickle.load( open( ''.join(["amino_acid_",pdb_name,".p"]), "rb" ))
fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
MOTIF_indexs_all = pickle.load( open( ''.join(["MOTIF_indexs_all_",pdb_name,".p"]), "rb" )) 

c_alpha_indexes_MOTIF= sum(MOTIF_indexs_all, [])
res_factor = 2.25 # see the documentation twhy 2.25 is chosen
sur_res_cor_intial = []
MOTIF_prop =[]
sur_res_MOTIF=[]
#% to find out the surface atoms residues 
for i in range(0,len(fin_res_depth_all)):
    if fin_ca_depth_all[i] <= thresh_hold_ca:
        if fin_res_depth_all[i] <= thresh_hold_res:
            # multiply each coordinate by 2 (just for increasing the resolution) and then round them to decimal numbers.
            #sur_res_cor_intial_round.append([round(res_factor*coordinates[i][0]),round(res_factor*coordinates[i][1]),round(res_factor*coordinates[i][2])])
            sur_res_cor_intial.append([res_factor*coordinates[i][0],res_factor*coordinates[i][1],res_factor*coordinates[i][2]])        
            
            if i in c_alpha_indexes_MOTIF:
                sur_res_MOTIF.append([res_factor*coordinates[i][0],res_factor*coordinates[i][1],res_factor*coordinates[i][2]])        
            else:
                MOTIF_prop.append(0)
#%%
if len(sur_res_cor_intial)==0:
    print(pdb_name," Has no atoms")
    eig_vec_Ca=np.zeros((3,3))
    e_val_Ca=np.zeros((3,1))
else:         
    e_val_Ca, eig_vec_Ca =get_C_alpha_optimal_direc(sur_res_cor_intial)
if len(sur_res_MOTIF)>0:
    e_val_MOTIF, eig_vec_MOTIF =get_C_alpha_optimal_direc(sur_res_MOTIF)
else:   
    print(pdb_name," Has no MOTIF atoms")
    eig_vec_MOTIF=np.zeros((3,3))
    e_val_MOTIF=np.zeros((3,1))