# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
#from copy import deepcopy
import pickle
import os
#%%
#raise("Change the pickles to load mode")
loading_dir ="F:/BIBM_suceed_model_eig_direc_chk"   
os.chdir("/")
os.chdir(loading_dir)

OG_eig_vec_Ca_avg = pickle.load(open("OG_eig_vec_Ca_avg.p","rb"))
OG_eig_vec_MOTIF_avg = pickle.load(open("OG_eig_vec_MOTIF_avg.p","rb"))
TSG_eig_vec_Ca_avg = pickle.load(open("TSG_eig_vec_Ca_avg.p","rb"))
TSG_eig_vec_MOTIF_avg = pickle.load(open("TSG_eig_vec_MOTIF_avg.p","rb"))
Fusion_eig_vec_Ca_avg = pickle.load(open("OG_eig_vec_Ca_avg.p","rb"))


OG_eig_vec_Ca_nope_avg = pickle.load(open("OG_eig_vec_Ca_nope_avg.p","rb"))
OG_eig_vec_MOTIF_nope_avg = pickle.load(open("OG_eig_vec_MOTIF_nope_avg.p","rb"))
TSG_eig_vec_Ca_nope_avg = pickle.load(open("TSG_eig_vec_Ca_nope_avg.p","rb"))
TSG_eig_vec_MOTIF_nope_avg = pickle.load(open("TSG_eig_vec_MOTIF_nope_avg.p","rb"))
Fusion_eig_vec_Ca_nope_avg = pickle.load(open("OG_eig_vec_Ca_nope_avg.p","rb"))
#%%
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

## plotting in 3d
fig = plt.figure()
plt.figure(figsize=(0.5,0.5))

ax = fig.add_subplot(111, projection='3d')

centroid = np.array([0, 0,0])
ax.scatter(centroid[0], centroid[1], centroid[2], marker='o', color='r')
# labelling the axes
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")

#ax.set_title('Eigen vector ')
ax.set_xlim(-0.1, 0.15)
ax.set_ylim(-0.05, 0.2)
ax.set_zlim(-0.1, 0.4)


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
l1 = ax.legend([], [" 1"])
    

for vec in TSG_eig_vec_Ca_avg.T:  # fetching one vector from list of eigvecs
    # drawing the vec, basically drawing a arrow form centroid to the end
    # point of vec
    vec += centroid
    drawvec2 = Arrow3D([centroid[0], vec[0]], [centroid[1], vec[1]], [centroid[2], vec[2]],
                      mutation_scale=20, lw=3, arrowstyle="-|>", color='b')
    # adding the arrow to the plot
    ax.add_artist(drawvec2)
#ax.legend(['Correct_ONGO_eig_vec_avg','Correct_TSG_eig_vec_avg'])

for vec in OG_eig_vec_Ca_nope_avg.T:
    # drawing the vec, basically drawing a arrow form centroid to the end
    # point of vec
    vec += centroid
    drawvec3 = Arrow3D([centroid[0], vec[0]], [centroid[1], vec[1]], [centroid[2], vec[2]],
                      mutation_scale=20, lw=3, arrowstyle="-|>", color='y')
    # adding the arrow to the plot
    ax.add_artist(drawvec3)

for vec in TSG_eig_vec_Ca_nope_avg.T:
    # drawing the vec, basically drawing a arrow form centroid to the end
    # point of vec
    vec += centroid
    drawvec4 = Arrow3D([centroid[0], vec[0]], [centroid[1], vec[1]], [centroid[2], vec[2]],
                      mutation_scale=20, lw=3, arrowstyle="-|>", color='c')
    # adding the arrow to the plot
    ax.add_artist(drawvec4)
    
ax.legend([drawvec,drawvec2,drawvec3,drawvec4], ["Correct_ONGO_eig_vec_avg","Correct_TSG_eig_vec_avg","Wrong_ONGO_eig_vec_avg","Wrong_TSG_eig_vec_avg"],title='Average Eigen vector of  \n surface C_a coordinates', bbox_to_anchor=(0.45, 1), loc='upper left', borderaxespad=0.)#ncol=2
        
#for vec in Fusion_eig_vec_Ca_avg.T:  # fetching one vector from list of eigvecs
#    # drawing the vec, basically drawing a arrow form centroid to the end
#    # point of vec
#    vec += centroid
#    drawvec = Arrow3D([centroid[0], vec[0]], [centroid[1], vec[1]], [centroid[2], vec[2]],
#                      mutation_scale=1, lw=3, arrowstyle="-|>", color='g')
#    # adding the arrow to the plot
#    ax.add_artist(drawvec)
# plot show\
#ax.legend()
print("Since the MOTIF eigen have imaginary part")


