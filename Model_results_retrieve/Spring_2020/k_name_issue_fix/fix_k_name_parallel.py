# -*- coding: utf-8 -*-
"""
Created on  %(11-Feb-2020) 09.30Pm

@author: %A.Nishanth C00294860

This script is formed to change the name and save it in the same place with the different name
for the models
"""
import os
from copy import deepcopy

load_model_main_dir='E:/scrach_cacs_1083/Before_thresh_changed/optimaly_tilted_17_quarter/10-fold/'
model_name = 'model_accidental_p1_3_d1_32'
#model_name = 'model_accidental_p1_5_d1_32'

load_model_dir=''.join([load_model_main_dir,model_name])
#%%
os.chdir('/')
os.chdir(load_model_dir)

lists_of_file=os.listdir()
model_names_old=[]
for files in lists_of_file:
    if files[-3:-1]=='.h':
        if files[0:27]==model_name:
            model_names_old.append(deepcopy(files))

for load_model in model_names_old:
    'load the model here'
    model_new_name=''.join(['model_accidental_par',model_name[16:27],load_model[28:-1],'5'])
    'save the model with new name here'
    print(model_new_name)