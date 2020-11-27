#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:29:45 2019

@author: c00294860
"""
from model_train_class_dat_k import ovear_all_training

'''After learned inception and Residual Deep_NN'''
from Deep_model_architechtures_quat import model_par_brain_inception_only_vin_1
#from Deep_model_architechtures_quat import model_brain_incept_RESIDUAL_quat_vin_1

'''Not all the inception is done here'''
from copy import deepcopy
import pickle
import os


'''To save the results'''
overall_history_log_list=[]
highest_test_accuracy_list=[]
model_name_list=[]

def chk_to_save(overall_history_log_list,highest_test_accuracy_list,model_name_list,highest_test_accuracy,overall_history_log,not_satisfied,model_name):
    '''To save the results'''

    highest_test_accuracy_list.append(deepcopy(highest_test_accuracy))
    overall_history_log_list.append(deepcopy(overall_history_log))
    model_name_list.append(deepcopy(model_name))
    if not_satisfied:
        return overall_history_log_list,highest_test_accuracy_list,model_name_list
    else:
        '''To save the results for future reference'''
        os.chdir('/')
        os.chdir('scratch/optimaly_tilted/Train')
        pickle.dump(overall_history_log_list, open(''.join([model_name,"_overall_history_log.p"]), "wb"))  
        pickle.dump(highest_test_accuracy_list, open(''.join([model_name,"highest_test_accuracy_list.p"]), "wb"))  
        pickle.dump(model_name_list, open("model_name_list.p", "wb"))  
        return overall_history_log_list,highest_test_accuracy_list,model_name_list
    
    
def call_channel(channel_number,activation,number_of_overall_iertations_to_the_model,model_loss,optimizer,Full_clean=True,balance_batch_must=False,tower_min_max_only=False,shuffle=True):
    '''
    Default values
    
    Full_clean=True
    
    validation_split=0.2
        
    balance_batch_must=False
    tower_min_max_only=False
    '''
    print('-------------  training channel no: ',channel_number)
    print('')
    print('-------------  Activation function: ',activation)
    print('')

    '''To save the results'''
    overall_history_log_list=[]
    highest_test_accuracy_list=[]
    model_name_list=[]

    '''model_chk `1'''
    model_train = ovear_all_training(channels=channel_number,number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model,model_loss=model_loss,optimize=optimizer,Full_clean=Full_clean,balance_batch_must=balance_batch_must,shuffle=shuffle)
    model_chk_model = model_par_brain_inception_only_vin_1(channels=channel_number,activation=activation)
    model,model_name= model_chk_model.model_maker()#d1=80,d2=16
    highest_test_accuracy,overall_history_log,not_satisfied1 =model_train.model_parameter_train(model,model_name)
    del model
    del model_train
    overall_history_log_list,highest_test_accuracy_list,model_name_list =  chk_to_save(overall_history_log_list,highest_test_accuracy_list,model_name_list,highest_test_accuracy,overall_history_log,not_satisfied1,model_name)  

#    raise("k-fold for loop in 200th line change to start from 0")
#    model_train = ovear_all_training(channels=channel_number,number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model,model_loss=model_loss,optimize=optimizer,Full_clean=Full_clean,balance_batch_must=balance_batch_must,shuffle=shuffle)
#    model_chk_model = model_brain_incept_RESIDUAL_quat_vin_1(channels=channel_number,activation=activation)
#    model,model_name= model_chk_model.model_maker()#d1=80,d2=16
#    highest_test_accuracy,overall_history_log,not_satisfied =model_train.model_parameter_train(model,model_name)
#    del model
#    del model_train
#    overall_history_log_list,highest_test_accuracy_list,model_name_list =  chk_to_save(overall_history_log_list,highest_test_accuracy_list,model_name_list,highest_test_accuracy,overall_history_log,not_satisfied1,model_name)  
        
    
    if  not_satisfied1:
        print("Try with different dataset or model")
        return False

    else:
        print(channel_number, ' satisfied some where chk')
        return True

#first find which dataset works best with the model
Full_clean=True
balance_batch_must=True
validation_split=0.2
#tower_min_max_only=True
number_of_overall_iertations_to_the_model=1

activation =  'swish'

chk_bal_tow_2_swish =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True)

#model_brain_incept_RESIDUAL_quat_vin_1