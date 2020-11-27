#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:29:45 2019

@author: c00294860
"""
from model_train_class_dat_valid_overall_vin_4 import ovear_all_training
'''Models'''
#from Deep_model_architechtures import model_par_inception_w_o_addition_vin_exp
#from Deep_model_architechtures import model_accidental
#from Deep_model_architechtures import model_inception_layer_1
#from Deep_model_architechtures import model_inception_layer_1_normal
#from Deep_model_architechtures import model_inception_layer_1_alias_1
#from Deep_model_architechtures import model_inception_layer_1_normal_alias_1
#from Deep_model_architechtures import model_inception_layer_1_normal_alias_2
#from Deep_model_architechtures import model_vin_4
#from Deep_model_architechtures import model_inception_vin_1
#from Deep_model_architechtures import model_inception_w_o_addition_vin_1
#from Deep_model_architechtures import model_inception_all__depths__inception_vin_1
#from Deep_model_architechtures import model_parallel_inception_all__depths_min_max_inception_vin_1
#from Deep_model_architechtures import model_inception_all__depths__inception_complic_vin_1
#from Deep_model_architechtures import model_parallel_inception_all__depths_min_max_inception_complic_vin_1
'''import parallel towers'''
#from Deep_model_architechtures import model_par_inception_all__depths__inception_vin_1
#from Deep_model_architechtures import model_par_inception_w_o_addition_vin_1
#from Deep_model_architechtures import model_par_parallel_inception_all__depths_min_max_inception_vin_1
#from Deep_model_architechtures import model_par_inception_all__depths__inception_complic_vin_1
#from Deep_model_architechtures import model_par_parallel_inception_all__depths_min_max_inception_complic_vin_1
#from Deep_model_architechtures import model_par_parallel_inception_all_k3x3_k_5_depths_min_max_inception_vin_1

'''After learned inception and Residual Deep_NN'''
#from Deep_model_architechtures import model_par_inception_only_mentor_vin_1
#from Deep_model_architechtures import model_inception_only_mentor_vin_1
from Deep_model_architechtures import model_par_inception_residual_mentor_vin_1
from Deep_model_architechtures import model_inception_residual_mentor_vin_1
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
        pickle.dump(overall_history_log_list, open("overall_history_log.p", "wb"))  
        pickle.dump(highest_test_accuracy_list, open("highest_test_accuracy_list.p", "wb"))  
        pickle.dump(model_name_list, open("model_name_list.p", "wb"))  
        return overall_history_log_list,highest_test_accuracy_list,model_name_list
    
    
def call_channel(channel_number,activation,number_of_overall_iertations_to_the_model,model_loss,optimizer,Full_clean=True,balance_batch_must=False,validation_split=0.15,tower_min_max_only=False,shuffle=False,SITE_MUST=True):
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
    satisfied=False
    not_satisfied=True
#    model_inc_layer_1_chk=0

#    if not_satisfied:
#        '''model_chk `1'''
#        model_chk=1
#        model_train = ovear_all_training(channels=channel_number,number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model,model_loss=model_loss,optimize=optimizer,Full_clean=Full_clean,balance_batch_must=balance_batch_must,validation_split=validation_split,shuffle=shuffle,SITE_MUST=SITE_MUST)
#        model_chk_model = model_inception_only_mentor_vin_1(channels=channel_number,tower_min_max_only=tower_min_max_only,activation=activation)
#        model,model_name= model_chk_model.model_maker()#d1=80,d2=16
#        highest_test_accuracy,overall_history_log,not_satisfied =model_train.model_parameter_train(model,model_name)
#        del model
#        del model_train
#        overall_history_log_list,highest_test_accuracy_list,model_name_list =  chk_to_save(overall_history_log_list,highest_test_accuracy_list,model_name_list,highest_test_accuracy,overall_history_log,not_satisfied,model_name)

    if not_satisfied:
        '''model_chk `1'''
        model_train = ovear_all_training(channels=channel_number,number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model,model_loss=model_loss,optimize=optimizer,Full_clean=Full_clean,balance_batch_must=balance_batch_must,validation_split=validation_split,shuffle=shuffle,SITE_MUST=SITE_MUST)
        model_chk_model = model_par_inception_residual_mentor_vin_1(channels=channel_number,tower_min_max_only=tower_min_max_only,activation=activation)
        model,model_name= model_chk_model.model_maker()#d1=80,d2=16
        highest_test_accuracy,overall_history_log,not_satisfied =model_train.model_parameter_train(model,model_name)
        del model
        del model_train
        overall_history_log_list,highest_test_accuracy_list,model_name_list =  chk_to_save(overall_history_log_list,highest_test_accuracy_list,model_name_list,highest_test_accuracy,overall_history_log,not_satisfied,model_name)

    if not_satisfied:
        '''model_chk `1'''
        model_train = ovear_all_training(channels=channel_number,number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model,model_loss=model_loss,optimize=optimizer,Full_clean=Full_clean,balance_batch_must=balance_batch_must,validation_split=validation_split,shuffle=shuffle,SITE_MUST=SITE_MUST)
        model_chk_model = model_inception_residual_mentor_vin_1(channels=channel_number,tower_min_max_only=tower_min_max_only,activation=activation)
        model,model_name= model_chk_model.model_maker()#d1=80,d2=16
        highest_test_accuracy,overall_history_log,not_satisfied =model_train.model_parameter_train(model,model_name)
        del model
        del model_train
        overall_history_log_list,highest_test_accuracy_list,model_name_list =  chk_to_save(overall_history_log_list,highest_test_accuracy_list,model_name_list,highest_test_accuracy,overall_history_log,not_satisfied,model_name) 
    '''memory issue run on higher GPU machine'''
#    if not_satisfied:
#        '''model_chk `1'''
#        model_train = ovear_all_training(channels=channel_number,number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model,model_loss=model_loss,optimize=optimizer,Full_clean=Full_clean,balance_batch_must=balance_batch_must,validation_split=validation_split,shuffle=shuffle,SITE_MUST=SITE_MUST)
#        model_chk_model = model_par_inception_residual_mentor_vin_1(channels=channel_number,tower_min_max_only=tower_min_max_only,activation=activation)
#        model,model_name= model_chk_model.model_maker(f_inc=2,f_d=1)
#        highest_test_accuracy,overall_history_log,not_satisfied =model_train.model_parameter_train(model,model_name)
#        del model
#        del model_train
#        overall_history_log_list,highest_test_accuracy_list,model_name_list =  chk_to_save(overall_history_log_list,highest_test_accuracy_list,model_name_list,highest_test_accuracy,overall_history_log,not_satisfied,model_name)
#
#    if not_satisfied:
#        '''model_chk `1'''
#        model_train = ovear_all_training(channels=channel_number,number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model,model_loss=model_loss,optimize=optimizer,Full_clean=Full_clean,balance_batch_must=balance_batch_must,validation_split=validation_split,shuffle=shuffle,SITE_MUST=SITE_MUST)
#        model_chk_model = model_inception_residual_mentor_vin_1(channels=channel_number,tower_min_max_only=tower_min_max_only,activation=activation)
#        model,model_name= model_chk_model.model_maker(f_inc=2,f_d=1)
#        highest_test_accuracy,overall_history_log,not_satisfied =model_train.model_parameter_train(model,model_name)
#        del model
#        del model_train
#        overall_history_log_list,highest_test_accuracy_list,model_name_list =  chk_to_save(overall_history_log_list,highest_test_accuracy_list,model_name_list,highest_test_accuracy,overall_history_log,not_satisfied,model_name) 
#     
#    

#    if not_satisfied:
#        '''model_chk `1'''
#        model_chk=1
#        model_train = ovear_all_training(channels=channel_number,number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model,model_loss=model_loss,optimize=optimizer,Full_clean=Full_clean,balance_batch_must=balance_batch_must,validation_split=validation_split,shuffle=shuffle,SITE_MUST=SITE_MUST)
#        model_chk_model = model_par_inception_only_mentor_vin_1(channels=channel_number,tower_min_max_only=tower_min_max_only,activation=activation)
#        model,model_name= model_chk_model.model_maker()#d1=80,d2=16
#        highest_test_accuracy,overall_history_log,not_satisfied =model_train.model_parameter_train(model,model_name)
#        del model
#        del model_train
#        overall_history_log_list,highest_test_accuracy_list,model_name_list =  chk_to_save(overall_history_log_list,highest_test_accuracy_list,model_name_list,highest_test_accuracy,overall_history_log,not_satisfied,model_name)
  
  
    if  not_satisfied:
        print("Try with different dataset or model")
        return False

    else:
        print(channel_number, ' satisfied some where chk')
        return True

#first find which dataset works best with the model
Full_clean=False
balance_batch_must=True
validation_split=0.2
tower_min_max_only=True
number_of_overall_iertations_to_the_model=2

activation =  'swish'

#chk_bal_tow_2_swish =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=True)
#if not chk_bal_tow_2_swish:
chk_bal_2_swish = call_channel(20,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=False,SITE_MUST=False)
if not chk_bal_2_swish:
    print(" -------------------- Running on SITE must new prop with threshold changed ----------------- ")
    chk_bal_2_swish_SITE_MUST =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=False,SITE_MUST=True)
    if not chk_bal_2_swish_SITE_MUST:
        print(" --------------------- Running on SITE must with old threshold  ---------------")
        chk_bal_2_swish_SITE_MUST_old =call_channel(17,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=False,SITE_MUST=True,fusion_test_chk=0.2)

if chk_bal_2_swish or chk_bal_2_swish_SITE_MUST:
    chk_bal_2_swish_with_fusion = call_channel(20,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=False,SITE_MUST=False,fusion_test_chk=0.2)
    if not chk_bal_2_swish_with_fusion:
        chk_bal_2_swish_SITE_MUST_with_fusion =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=False,SITE_MUST=True,fusion_test_chk=0.2)


#    if not chk_bal_2_swish:
#        activation =  'relu'
#        chk_bal_tow_2_relu =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=True)
#        if not chk_bal_tow_2_relu:
#            chk_bal_2_relu =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=False)
#    
#            activation =  'swish'                   
#            print("")
#            if not chk_bal_2_relu:
#                print('')
#                print('')
#                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<----  Shuffle activated  here-on   number_of_overall_iertations_to_the_model: 2 ---->>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#                print('')
#                number_of_overall_iertations_to_the_model=1
#                print('')
#                chk_bal_tow_2_shuf_1_swish =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=True,shuffle=True)
#                if not chk_bal_tow_2_shuf_1_swish:   
#                    chk_bal_2_shuf_1_swish =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=False,shuffle=True)
#                    if not chk_bal_2_shuf_1_swish:
#                        print('')
#                        print('')
#                        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<----  Shuffled & Not balanced number_of_overall_iertations_to_the_model: 1 ---->>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#                        print('')
#                        number_of_overall_iertations_to_the_model=1
#                        print('')
#                        chk_bal_tow_2_shuf_Not_bal1_swish =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=True,shuffle=True)
#                        if not chk_bal_tow_2_shuf_Not_bal1_swish:
#                            chk_bal_2_shuf_Not_bal1_swish =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=False,shuffle=True)
#                            if not chk_bal_2_shuf_Not_bal1_swish:
#                                print('')
#                                print('')
#                                print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<----  Shuffle activated  here-on   number_of_overall_iertations_to_the_model: 2 ---->>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#                                print('')
#                                number_of_overall_iertations_to_the_model=1
#                                print('')
#                                chk_bal_tow_2_shuf_1_relu =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=True,shuffle=True)
#                                if not chk_bal_tow_2_shuf_1_relu:   
#                                    chk_bal_2_shuf_1_relu =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=False,shuffle=True)
#                                    if not chk_bal_2_shuf_1_relu:
#                                        print('')
#                                        print('')
#                                        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<----  Shuffled & Not balanced number_of_overall_iertations_to_the_model: 1 ---->>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#                                        print('')
#                                        number_of_overall_iertations_to_the_model=1
#                                        print('')
#                                        chk_bal_tow_2_shuf_Not_bal1_relu =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=True,shuffle=True)
#                                        if not chk_bal_tow_2_shuf_Not_bal1_relu:
#                                            chk_bal_2_shuf_Not_bal1_relu =call_channel(21,activation,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',Full_clean=Full_clean,balance_batch_must=True,tower_min_max_only=False,shuffle=True)



# chk_bal_2 =call_channel(21,number_of_overall_iertations_to_the_model,model_loss='msle',optimizer='RMSprop',balance_batch_must=True)
# chk_bal_0 =call_channel(21,number_of_overall_iertations_to_the_model,model_loss='categorical_crossentropy',optimizer='Adam',balance_batch_must=True)

# chk_bal_0 =call_channel(21,number_of_overall_iertations_to_the_model,model_loss='categorical_crossentropy',optimizer='Adam',balance_batch_must=True)
# chk_bal_3 =call_channel(21,number_of_overall_iertations_to_the_model,model_loss='categorical_hinge',optimizer='Adagrad',balance_batch_must=True)
# chk_bal_4= call_channel(21,number_of_overall_iertations_to_the_model,model_loss='categorical_hinge',optimizer='Adadelta',balance_batch_must=True)