#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:31:50 2020
@author: %A.Nishanth C00294860
"""
#%%
#import tensorflow as tf
from data_generation_valid_quat_10_fold_vin1 import DataGenerator_k_fold 
from data_generation_valid_quat_10_fold_vin1 import Data_test
from data_generation_valid_quat_10_fold_vin1 import DataGenerator_k_fold_given

from tensorflow import keras
from copy import deepcopy
import numpy as np
import os
import pickle

from time import sleep# to make the GPU cool down
import gc
'''
This approach doesn't consider Fusion at all in Validation performance check
'''
class ovear_all_training:
    def __init__(self,channels,total_iteration=100,number_of_overall_iertations_to_the_model=10,validation_steps_check=5,minimum_iteration_validation_chk=1,sleep_time=30, model_loss='MSE', optimize='Adam',Full_clean=True,balance_batch_must=True,k=10,shuffle=False,threshold=True,batch_size=10,SITE_MUST=True):
        '''Initialization
        channels        : the depth of the input
        total_iteration : This decide the number of time the data set is used; since the dataset is feed forward and backward once mean epoch count is one)
                            preferes if valid split 0.15 then 82    
                                if valid split 0.2 then 73   
        number_of_overall_iertations_to_the_model: To validate the model architechture gonna satisfy the condition or not, 
                            thus the training is done on  different ways like changing the hyper aprameters(like depth in each layer or kernal size)
                            of the model at each time
                            and check the performance changed or not   
        sleep_time       : define the waiting time(in milli seconds) to cool down between training sessions
        
        **
        checking the model whether it stuck in the suboptimal or optimal
        
        minimum_iteration_validation_chk: How many iterations(minimum should be 1) should run before going on check it reached optimal or not

        Inorder to do this first check the training accuracy cross the limitation such as 
                self.train_acc_limit=0.82 thus training accuracy must cross the given accuracy

            if the training accuracy cross the limitation then check Validation accuracy
            First check the validation acuracy is higher than 0.7(70%);
                
                validation_steps_check(limit assigned as 5): How many times the validation accuracy stucked in the same position; 
                                            to avoid leaving the optimal model
                                            if the accuracy is <70% then it is reset
                                            
            if validation accuracy <70%:
                validation_low_steps_check(limit assigned as 10): If the validation accuracy less than 70 for long time while the training accuracy is high

      '''
        self.model_loss=model_loss#'MSE'#"mean squared error"
        self.optimizer=optimize#'Adam'

        self.channels = channels
        print('channel initialised: ',self.channels)

        self.sleep_time = sleep_time
		
        #the parameters for data split
        self.k = k # howmany fold validation if k=10 means 10 fold cross validation
        self.shuffle = shuffle
        self.balance_batch_must=balance_batch_must
        self.Full_clean = Full_clean
        self.batch_size=batch_size
#        self.batch_size=10
        print("")
        print("Over ridden self.batch_size as ",self.batch_size)
        print("")
        self.load_train_size=10

        self.total_iteration=total_iteration #earlier v102
        self.number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model
        self.validation_steps_check = validation_steps_check#earliet its 2
        self.minimum_iteration_validation_chk=minimum_iteration_validation_chk
        
        self.valid_partial_sat =True#thiw skips the Vlaidation accuracy check limit lest atleast one batch cross the accuracy of TSG or OG in validation set
        self.validation_acc_chk_limit = 0.73
        self.validation_acc_OG_TSG_limit = 0.73
        self.train_acc_limit=0.82#to check the validation if it cross the training limitations 
        self.validation_low_steps_check=10
       
#        optimal_tilted=False
        if self.channels==21:
            self.main_dir_part ='scratch/before_thresh_change/optimaly_tilted_21_quarter_k_fold'
        elif self.channels==17:
            self.main_dir_part ='scratch/before_thresh_change/optimaly_tilted_17_quarter_k_fold'         
        self.SITE_MUST=True
#        if SITE_MUST:
#            self.SITE_MUST=True
#            if channels==21:
#                self.main_dir=''.join(['scratch/optimaly_tilted_21'])
#                if threshold and optimal_tilted:
#                    '''Different threshold condition for surface define'''
#                    self.main_dir =''.join(['scratch/21_aminoacids__thrsh_new_SITE_MUST'])
#                elif threshold:
#                    self.main_dir =''.join(['scratch/21_aminoacids_NO_TILT_thrsh_new_SITE_MUST'])
#            elif channels==20:
#                self.main_dir=''.join(['scratch/neg_size_20_21_aminoacids'])
#            elif channels==17:
#                if optimal_tilted:
#                    self.main_dir=''.join(['scratch/optimaly_tilted_17'])
#                else:
#                    self.main_dir=''.join(['scratch/Old_thresh/SITE_MUST_17'])
#
#            elif  channels==15:
#                if threshold:
#                    self.main_dir=''.join(['scratch/15_aminoacids_thrsh_new_SITE_MUST'])
#        else:
#            self.SITE_MUST=False
#            if channels==20:
##                self.main_dir=''.join(['scratch/optimaly_tilted_21'])
#                if threshold:
#                    '''Different threshold condition for surface define'''
#                    self.main_dir =''.join(['scratch/20_aminoacids__thrsh'])
#            if channels==14:
#                if threshold:
#                    self.main_dir=''.join(['scratch/neg_14_21_aminoacids'])       

    def validation_accuracy_reached_optimal(self,validation_accuracy_new,validation_accuracy_last,validation_count):
        '''
        This fucntion check
        if the training stucked in sub optimal point then retrive the model, to check the performance
       
        validation_count        : check howmany times the validation accuracy hasn't changed
        validation_accuracy_last: The validation accuracy got for last time
        
        it returns
            to stop the training(True) or (False)
        '''
        if validation_accuracy_new==validation_accuracy_last:
            validation_count = validation_count+1
            if validation_count>self.validation_steps_check:
                return True,self.validation_steps_check-1
            else:
                return False,validation_count
        else:
            return False,0#reset the validation count to 0
     
    def validation_accuracy_reached_low_optimal(self,validation_accuracy_new,validation_accuracy_last,validation_low_count):
        '''
        This fucntion check
        if the training stucked in sub optimal point then retrive the model, to check the performance
       
        validation_count        : check howmany times the validation accuracy hasn't changed
        validation_accuracy_last: The validation accuracy got for last time
        
        it returns
            to stop the training(True) or (False)
        '''
        if validation_accuracy_new==validation_accuracy_last:
            validation_low_count = validation_low_count + 1 
            if validation_low_count > self.validation_low_steps_check:
                return True,validation_low_count
            else:
                return False,validation_low_count
        else:
            return False,0#reset the validation count to 0    
        
    def validation_accuracy_reached_optimal_loop(self,validation_accuracy_new,validation_accuracy_last_1,validation_accuracy_last,validation_count_loop):
        '''
        This fucntion check
        if the training stucked in sub optimal point then retrive the model, to check the performance
        this check the optimal solution fell in loop(varying between optimal solution back and forward)
       
        validation_count        : check howmany times the validation accuracy hasn't changed
        validation_accuracy_last: The validation accuracy got for last time
        
        it returns
            to stop the training(True) or (False)
        '''

        if (validation_accuracy_new==validation_accuracy_last) and (validation_accuracy_last==validation_accuracy_last_1):

            validation_count_loop = validation_count_loop+1
            if validation_count_loop > (self.validation_steps_check):
                return True,((self.validation_steps_check)-1) # to check whether it stuctk in suboptimal again
            else:
                return False,validation_count_loop+1
        elif validation_accuracy_last<validation_accuracy_new:
            return False,0
        else:
            return False,(validation_count_loop-1)

        
    def model_parameter_train(self,model,model_name):
        
        self.main_dir =  ''.join([self.main_dir_part,'/',model_name[0:-3]])
        os.chdir('/')
        os.chdir(self.main_dir_part)
        '''To load the train dataset'''
        train_labels = pickle.load(open("train_labels_dic.p", "rb"))
        '''To create the dataset'''
        k_fold_training_generator = DataGenerator_k_fold(train_labels,self.main_dir_part,batch_size=self.load_train_size,n_channels=self.channels,Full_clean=self.Full_clean,balance_batch_must=self.balance_batch_must,shuffle=self.shuffle,k=self.k,model_name=model_name[0:-3])
#        model_given = deepcopy(model)
        model.save('INTIAL_MODEL.h5')

        print("This For loop has to start with 0 and ending as well")
        for n in range(0,self.k):
            self.avoid_sub_optimal_loop = False#one any of the highest accuracy limit pass the limitations this will make supoptimal avoid since already wanted accuracy is persued
            os.chdir('/')
            os.chdir(self.main_dir)
            #TO GET THE MODEL
            model_given = keras.models.load_model('INTIAL_MODEL.h5')
            print('')
            print(model_name,"'s",str(n),' th fold training is started in ',self.k, ' folds')
            print('')
            '''This should be run to create the data for the folds'''
            k_fold_training_generator.k_fold_list_formation(n)
            k_fold_training_generator.train_test_data(n)
            self.k_fold_training_generator = k_fold_training_generator#to get the object while training and validating
            self.model_parameter_train_k(model_given,model_name,n)
            print('')
            print(model_name,"'s",str(n), ' th fold training is done in ',self.k, ' folds')
            print('')
        
    def model_parameter_train_n_fold_given(self,model,model_name,n):
        
        self.main_dir =  ''.join([self.main_dir_part,'/',model_name[0:-3]])
        os.chdir('/')
        os.chdir(self.main_dir_part)
        '''To load the train dataset'''
        train_labels = pickle.load(open("train_labels_dic.p", "rb"))
        '''To create the dataset'''
        k_fold_training_generator = DataGenerator_k_fold_given(train_labels,self.main_dir,batch_size=self.load_train_size,n_channels=self.channels,Full_clean=self.Full_clean,balance_batch_must=self.balance_batch_must,shuffle=self.shuffle,k=self.k,model_name=model_name[0:-3])
        self.avoid_sub_optimal_loop = False#one any of the highest accuracy limit pass the limitations this will make supoptimal avoid since already wanted accuracy is persued
        os.chdir('/')
        os.chdir(self.main_dir)
        #TO GET THE MODEL
        model_given = keras.models.load_model('INTIAL_MODEL.h5')
        print('')
        print(model_name,"'s",str(n),' th fold training is started in ',self.k, ' folds')
        print('')
        '''This should be run to create the data for the folds'''
        k_fold_training_generator.k_fold_list_formation(n)
        k_fold_training_generator.train_test_data(n)
        self.k_fold_training_generator = k_fold_training_generator#to get the object while training and validating
        self.model_parameter_train_k(model_given,model_name,n)
        print('')
        print(model_name,"'s",str(n), ' th fold training is done in ',self.k, ' folds')
        print('')
                        
    def model_parameter_train_k(self,model,model_name,n):
        
        k_fold_training_generator = self.k_fold_training_generator
        self.n = n

        train_data_dir= ''.join([self.main_dir_part,'/Train'])
        number_of_overall_iertations_to_the_model=  self.number_of_overall_iertations_to_the_model      
        total_iteration = self.total_iteration   
        '''train the model
        
        If Adam optimizer or RMSprop: it take care of the leraning rate after it intialised
        
        '''
        if self.model_loss=="MSE":
            if self.optimizer=='RMSprop':
                model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=['accuracy'])
            elif self.optimizer=='Adam':
                 model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.Adagrad(),
                      metrics=['accuracy'])
        elif  self.model_loss == 'msle':
            if self.optimizer=='RMSprop':
                model.compile(loss='mean_squared_logarithmic_error',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
            elif self.optimizer=='Adam':
                 model.compile(loss='mean_squared_logarithmic_error',
                      optimizer=keras.optimizers.Adagrad(),
                      metrics=['accuracy'])
        elif self.model_loss == 'categorical_crossentropy':
            if self.optimizer=='RMSprop':
                model.compile(loss='categorical_crossentropy',
                              optimizer=keras.optimizers.RMSprop(),
                              metrics=['accuracy'])
            elif self.optimizer=='Adam':
                 model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adagrad(),
                      metrics=['accuracy'])    
        elif self.model_loss == 'sparse_categorical_crossentropy' and self.optimizer=='RMSprop':
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=keras.optimizers.RMSprop(),
                          metrics=['accuracy'])
        elif self.model_loss =='categorical_hinge':
           if self.optimizer=='Adagrad':
              model.compile(loss='categorical_hinge',
                          optimizer=keras.optimizers.Adagrad(),
                          metrics=['accuracy'])
           elif self.optimizer=='Adadelta':
              model.compile(loss='categorical_hinge',
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
        '''To break this while if it auceed the limiattions provided'''
        #temperory variables
        total_sub_optimal_loop=0
        self.highest_valid_accuracy=0
        while number_of_overall_iertations_to_the_model>0:
            print('')
            if total_sub_optimal_loop>5:
                print('The mdel exceed the suboptimal limitations')
                break# while loop Session break

            '''To check the validation accuracy stuck in suboptimal'''
            self.validation_low_count=0#if the validation accuracy less than 50 for more than 15 iterations continiously 
                                    #check the validation and flush it as new iteration
            self.validation_count=0
            self.validation_accuracy_last=0
            self.validation_accuracy_last_1=-1#to make both different
            self.validation_count_loop=0
    
            print('')
            print('')
            print('')
            print('')
            print('*****************************************---------------------->>>>>>>>>>>>>>>>>>>>>>>>>>> ******************')
            print('')
            print('Remaining iterations ', number_of_overall_iertations_to_the_model, ' in model ', model_name)
            print('')
            print('loss function: ',self.model_loss)
            print('')
            print('Optimiser function: ',self.optimizer)
            print('')
            print('Highest accuracy upto now sessions in validation set in fold ',str(self.n),'th Accuracy: ',self.highest_valid_accuracy)
            print('')
#            training_generator = DataGenerator_splited(train_labels,self.main_dir,batch_size=self.batch_size,n_channels=self.channels,Full_clean=self.Full_clean,balance_batch_must=self.balance_batch_must,validation_split=self.validation_split,shuffle=self.shuffle)

            if number_of_overall_iertations_to_the_model == (self.number_of_overall_iertations_to_the_model-1):
                # to avoid unnecessary reinitialisation in the beginning
                model.compile(loss='categorical_hinge',optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])    
                print('Reintialised with different ---Adadelta--- Optimiser and ---categorical_hinge--- loss function')
            elif number_of_overall_iertations_to_the_model == (self.number_of_overall_iertations_to_the_model-2):
                 # to avoid unnecessary reinitialisation in the beginning
                 model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adagrad(),metrics=['accuracy'])              
                 print('Reintialised with different ---Adagrad--- optimiser and ---mean_squared_error--- loss function')
            overall_history_log=[]
            validation_history_log=[]
            
            for itera_train in range(0,total_iteration):
                gc.collect()
            #    print(history.history.keys())
            #    print('validation_accuracy: ',history.history['val_acc'][-1])
                train_correct=0
                session_break=False
                overall_history_log_iter_temp=[]
#                for fetch in range(0,-(-training_generator.fetch_size()//self.batch_size)):
                for fetch in range(0,-(-k_fold_training_generator.fetch_size()//self.load_train_size)):
                    os.chdir('/')
                    os.chdir(train_data_dir)
                    # Train model on dataset
                    x_train, y_train = k_fold_training_generator.__getitem__(fetch)
    #                train_ids_temp = training_generator.__getitem_list_id(itera)
                    if self.batch_size<32:
                        history = model.fit(x_train, y_train,batch_size=self.batch_size,epochs=1)#,validation_split=0.2)
                        train_correct =train_correct+ history.history['acc'][-1]*self.load_train_size
#                        print(history.history.keys())
#                        if itera_train==0:
#                            if fetch==1:
#                                print('History accuarcys look like')
#                                print(history.history['acc'])
                    else:
                        history = model.fit(x_train, y_train,batch_size=self.batch_size,epochs=1)#,validation_split=0.2)
#                        history = model.fit(x_train, y_train,batch_size=16,epochs=1)#,validation_split=0.2)
                        train_correct =train_correct+ history.history['acc'][-1]*self.load_train_size
                    overall_history_log_iter_temp.append(deepcopy(history.history))
                    train_acc=(train_correct/((fetch+1)*self.load_train_size))
                    if train_acc>0.91 and fetch%4==0 and itera_train>9:
                        session_break,while_break_suboptimal,while_break_correct,validation_history_log = self.overall_model_per_chk(train_acc,itera_train,validation_history_log,model,model_name)
                        if session_break:
                            overall_history_log.append(deepcopy(overall_history_log_iter_temp))
                            break
                        
                overall_history_log.append(deepcopy(overall_history_log_iter_temp))
                if session_break:
                    break
                else:
                    train_acc = train_correct/k_fold_training_generator.fetch_size()
                    print("")
                    print("Overall train iteration of ",itera_train,"'s accuracy is ",train_acc*100,' %')
                    print("")

                    session_break,while_break_suboptimal,while_break_correct,validation_history_log = self.overall_model_per_chk(train_acc,itera_train,validation_history_log,model,model_name)
                if session_break:
                    break              
                
            '''To break the overall while'''
            if while_break_correct:
                print('Model suceed the limitations validatin set get higher than 85%  :)')
                break
            
            elif while_break_suboptimal:
                overall_history_log=[]
                validation_history_log=[]
                if self.avoid_sub_optimal_loop:
                    #since atleast satisfied one time the limitations provided
                    print("Atlease a model satisfied the given condition in one of the iteration")

                    number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model-1
                else:
                    total_sub_optimal_loop=total_sub_optimal_loop+1
                    print("since stuck in suboptimal doing the iteration agian")

            else:
                print("Session redo")
                number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model-1
                overall_history_log=[]
                validation_history_log=[]
        print('Highest_valid_accuracy: ',self.highest_valid_accuracy)
        if not (while_break_correct):
            gc.collect()
        del model
     
        if self.highest_valid_accuracy >= self.validation_acc_chk_limit:
            pickle.dump(overall_history_log, open(''.join(["overall_history_log_",str(self.n),".p"]), "wb"))  
            print('')
            print(str(self.sleep_time),' Sec break :) satisfied ')
            sleep(self.sleep_time)

            print('')
        else:
            print('')
            print(str(self.n),"th fold hasn't satisfied by the model")
            print(str(self.sleep_time),' Sec break ')
            sleep(self.sleep_time)

            print('')
        


    def shuffle_weights(self,model, weights=None):
        """Randomly permute the weights in `model`, or the given `weights`.
        This is a fast approximation of re-initializing the weights of a model.
        Assumes weights are distributed independently of the dimensions of the weight tensors
          (i.e., the weights have the same distribution along each dimension).
        :param Model model: Modify the weights of the given model.
        :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
          If `None`, permute the model's current weights.
        """
        if weights is None:
            weights = model.get_weights()
        
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        # Faster, but less random: only permutes along the first dimension
        # weights = [np.random.permutation(w) for w in weights]
        model.set_weights(weights)
        return model
   
    def create_train_accuracy_for_whole_set(self,model,train_data_dir,test_probabilities,pdbs_test):
        '''
        This function go through the whole training data set and produce the corresponding results
        '''
        os.chdir('/')
        os.chdir(self.main_dir)
        pickle.dump(test_probabilities, open(''.join(["valid_probabilities_fold_",str(self.n),".p"]), "wb"))  
        pickle.dump(pdbs_test, open(''.join(["pdbs_valid_fold_",str(self.n),".p"]), "wb"))  
        
        train_test_list_IDs = self.k_fold_training_generator.whole_train_data_ids
        train_test_labels= self.k_fold_training_generator.whole_train_data_labels
        
        os.chdir('/')
        os.chdir(train_data_dir)
        train_test_generator = Data_test(train_test_list_IDs, train_test_labels,batch_size=self.batch_size,n_channels=self.channels)

        c_train_test=0
        for itera in range(0,len(train_test_list_IDs)//self.batch_size):
            x_train_test, y_train_test = train_test_generator.getitem_test(itera)
            train_test_scores = model.evaluate(x_train_test, y_train_test, verbose=2)
            c_train_test=c_train_test+int(train_test_scores[1]*self.batch_size)
            
        list_IDs_temp=[]
        for itera in range((len(train_test_list_IDs)-len(train_test_list_IDs)%self.batch_size),len(train_test_list_IDs)):
            list_IDs_temp.append(train_test_list_IDs[itera])
        
        if len(list_IDs_temp)>0:
            x_train_test, y_train_test = train_test_generator.data_generation_test(list_IDs_temp)
            test_scores = model.evaluate( x_train_test, y_train_test, verbose=2)
            
            c_train_test=c_train_test+int(test_scores[1]*(len(train_test_list_IDs)%self.batch_size))
        
        os.chdir('/')
        os.chdir(self.main_dir)
        print("Corresponding_training accuracy: ",100*c_train_test/len(train_test_list_IDs)," %")
        os.chdir('/')
        os.chdir(train_data_dir)
        
        
    def validating_per_chk(self,valid_list_IDs,valid_generator,model,validation_history_log,itera_train):
        '''validating performance check'''
        train_data_dir= ''.join([self.main_dir_part,'/Train'])
        os.chdir('/')
        os.chdir(train_data_dir)
        valid_may_sat=False

        valid_correct=0
        valid_correct_combo=[]
        
        break_valid=False
        valid_chk_rest_TSG_OK=True
        valid_chk_rest_ONGO_OK=True
        valid_chk_rest_Fusion_OK=True
        

        for itera in range(0,len(valid_list_IDs)//self.batch_size):
            x_valid, y_valid = valid_generator.getitem_test(itera)
        
            valid_scores = model.evaluate(x_valid, y_valid, verbose=2)
            print("Valid_iter: ",itera," Validation acc: ",valid_scores[1])
            valid_correct=valid_correct+int(valid_scores[1]*self.batch_size)
            valid_acc=valid_correct/((itera+1)*self.batch_size)
            break_valid,valid_chk_rest_ONGO_OK,valid_chk_rest_TSG_OK,valid_chk_rest_Fusion_OK,valid_acc= self.validation_break_chk(valid_scores,itera,valid_correct,valid_acc,break_valid,valid_chk_rest_TSG_OK,valid_chk_rest_ONGO_OK,valid_chk_rest_Fusion_OK)
            valid_correct_combo.append(valid_scores[1])     
            if break_valid:
                break
            
        if valid_chk_rest_ONGO_OK and valid_chk_rest_TSG_OK and valid_chk_rest_Fusion_OK:
            valid_ids_temp=[]
            for itera in range((len(valid_list_IDs)-len(valid_list_IDs)%self.batch_size),len(valid_list_IDs)):
                valid_ids_temp.append(valid_list_IDs[itera])
            if len(valid_ids_temp)>0:
                x_valid, y_valid = valid_generator.data_generation_test(valid_ids_temp)
                valid_scores = model.evaluate(x_valid, y_valid, verbose=2)
                print("Valid_iter final Validation acc: ",valid_scores[1])
    
                valid_correct=valid_correct+int(valid_scores[1]*(len(valid_list_IDs)%self.batch_size))
           
            valid_acc=valid_correct/len(valid_list_IDs)
            print('In iteration ',itera_train)
            print('Validation accuracy : ', 100*valid_acc, '%')

            validation_history_log.append([itera_train,valid_acc])
            if valid_acc>self.train_acc_limit:
                print('Validation accuracy:', 100*valid_acc, '%')
            test_OG_chk=False
            test_TSG_chk=False
            
            
            os.chdir('/')
            os.chdir(self.main_dir)
            
            lengths_summery_dic = self.k_fold_training_generator.lengths_summery_dic
            valid_OG_len = lengths_summery_dic['valid_OG_len']
            valid_TSG_len = lengths_summery_dic['valid_TSG_len']
            
            os.chdir('/')
            os.chdir(train_data_dir)
            for val_acc_i in range(0,len(valid_correct_combo)):
                if val_acc_i<valid_OG_len//self.batch_size:
                    if valid_correct_combo[val_acc_i]>= self.validation_acc_OG_TSG_limit:
                        test_OG_chk=True
                if valid_OG_len//self.batch_size <val_acc_i< (valid_OG_len+valid_TSG_len)//self.batch_size:
                    if valid_correct_combo[val_acc_i]>= self.validation_acc_OG_TSG_limit:
                        test_TSG_chk=True
            if test_OG_chk and test_TSG_chk:
                if self.valid_partial_sat or valid_acc >= self.validation_acc_chk_limit:
                    #this skips the validation accuracy condition 
                    valid_may_sat=True
        return valid_acc,valid_may_sat,validation_history_log
    
    def overall_model_per_chk(self,train_acc,itera_train,validation_history_log,model,model_name):
        '''
        check the overall peroformance of the model
        '''
        '''Initialising validation steps'''
        validation_low_count=self.validation_low_count
        validation_count=self.validation_count
        validation_accuracy_last=self.validation_accuracy_last
        validation_count_loop=self.validation_count_loop
        validation_accuracy_last_1=self.validation_accuracy_last_1
        
        train_data_dir= ''.join([self.main_dir_part,'/Train'])
        
        while_break_suboptimal=False
        while_break_correct=False
        session_break=False
        os.chdir('/')
        os.chdir(self.main_dir)

        valid_labels = self.k_fold_training_generator.valid_labels_dic
        valid_list_IDs =  self.k_fold_training_generator.valid_ids
        valid_generator = Data_test(valid_list_IDs, valid_labels,batch_size=self.batch_size,n_channels=self.channels)           
        valid_acc,valid_may_sat,validation_history_log = self.validating_per_chk(valid_list_IDs,valid_generator,model,validation_history_log,itera_train)
    
        if (train_acc>self.train_acc_limit or valid_acc>= self.validation_acc_chk_limit):
            '''To check it reaches over all optimal  Given limit validation 71%'''
            if (train_acc>self.train_acc_limit and itera_train>46):
                validation_cond_low,validation_low_count= self.validation_accuracy_reached_low_optimal(valid_acc,validation_accuracy_last,validation_low_count)
                validation_cond,validation_count =  self.validation_accuracy_reached_optimal(valid_acc,validation_accuracy_last,validation_count)
            else:
                validation_cond_low=False
                validation_cond=False

            validation_cond_loop,validation_count_loop =  self.validation_accuracy_reached_optimal_loop(valid_acc,validation_accuracy_last_1,validation_accuracy_last,validation_count_loop)
            validation_accuracy_last_1=deepcopy(validation_accuracy_last)
            validation_accuracy_last=deepcopy(valid_acc)

            if validation_cond:
                '''
                if accuracy above 70% only considered
                    else reset the conditioin
                '''
                if valid_acc<self.train_acc_limit:
                    validation_count=0
                    validation_cond=False
                else:
                    print("Validation accuracy optimal")
                
            if (((valid_acc>self.validation_acc_chk_limit) or (valid_may_sat) or (validation_cond)) or validation_cond_loop or validation_cond_low) and (itera_train+1)>self.minimum_iteration_validation_chk:
                os.chdir('/')
                os.chdir(train_data_dir)
                if self.highest_valid_accuracy<100*valid_acc:
                    self.highest_valid_accuracy=100*valid_acc 
                    #create the probabilities for the validation set
                    valid_probabilities,pdbs_valid  = self.validation_prob_chk(valid_list_IDs,valid_generator,model)
                    print(str(self.n), " th folds highest accuracy updated :)")
                    print("Validation accuracy high: ",valid_acc*100," %")
                    self.create_train_accuracy_for_whole_set(model,train_data_dir,valid_probabilities,pdbs_valid)
                    
                    pickle.dump(validation_history_log, open(''.join(["validation_history_log_",str(self.n),".p"]), "wb"))  
   
                    if valid_acc>= 0.85:
                        while_break_correct=True
                        session_break=True#to break the iteration For loop
                    if valid_acc > 0.81:
                        self.avoid_sub_optimal_loop = True
                        if self.SITE_MUST:                           
                            #'''For saving the suboptimal soultion like if the results obtained like 80%'''
                            model_name_81_part=''.join([model_name[0:-3],'_SITE_',self.model_loss,'_',self.optimizer,'_n_',str(self.n),'_th_fold_acc_',str(int(100*(valid_acc)))])
                        else:
                            model_name_81_part=''.join([model_name[0:-3],'_',self.model_loss,'_',self.optimizer,'_n_',str(self.n),'_th_fold_acc_',str(int(100*(valid_acc)))])
    
                        if self.Full_clean and self.balance_batch_must:
                            model_name_81=''.join([model_name_81_part,'_Full_clean_bal_bat.h5'])
                        elif self.Full_clean:
                            model_name_81=''.join([model_name_81_part,'_Full_clean.h5'])
                        elif self.balance_batch_must:
                            model_name_81=''.join([model_name_81_part,'_bal_bat.h5'])
                        else:
                            model_name_81=''.join([model_name_81_part,'.h5'])
                        os.chdir('/')
                        os.chdir(self.main_dir)
                        model.save(model_name_81)
                        sleep(self.sleep_time)
                    
                    elif valid_acc > self.validation_acc_chk_limit:
                        self.avoid_sub_optimal_loop = True
                        if self.SITE_MUST:       
                            #'''For saving the suboptimal soultion like if the results obtained like 80%'''
                            model_name_71_part=''.join([model_name[0:-3],'_SITE_',self.model_loss,'_',self.optimizer,'_n_',str(self.n),'_th_fold_acc_',str(int(100*(valid_acc)))])
                        else:
                            model_name_71_part=''.join([model_name[0:-3],'_',self.model_loss,'_',self.optimizer,'_n_',str(self.n),'_th_fold_acc_',str(int(100*(valid_acc)))])
                        if self.Full_clean and self.balance_batch_must:
                            model_name_71=''.join([model_name_71_part,'_Full_clean_bal_bat.h5'])
                        elif self.Full_clean:
                            model_name_71=''.join([model_name_71_part,'_Full_clean.h5'])
                        elif self.balance_batch_must:
                            model_name_71=''.join([model_name_71_part,'_bal_bat.h5'])
                        else:
                            model_name_71=''.join([model_name_71_part,'.h5'])
    
                        model.save(model_name_71)
                        sleep(self.sleep_time)
                    
                if validation_cond_loop and valid_acc < self.validation_acc_chk_limit:                       
                    while_break_suboptimal=True
                    session_break=True       
                else:
                    validation_cond_loop=False
                    validation_count_loop=0
        
        self.validation_low_count=validation_low_count
        self.validation_count=validation_count
        self.validation_accuracy_last=validation_accuracy_last
        self.validation_count_loop=validation_count_loop
        self.validation_accuracy_last_1=validation_accuracy_last_1
        
        gc.collect()
        return session_break,while_break_suboptimal,while_break_correct,validation_history_log
    
    def validation_prob_chk(self,valid_list_IDs,valid_generator,model):
        '''
                validating probability creation
        
        '''
        
        train_data_dir= ''.join([self.main_dir_part,'/Train'])
        os.chdir('/')
        os.chdir(train_data_dir)
       
        valid_probabilities=[]
        pdbs_valid=[]

        for itera in range(0,len(valid_list_IDs)//self.batch_size):
            x_valid, y_valid = valid_generator.getitem_test(itera)
            valid_ids_temp = valid_generator.__getitem_list_id__(itera)

            probabilities = model.predict(x_valid)
            valid_probabilities.append(deepcopy(probabilities))
            pdbs_valid.append(deepcopy(valid_ids_temp))
            
        valid_ids_temp=[]
        for itera in range((len(valid_list_IDs)-len(valid_list_IDs)%self.batch_size),len(valid_list_IDs)):
            valid_ids_temp.append(valid_list_IDs[itera])
            
        if len(valid_ids_temp)>0:
            x_valid, y_valid = valid_generator.data_generation_test(valid_ids_temp)

            probabilities = model.predict(x_valid)
            valid_probabilities.append(deepcopy(probabilities))
            pdbs_valid.append(deepcopy(valid_ids_temp))
            
        return  valid_probabilities,pdbs_valid
    
    
    def validation_break_chk(self,valid_scores,itera,valid_correct,valid_acc,break_valid,valid_chk_rest_TSG_OK,valid_chk_rest_ONGO_OK,valid_chk_rest_Fusion_OK):
        '''
        This function used for early stopping in validation evaluation
        '''
        os.chdir('/')
        os.chdir(self.main_dir)
        
        lengths_summery_dic = self.k_fold_training_generator.lengths_summery_dic
        
        valid_OG_len = lengths_summery_dic['valid_OG_len']
        valid_TSG_len = lengths_summery_dic['valid_TSG_len']
        valid_len = lengths_summery_dic['valid_len']
        
        if itera==0 and valid_scores[1]<0.1:
            valid_chk_rest_ONGO_OK=False
            valid_acc=0.33#since only fit with TSG
            print("Validation since Not fit with ONGO")
            break_valid=True
            
        elif itera==(-(-valid_OG_len//self.batch_size))+1 and valid_scores[1]<0.1:
            valid_chk_rest_TSG_OK=False
            valid_acc=0.33#since only fit with ONGO
            print("Validation since NOt fit with TSG")
            break_valid=True
            
        elif itera== itera==(-(-(valid_OG_len+valid_TSG_len)//self.batch_size))+1 and valid_scores[1]<0.1:
#            valid_chk_rest_Fusion_OK=False#Not fit with Fusion at all
            valid_acc = valid_correct/valid_len
#            print("Validation since Not fit with Fusion")
#            break_valid=True
            print("Warning Fusion is not checked in Validation")
        train_data_dir= ''.join([self.main_dir_part,'/Train'])
        os.chdir('/')
        os.chdir(train_data_dir)
        
        return break_valid,valid_chk_rest_ONGO_OK,valid_chk_rest_TSG_OK,valid_chk_rest_Fusion_OK,valid_acc