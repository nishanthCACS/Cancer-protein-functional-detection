#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 17:31:12 2019

@author: %A.Nishanth C00294860
"""
#%%
#import tensorflow as tf
from data_generation_valid_vin2 import DataGenerator_splited 
from data_generation_valid_vin2 import Data_test


from tensorflow import keras
from copy import deepcopy
import numpy as np
import os
import pickle

from time import sleep# to make the GPU cool down
import gc

class ovear_all_training:
    def __init__(self,channels,test_accuracy=82,total_iteration=55,number_of_overall_iertations_to_the_model=10,validation_steps_check=5,minimum_iteration_validation_chk=1,sleep_time=30, model_loss='MSE', optimize='Adam',Full_clean=True,balance_batch_must=True,validation_split=0.15,shuffle=False,threshold=True):
        '''Initialization
        channels        : the depth of the input
        test_accuracy   : the wanted minimum test accuracy(since I obtained 86% in earlier model accidently/ but training accuracy is 85.--)
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

      validation_split: gives the validation data seperately
      '''
        self.model_loss=model_loss#'MSE'#"mean squared error"
        self.optimizer=optimize#'Adam'

        self.channels = channels
        print('channel initialised: ',self.channels)

        self.sleep_time = sleep_time
		
        #the parameters for data split
        self.validation_split = validation_split#0.2 
        self.shuffle = shuffle
        self.balance_batch_must=balance_batch_must
        self.Full_clean = Full_clean
        
        self.test_accuracy=test_accuracy #82 is default the wanted minimum test accuracy
        self.total_iteration=total_iteration #earlier v102
        self.number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model
        self.validation_steps_check = validation_steps_check#earliet its 2
        self.minimum_iteration_validation_chk=minimum_iteration_validation_chk
        
        self.validation_acc_chk_limit = 0.595
        self.validation_acc_OG_TSG_limit = 0.712#0.81
        self.train_acc_limit=0.82#to check the validation if it cross the training limitations 
        self.validation_low_steps_check=10
        self.fusion_test_chk=0.2
        self.epochs =55#howmany timne the same data used for training/ or feed through the network forward and backward
        if channels==21:
            self.main_dir=''.join(['scratch/optimaly_tilted_21'])
            if threshold:
                '''Different threshold condition for surface define'''
                self.main_dir =''.join(['scratch/21_aminoacids__thrsh_new_SITE_MUST'])
        elif channels==20:
            self.main_dir=''.join(['scratch/neg_size_20_21_aminoacids'])
        elif channels==17:
            self.main_dir=''.join(['scratch/optimaly_tilted_17'])
        else:
            raise("Define channel size directory to 15")
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
        # print('validation_accuracy_new: ',validation_accuracy_new)
        # print('validation_accuracy_last_1: ',validation_accuracy_last_1)
        # print('validation_accuracy_last: ',validation_accuracy_last)
        # print('validation_count_loop: ',validation_count_loop)
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
       
        number_of_overall_iertations_to_the_model=  self.number_of_overall_iertations_to_the_model
        
        total_iteration = self.total_iteration
        minimum_iteration_validation_chk=self.minimum_iteration_validation_chk      
    
        train_data_dir= ''.join([self.main_dir,'/Train'])
        test_data_dir= ''.join([self.main_dir,'/Test'])     

        os.chdir('/')
        os.chdir(self.main_dir)
        
        '''To load the train dataset'''
        train_labels = pickle.load(open("train_labels_dic.p", "rb"))
        train_list_ids=pickle.load(open("train_list_ids.p", "rb"))
        '''To load the test dataset'''
        test_list_IDs = pickle.load(open("test_list_ids.p", "rb"))
        test_labels = pickle.load(open("test_labels_dic.p", "rb"))
        test_generator = Data_test(test_list_IDs, test_labels,n_channels=self.channels)


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
        highest_test_accuracy=0
        while number_of_overall_iertations_to_the_model>0:
            
            print('')
            if total_sub_optimal_loop>5:
                print('The mdel exceed the suboptimal limitations')
                break# while loop Session break

            '''To check the validation accuracy stuck in suboptimal'''
            self.validation_low_count=0#if the validation accuracy less than 50 for more than 15 iterations continiously 
                                    #check the test and flush it as new iteration
            correct=0 
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
            print('Highest accuracy upto now sessions in test set: ',highest_test_accuracy)
            print('')
            training_generator = DataGenerator_splited(train_labels,self.main_dir,n_channels=self.channels,iter_train=self.total_iteration,Full_clean=self.Full_clean,balance_batch_must=self.balance_batch_must,validation_split=self.validation_split,shuffle=self.shuffle)
            

            if number_of_overall_iertations_to_the_model != self.number_of_overall_iertations_to_the_model:
                # to avoid unnecessary reinitialisation in the beginning
                model = self.shuffle_weights(model)
                print('Reintialised :)')
                

            overall_history_log=[]
            validation_history_log=[]
            train_low_count=0
            
            for itera_train in range(0,total_iteration):
                gc.collect()

                os.chdir('/')
                os.chdir(train_data_dir)
                # Train model on dataset
                print('creating itera_train: ', itera_train, 'train set')
                x_train, y_train = training_generator.__getitem__(itera_train)
#                train_ids_temp = training_generator.__getitem_list_id(itera)
                print('itera_train: ', itera_train, 'train set created')
                '''
                In training almost 3% of the data is misslabeled due to in Fusion and ONGO 
                or Fusion and TSG, sice the data set is low for Fusion higher priority is given to Fusion class
                
                Calling the model. fit method for a second time is not going to reinitialize our 
                already trained weights, which means we can actually make
                consecutive calls to fit if we want to and then manage it properly.
                
                On https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/ on 11-21-2019 at 12.46pm    
                '''
                if minimum_iteration_validation_chk==-1:
                    '''For test run the code or deep learning archtecture'''
                    history = model.fit(x_train, y_train,batch_size=16,epochs=1,validation_split=0.2)
                    os.chdir('/')
                    os.chdir(test_data_dir)
                    x_test, y_test = test_generator.getitem_test(itera_train)
                    test_ids_temp = test_generator.__getitem_list_id__(itera_train)
                    print('checking only so test skipped')
                    print('-----------------------------')
                    print('checking only so test skipped')
                    print('-----------------------------')
                    print('-- *********************   --')
                    os.chdir('/')
                    os.chdir(train_data_dir)
                    while_break_suboptimal=False
                    correct=0 
                    break#to break the for loop
                
            #    print(history.history.keys())
            #    print('validation_accuracy: ',history.history['val_acc'][-1])
                for fetch in range(0,self.epochs):
                    history = model.fit(x_train, y_train,batch_size=16,epochs=1)#,validation_split=0.2)
                    if (history.history['acc'][-1]>self.train_acc_limit):#since 32 data feeded
                        overall_history_log.append(history.history)
                        break#to break the for loop in epoaches if the data used in training reached its optimal


                '''check in each iterations'''
                session_break,while_break_suboptimal,while_break_correct = self.overall_model_per_chk(train_acc,itera_train,validation_history_log,model,model_name)

            '''To break the overall while'''
            if while_break_correct:
                print('Model suceed the limitations :)')
                model_name_latest_part=''.join([model_name[0:-3],'_',self.model_loss,'_',self.optimizer,'_',str(int(100*(correct/len(test_labels))))])
                if self.Full_clean and self.balance_batch_must:
                    model_name_latest=''.join([model_name_latest_part,'_Full_clean_bal_bat.h5'])
                elif self.Full_clean:
                    model_name_latest=''.join([model_name_latest_part,'_Full_clean.h5'])
                elif self.balance_batch_must:
                    model_name_latest=''.join([model_name_latest_part,'_bal_bat.h5'])
                else:
                    model_name_latest=''.join([model_name_latest_part,'.h5'])
             
                print("")                
                print("suceed model")
                print(model_name_latest)
                print("")
                model.save(model_name_latest)
                gc.collect()
                del model
                break
            
            elif while_break_suboptimal:
                print("since stuck in suboptimal doing the iteration agian")
                overall_history_log=[]
                validation_history_log=[]
                total_sub_optimal_loop=total_sub_optimal_loop+1
            else:
                print("Session redo")
                number_of_overall_iertations_to_the_model=number_of_overall_iertations_to_the_model-1
                overall_history_log=[]
                validation_history_log=[]
        print('Highest_test_accuracy: ',highest_test_accuracy)
        if not (while_break or while_break_correct):
            gc.collect()
            del model
     
        history_all=[validation_history_log,overall_history_log]    
        if 100*(correct/len(test_labels))>=self.test_accuracy:
            print('')
            print(str(self.sleep_time),' Sec break :) satisfied ')
            sleep(self.sleep_time)
            print('')
            return highest_test_accuracy,history_all,False#return not_satisfied as false
        else:
            print('')
            print(str(self.sleep_time),' Sec break ')
            sleep(self.sleep_time)
            print('')
            return highest_test_accuracy,history_all,True##return not_satisfied as True
        


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
        pickle.dump(test_probabilities, open("test_probabilities.p", "wb"))  
        pickle.dump(pdbs_test, open("pdbs_test.p", "wb"))  
        
        train_test_list_IDs = pickle.load(open("train_list_ids.p.p", "rb"))
        train_test_labels= pickle.load(open("train_labels_dic.p", "rb"))
        
        os.chdir('/')
        os.chdir(train_data_dir)
        train_test_generator = Data_test(train_test_list_IDs, train_test_labels,n_channels=self.channels)
        train_test_probabilities=[]
        pdbs_train_test=[]
        c_train_test=0
        for itera in range(0,len(train_test_list_IDs)//32):
            x_train_test, y_train_test = train_test_generator.getitem_test(itera)
            test_ids_temp = train_test_generator.__getitem_list_id__(itera)
        
            train_test_scores = model.evaluate(x_train_test, y_train_test, verbose=2)
            probabilities = model.predict(x_train_test)
           
            train_test_probabilities.append(deepcopy(probabilities))
            pdbs_train_test.append(deepcopy(test_ids_temp))
            c_train_test=c_train_test+int(train_test_scores[1]*32)
            
        list_IDs_temp=[]
        for itera in range((len(train_test_list_IDs)-len(train_test_list_IDs)%32),len(train_test_list_IDs)):
            list_IDs_temp.append(train_test_list_IDs[itera])
        
        x_test, y_test = train_test_generator.data_generation_test(list_IDs_temp)
        test_scores = model.evaluate(x_test, y_test, verbose=2)
        
        c_train_test=c_train_test+int(test_scores[1]*(len(train_test_list_IDs)%32))
        probabilities = model.predict(x_test)
        train_test_probabilities.append(deepcopy(probabilities))
        pdbs_train_test.append(deepcopy(list_IDs_temp))
        
        os.chdir('/')
        os.chdir(self.main_dir)
        pickle.dump(train_test_probabilities, open("train_test_probabilities.p", "wb"))  
        pickle.dump(pdbs_train_test, open("pdbs_train_test.p", "wb"))
        print("Corresponding_training accuracy: ",100*c_train_test/len(train_test_list_IDs)," %")
        os.chdir('/')
        os.chdir(train_data_dir)
        
        
    def validating_per_chk(self,valid_list_IDs,valid_generator,model,validation_history_log,itera_train):
        '''validating performance check'''
        valid_may_sat=False
        valid_chk_rest_TSG_OK=True
        valid_chk_rest_ONGO_OK=True
        valid_chk_rest_Fusion_OK=True
        valid_correct=0
        valid_correct_combo=[]
        for itera in range(0,len(valid_list_IDs)//32):
            x_valid, y_valid = valid_generator.getitem_test(itera)
        
            valid_scores = model.evaluate(x_valid, y_valid, verbose=2)
            valid_correct=valid_correct+int(valid_scores[1]*32)
            if itera==0 and valid_scores[1]<0.1:
                valid_chk_rest_ONGO_OK=False
                valid_acc=0.4#since only fit with TSG
                break
            elif itera==4 and valid_scores[1]<0.1:
                valid_chk_rest_TSG_OK=False
                valid_acc=0.375#since only fit with ONGO
                break
            elif itera==8 and valid_scores[1]<0.1:
                valid_chk_rest_Fusion_OK=False#Not fit with Fusion at all
                valid_acc = valid_correct/len(valid_list_IDs)
                break
            valid_correct_combo.append(valid_scores[1])     
            
        if valid_chk_rest_ONGO_OK and valid_chk_rest_TSG_OK and valid_chk_rest_Fusion_OK:
            valid_ids_temp=[]
            for itera in range((len(valid_list_IDs)-len(valid_list_IDs)%32),len(valid_list_IDs)):
                valid_ids_temp.append(valid_list_IDs[itera])
            
            x_valid, y_valid = valid_generator.data_generation_test(valid_ids_temp)
            valid_scores = model.evaluate(x_valid, y_valid, verbose=2)
            
            valid_correct=valid_correct+int(valid_scores[1]*(len(valid_list_IDs)%32))
            valid_acc=valid_correct/len(valid_list_IDs)
            print('Validation accuracy in iteration ',itera_train,': ', 100*valid_acc, '%')
            validation_history_log.append([itera_train,valid_acc])
            if valid_acc>self.train_acc_limit:
                print('Validation accuracy:', 100*valid_acc, '%')
            test_OG_chk=False
            test_TSG_chk=False
            for val_acc_i in range(0,len(valid_correct_combo)):
                if val_acc_i<3:
                    if valid_correct_combo[val_acc_i]>= self.validation_acc_OG_TSG_limit:
                        test_OG_chk=True
                if 3<val_acc_i<7:
                    if valid_correct_combo[val_acc_i]>= self.validation_acc_OG_TSG_limit:
                        test_TSG_chk=True
            if test_OG_chk and test_TSG_chk and valid_acc >= self.validation_acc_chk_limit:
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
        
        train_data_dir= ''.join([self.main_dir,'/Train'])
        test_data_dir= ''.join([self.main_dir,'/Test'])     
        
        while_break_suboptimal=False
        while_break_correct=False
        session_break=False
        
        if self.validation_split>0:
            valid_labels = pickle.load(open("valid_labels_dic.p","rb"))
            valid_list_IDs = pickle.load(open("valid_ids.p", "rb"))
            valid_generator = Data_test(valid_list_IDs, valid_labels,n_channels=self.channels)         
            valid_acc,valid_may_sat,validation_history_log = self.validating_per_chk(valid_list_IDs,valid_generator,model,validation_history_log,itera_train)
        
        if (train_acc>self.train_acc_limit) or (valid_acc>(self.test_accuracy/100) or valid_acc>= self.validation_acc_chk_limit):
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
                
            if (((valid_acc>(self.test_accuracy/100)) or (valid_may_sat) or (validation_cond)) or validation_cond_loop or validation_cond_low) and (itera_train+1)>self.minimum_iteration_validation_chk:
               
                os.chdir('/')
                os.chdir(self.main_dir)
                
                '''To load the test dataset'''
                test_list_IDs = pickle.load(open("test_list_ids.p", "rb"))
                test_labels = pickle.load(open("test_labels_dic.p", "rb"))
                test_generator = Data_test(test_list_IDs, test_labels,n_channels=self.channels)

                print(' ')
                print("Since epach size is 1 one PDB used once per training ")
                print(' ')
                os.chdir('/')
                os.chdir(test_data_dir)
                #% testring
                correct=0
                list_IDs_temp=[]
                '''check the fusion class'''
                for itera in range(len(test_labels)-18,len(test_labels)):
                    list_IDs_temp.append(test_list_IDs[itera])
                x_test, y_test = test_generator.data_generation_test(list_IDs_temp)
                test_scores = model.evaluate(x_test, y_test, verbose=2)
                Fusion_test_acc = test_scores[1]
        #            print('Test loss    :', test_scores[0])
                print('Fusion test accuracy:', 100*Fusion_test_acc, '%')
                itera_chk=False #if TSG and Fusion not satisfied 
                test_acc_tsg=0
                if validation_cond_low and test_scores[1]<self.fusion_test_chk:
                    print("")
                    print("Stuck in Local Minimum")
                    print("Starting new iteration in Same **Session")
                    print("")
                    os.chdir('/')
                    os.chdir(train_data_dir)
                    session_break=True
                elif validation_cond_low:
                    list_IDs_temp=[]
                    '''check the TSG class'''
                    for itera in range(len(test_labels)-50,len(test_labels)-18):
                        list_IDs_temp.append(test_list_IDs[itera])
                    x_test, y_test = test_generator.data_generation_test(list_IDs_temp)
                    test_scores = model.evaluate(x_test, y_test, verbose=2)
                    list_IDs_temp_2=[]
                    for itera in range(len(test_labels)-82,len(test_labels)-50):
                        list_IDs_temp_2.append(test_list_IDs[itera])
                    x_test, y_test = test_generator.data_generation_test(list_IDs_temp_2)
                    test_scores_2 = model.evaluate(x_test, y_test, verbose=2)					 
                    test_acc_tsg=(test_scores[1]+test_scores_2[1])/2
                    print('TSG test accuracy:', 100*test_acc_tsg, '%')
                    if validation_cond_low and test_acc_tsg<self.validation_acc_OG_TSG_limit:
                        print("")
                        print("Stuck in Local Minimum Due to TSG not satisfied")
                        print("Starting new iteration in Same **Session")
                        print("")
                        os.chdir('/')
                        os.chdir(train_data_dir)
                        session_break=True
                    
                
                if Fusion_test_acc>=self.fusion_test_chk and test_acc_tsg>=self.validation_acc_OG_TSG_limit:
                    test_probabilities=[]
                    pdbs_test=[]
                    correct=0
                    itera_chk=True#to avoid unnecessary evaluation in test set 
                    for itera in range(0,len(test_labels)//32):
                        x_test, y_test = test_generator.getitem_test(itera)
                        test_ids_temp = test_generator.__getitem_list_id__(itera)

                        test_scores = model.evaluate(x_test, y_test, verbose=2)
                        probabilities = model.predict(x_test)
                       
                        test_probabilities.append(deepcopy(probabilities))
                        pdbs_test.append(deepcopy(test_ids_temp))
                        correct=correct+int(test_scores[1]*32)
                else:
                    if test_acc_tsg<self.validation_acc_OG_TSG_limit:
                        print("TSG test set haven't got higher than " ,self.validation_acc_OG_TSG_limit*100 ,"% accuracy so no test check")
                        #to avoid unnecessary evaluation in test set 
                        print("Back to normal iteration")
                    else:
                        print("Fusion test set haven't got higher than " ,self.fusion_test_chk*100 ,"% accuracy so no test check")
                        #to avoid unnecessary evaluation in test set 
                        print("Back to normal iteration")
                    
                
                
                if itera_chk:#to avoid unnecessary evaluation in test set 
                    '''To check the last test batch'''
                    list_IDs_temp=[]
                    for itera in range((len(test_labels)-len(test_labels)%32),len(test_labels)):
                        list_IDs_temp.append(test_list_IDs[itera])
                    
                    x_test, y_test = test_generator.data_generation_test(list_IDs_temp)
                    test_scores = model.evaluate(x_test, y_test, verbose=2)

                    correct=correct+int(test_scores[1]*(len(test_labels)%32))
                    probabilities = model.predict(x_test)
                    test_probabilities.append(deepcopy(probabilities))
                    pdbs_test.append(deepcopy(list_IDs_temp))

                    print("")
                    print("Overall test accuracy: ",100*correct/len(test_labels))
                    os.chdir('/')
                    os.chdir(train_data_dir)
                    if self.highest_test_accuracy<100*(correct/len(test_labels)):
                        self.highest_test_accuracy=100*(correct/len(test_labels))           
                        self.create_train_accuracy_for_whole_set(model,train_data_dir,test_probabilities,pdbs_test)
                        
                    if (100*(correct/len(test_labels)))>=85 or ((100*(correct/len(test_labels)))>=self.test_accuracy and self.channels !=17):
                        while_break_correct=True
                        session_break=True#to break the iteration For loop
                    elif  ((100*(correct/len(test_labels)))> 81 and self.channels != 17):
                        #'''For saving the suboptimal soultion like if the results obtained like 80%'''
                        model_name_81_part=''.join([model_name[0:-3],'_',self.model_loss,'_',self.optimizer,'_',str(int(100*(correct/len(test_labels))))])
                        if self.Full_clean and self.balance_batch_must:
                            model_name_81=''.join([model_name_81_part,'_Full_clean_bal_bat.h5'])
                        elif self.Full_clean:
                            model_name_81=''.join([model_name_81_part,'_Full_clean.h5'])
                        elif self.balance_batch_must:
                            model_name_81=''.join([model_name_81_part,'_bal_bat.h5'])
                        else:
                            model_name_81=''.join([model_name_81_part,'.h5'])

                        model.save(model_name_81)
                        sleep(self.sleep_time)
                    
                    elif  ((100*(correct/len(test_labels)))> 71 and self.channels != 17):
                        #'''For saving the suboptimal soultion like if the results obtained like 80%'''
                        model_name_71_part=''.join([model_name[0:-3],'_',self.model_loss,'_',self.optimizer,'_',str(int(100*(correct/len(test_labels))))])
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
                        
                if validation_cond_loop and itera_chk and (100*(correct/len(test_labels)))<self.test_accuracy:
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
        return session_break,while_break_suboptimal,while_break_correct
    
