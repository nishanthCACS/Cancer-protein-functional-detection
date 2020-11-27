#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 12:15:12 2019

@author: c00294860
"""
import numpy as np
import keras
from copy import deepcopy
import pickle
import os
import random

class DataGenerator_splited:
    'Generates 2D projected data for Keras'
    def __init__(self, labels,main_directory, batch_size=10, dim = (128,128), n_channels=21 ,n_classes=3,iter_train=199, shuffle=True,Full_clean=True,balance_batch_must=True,validation_split=0.2):
        '''Initialization
        labels
        main_directory:
        batch_size=10
        dim = (200,200)
        n_channels=17 
        n_classes=3
        iter_train=199
        shuffle=True
        Full_clean=True
        balance_batch_must: True make sure all batches has same class distribution
                            False not forcing the all batches have same class distribution
        validation_split: Fraction split the validation data from the whole training set
                        Default value is 20%
                     **   Especially the TSG split is same as OG split size
                       **  and the Fusion split is 0.5 x Split persent to avoid models biased to OG
        '''
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.dim = dim
        self.Full_clean= Full_clean
        self.iter_train=iter_train
        self.labels=labels
        self.main_directory= main_directory
        self.balance_batch_must=balance_batch_must
        self.validation_split=validation_split
     
        os.chdir('/')
        os.chdir(main_directory)    
        if validation_split>0:
            #split the training and validation data
            #first load the all training data
            all_OG_train_PDBs = pickle.load(open("OG_train_PDBs.p", "rb")) 
            all_TSG_train_PDBs = pickle.load(open("TSG_train_PDBs.p", "rb"))      
            all_Fusion_train_PDBs = pickle.load(open("Fusion_train_PDBs.p", "rb"))  
            
            print('')
            print('Over all training data in:')
            print('OG: ',len(all_OG_train_PDBs))
            print('TSG: ',len(all_TSG_train_PDBs))        
            print('Fusion: ',len(all_Fusion_train_PDBs))    
            #to avoid validation class has overlapping classes
            self.overlapped_PDBs_ONGO = pickle.load(open("overlapped_PDBs_ONGO.p", "rb"))    
            self.overlapped_PDBs_TSG = pickle.load(open("overlapped_PDBs_TSG.p", "rb"))  
            # shuffle the data first
            chk_OG=deepcopy(list(range(len(all_OG_train_PDBs))))
            chk_TSG=deepcopy(list(range(len(all_TSG_train_PDBs))))
            chk_Fusion=deepcopy(list(range(len(all_Fusion_train_PDBs))))    
            if shuffle:
                random.shuffle(chk_OG)
                random.shuffle(chk_TSG)
                random.shuffle(chk_Fusion)    

            validation_OG = []
            over_chk_OG=0
            # print(' self.overlapped_PDBs_ONGO: ',len(self.overlapped_PDBs_ONGO))
            # print('all_OG_train_PDBs: ',len(all_OG_train_PDBs))
            # print('len chk_OG: ',len(chk_OG))
            OG_valid_split_size = int(validation_split*len(chk_OG))
            for i in range(0,OG_valid_split_size):
                # print(' all_OG_train_PDBs[chk_OG[i+over_chk_OG]] i: ',i,'over_chk_OG ',over_chk_OG)
                if all_OG_train_PDBs[chk_OG[i+over_chk_OG]] not in self.overlapped_PDBs_ONGO:
                    validation_OG.append(all_OG_train_PDBs.pop(chk_OG[i+over_chk_OG]))
                else:
                    while (all_OG_train_PDBs[chk_OG[i+over_chk_OG]] in self.overlapped_PDBs_ONGO):
                        over_chk_OG=over_chk_OG+1
                    validation_OG.append(all_OG_train_PDBs.pop(chk_OG[i+over_chk_OG]))         
                chk_OG = self.pop_range_fix(chk_OG,chk_OG[i+over_chk_OG])

            self.validation_OG=validation_OG
            self.OG_train_PDBs = all_OG_train_PDBs
            print('validation creation ONGO done :)')
            print('')
            print('Training data size OG: ',len(self.OG_train_PDBs))
            print('Validati data size OG: ',len(validation_OG))
            print('')

            validation_TSG=[]
            over_chk_TSG=0
            # print(int(validation_split*len(chk_TSG)))
            # print('overlapped_PDBs_TSG: ',len(self.overlapped_PDBs_TSG))
            for i in range(0,OG_valid_split_size):
                '''here the validation split is equal as OG split'''
                if all_TSG_train_PDBs[chk_TSG[i+over_chk_TSG]] not in self.overlapped_PDBs_TSG:
                    validation_TSG.append(all_TSG_train_PDBs.pop(chk_TSG[i+over_chk_TSG]))
                else:
                    while (all_TSG_train_PDBs[chk_TSG[i+over_chk_TSG]] in self.overlapped_PDBs_TSG):
                        over_chk_TSG=over_chk_TSG+1
                    validation_TSG.append(all_TSG_train_PDBs.pop(chk_TSG[i+over_chk_TSG]))   
                chk_TSG = self.pop_range_fix(chk_TSG,chk_TSG[i+over_chk_TSG])

            print('validation creation TSG done :)')
            print('')
      
            validation_Fusion = []
            over_chk_Fusion=0
            for i in range(0,int(validation_split*0.5*len(chk_Fusion))):
                if (all_Fusion_train_PDBs[chk_Fusion[i+over_chk_Fusion]] not in self.overlapped_PDBs_TSG) and (all_Fusion_train_PDBs[chk_Fusion[i+over_chk_Fusion]] not in self.overlapped_PDBs_ONGO):
                    validation_Fusion.append(all_Fusion_train_PDBs.pop(chk_Fusion[i+over_chk_Fusion]))    
                else:
                    while  (all_Fusion_train_PDBs[chk_Fusion[i+over_chk_Fusion]] in self.overlapped_PDBs_TSG) or (all_Fusion_train_PDBs[chk_Fusion[i+over_chk_Fusion]] in self.overlapped_PDBs_ONGO):
                        over_chk_Fusion = over_chk_Fusion + 1
                    validation_Fusion.append(all_Fusion_train_PDBs.pop(chk_Fusion[i+over_chk_Fusion]))    
                chk_Fusion = self.pop_range_fix(chk_Fusion,chk_Fusion[i+over_chk_Fusion])

            print('validation creation Fusion done :)')
            print('')

            self.validation_TSG=validation_TSG
            self.validation_Fusion=validation_Fusion
            
            self.TSG_train_PDBs = all_TSG_train_PDBs  
            self.Fusion_train_PDBs = all_Fusion_train_PDBs
            print('')
            print('-------------After vlidation split ------------')

            print('Training data size TSG: ',len(self.TSG_train_PDBs))
            print('Validati data size TSG: ',len(validation_TSG))
            print('Training data size Fusion: ',len(self.Fusion_train_PDBs))
            print('Validati data size Fusion: ',len(validation_Fusion))
            print('')
            print('')

            valid_labels_dic={}
            valid_ids=validation_OG+validation_TSG+validation_Fusion
            
            for i in range(0,len(validation_OG)):
                valid_labels_dic.update({validation_OG[i]: 0})
            for i in range(0,len(validation_TSG)):
                valid_labels_dic.update({validation_TSG[i]: 1})
            for i in range(0,len(validation_Fusion)):
                valid_labels_dic.update({validation_Fusion[i]: 2})

            pickle.dump(valid_labels_dic, open("valid_labels_dic.p", "wb"))  
            pickle.dump(valid_ids, open("valid_ids.p", "wb"))  
            lengths_summery_dic={}
            lengths_summery_dic.update({'valid_len':len(self.validation_OG)+ len(self.validation_TSG)+ len(self.validation_Fusion)})
            lengths_summery_dic.update({'train_len':len(self.OG_train_PDBs)+len(self.TSG_train_PDBs)+len(self.Fusion_train_PDBs)})
            lengths_summery_dic.update({'valid_OG_len':len(self.validation_OG)})
            lengths_summery_dic.update({'valid_TSG_len':len(self.validation_TSG)})
            lengths_summery_dic.update({'valid_Fusion_len':len(self.validation_Fusion)})
            pickle.dump(lengths_summery_dic, open("lengths_summery_dic.p", "wb")) 

            self.lengths_summery_dic = lengths_summery_dic
            self.iter_train=len(self.OG_train_PDBs)+len(self.TSG_train_PDBs)+len(self.Fusion_train_PDBs)#or whole train set size
            #according to this only the data is fetched
        else:
            self.OG_train_PDBs = pickle.load(open("OG_train_PDBs.p", "rb"))  
            self.TSG_train_PDBs = pickle.load(open("TSG_train_PDBs.p", "rb"))      
            self.Fusion_train_PDBs = pickle.load(open("Fusion_train_PDBs.p", "rb"))  
            lengths_summery_dic={}
            lengths_summery_dic.update({'train_len':len(self.OG_train_PDBs)+len(self.TSG_train_PDBs)+len(self.Fusion_train_PDBs)})
            lengths_summery_dic.update({'train_OG_len':len(self.OG_train_PDBs)})
            lengths_summery_dic.update({'train_TSG_len':len(self.TSG_train_PDBs)})
            lengths_summery_dic.update({'train_Fusion_len':len(self.Fusion_train_PDBs)})

            self.lengths_summery_dic = lengths_summery_dic
            self.iter_train=len(self.OG_train_PDBs)+len(self.TSG_train_PDBs)+len(self.Fusion_train_PDBs)#or whole train set size
            #according to this only the data is fetched
        if self.Full_clean:
           self.clean_list_of_ids= []
           self.clean_list_of_id_classes = []
        else:
           self.overlapped_PDBs_ONGO = pickle.load(open("overlapped_PDBs_ONGO.p", "rb"))    
           self.overlapped_PDBs_TSG = pickle.load(open("overlapped_PDBs_TSG.p", "rb"))  
           self.list_of_ids= []
           self.list_of_id_classes = []
       
        # thisindex_former create the randomly shuffled data
        self.index_former()

    def fetch_size(self):
        return self.iter_train

    def pop_range_fix(self,list_1,pop_element_num):
        '''
        To fixing the overlap remoaval for validation data creation
        '''
        list_2=deepcopy(list_1)
        for k in range(0,len(list_1)):
            if list_1[k]>pop_element_num:
                list_2[k]=list_2[k]-1
        return list_2
    
    
    def index_former(self):
        
        self.indexes = np.arange(self.iter_train)
        #to creat the indexes easy
        index_ongo=[]
        index_tsg=[]
        index_fusion=[]
        # since the Fusion indexes runing out early 
        ''' reuse the Fusion class again or combined class'''

        if self.Full_clean:
            if self.balance_batch_must:
                for i in range(0,(self.iter_train)//5): 
                    #choose 2 from ONGO 2 from TSG and 1 From Fusion to make a batch that distribute the all most all PDBs
                    #Fusion_chk=True
                    '''Since the Models performance always biased to ONGO'''
                    index_ongo,index_tsg,index_fusion=self.adding_fully_clean(True,index_ongo,index_tsg,index_fusion)
                    index_ongo,index_tsg,index_fusion=self.adding_fully_clean(False,index_ongo,index_tsg,index_fusion)
#                    index_ongo,index_tsg,index_fusion=self.adding_fully_clean(False,index_ongo,index_tsg,index_fusion,OG_IN=False)

                if (self.iter_train)%5>0: 
                    index_ongo,index_tsg,index_fusion=self.adding_fully_clean(False,index_ongo,index_tsg,index_fusion,OG_IN=False)
                    index_ongo,index_tsg,index_fusion=self.adding_fully_clean(True,index_ongo,index_tsg,index_fusion)

                os.chdir('/')
                os.chdir(self.main_directory)               
                if self.validation_split>0:
                    pickle.dump(self.clean_list_of_ids, open("V_clean_balnce_must_list_of_ids.p", "wb"))  
                    pickle.dump(self.clean_list_of_id_classes, open("V_clean_balnce_must_list_of_id_classes.p", "wb")) 
                else:
                    pickle.dump(self.clean_list_of_ids, open("clean_balnce_must_list_of_ids.p", "wb"))  
                    pickle.dump(self.clean_list_of_id_classes, open("clean_balnce_must_list_of_id_classes.p", "wb")) 
            else:
                self.not_balance_must_and_Fully_clean()
            
        else:
            if self.balance_batch_must:
                '''First create the indexes for overlapped PDBs'''
                index_ongo_fusion = self.index_stack_overlapped_chk(True,False)
                index_tsg_fusion = self.index_stack_overlapped_chk(False,True)
                for i in range(0,(self.iter_train)//5): 
                    #choose 2 from ONGO 2 from TSG and 1 From Fusion to make a batch that distribute the all most all PDBs
                    #Fusion_chk=True
                    index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion=self.adding(True,index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion)
                    index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion=self.adding(False,index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion)
#                    index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion=self.adding(False,index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion,OG_IN=False)

                if (self.iter_train)%5>0: 
                    index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion=self.adding(False,index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion,OG_IN=False)
                    index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion=self.adding(True,index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion)

                os.chdir('/')
                os.chdir(self.main_directory)               
                if self.validation_split>0:
                    pickle.dump(self.list_of_ids, open("V_list_of_ids.p", "wb"))  
                    pickle.dump(self.list_of_id_classes, open("V_list_of_id_classes.p", "wb")) 
                else:
                    pickle.dump(self.list_of_ids, open("list_of_ids.p", "wb"))  
                    pickle.dump(self.list_of_id_classes, open("list_of_id_classes.p", "wb")) 
            else:
                self.overalp_included_but_not_balance_must()
    
    def not_balance_must_and_Fully_clean(self):
        '''
        This function craete the IDS and corresponding lable by randomly shuffling the data
        
        Not balance data only shuffled once; thus the data order doesn't changed
       
        Since the TSG class higer number of PDBs remove some of the PDBs from TSG to keep the dataset unique in all sessions
        (using the same data as train and validation)
        ''' 
       
        raise("This function has to be reconstructed to avoid  remove PDBs randomly from the TSG group")
        
        
        clean_list_of_ids_temp=self.OG_train_PDBs+deepcopy(self.TSG_train_PDBs)+self.Fusion_train_PDBs#since removing some PDBs from TSG PDBs
        clean_list_of_ids=[]
        removed_list_tsg=[]
        if (len(clean_list_of_ids_temp)%self.batch_size)>0:
            chk=list(range(len(self.OG_train_PDBs),len(self.OG_train_PDBs)+len(self.TSG_train_PDBs)))
            random.shuffle(chk)#shuffle to remove PDBs randomly from the TSG group
            for i in range(0,len(clean_list_of_ids_temp)%self.batch_size):
                removed_list_tsg.append(clean_list_of_ids_temp.pop(chk[i]))
  
        if self.shuffle:
            random.shuffle(clean_list_of_ids_temp)
            
        for i in range(0,(self.batch_size*self.iter_train)//len(clean_list_of_ids_temp)):
            clean_list_of_ids=clean_list_of_ids + deepcopy(clean_list_of_ids_temp)

        clean_list_of_ids = clean_list_of_ids + deepcopy(clean_list_of_ids_temp[0:(self.batch_size*self.iter_train)%len(clean_list_of_ids_temp)])           
        self.clean_list_of_ids = deepcopy(clean_list_of_ids)
        
        print('')
        print("Not balanced Clean list_of_ids lenth: ",len(self.clean_list_of_ids))
        print('')
 
        os.chdir('/')
        os.chdir(self.main_directory)
        if self.validation_split>0:
            pickle.dump(removed_list_tsg, open("V_removed_list_tsg_unbalanced_clean.p", "wb"))  
            pickle.dump(self.clean_list_of_ids, open("V_clean_list_of_ids.p", "wb"))  
            pickle.dump(self.clean_list_of_id_classes, open("V_clean_list_of_id_classes.p", "wb")) 
        else:
            pickle.dump(removed_list_tsg, open("removed_list_tsg_unbalanced_clean.p", "wb"))  
            pickle.dump(self.clean_list_of_ids, open("clean_list_of_ids.p", "wb"))  
            pickle.dump(self.clean_list_of_id_classes, open("clean_list_of_id_classes.p", "wb"))   
            
    def overalp_included_but_not_balance_must(self):
        '''
        This function craete the IDS and corresponding lable by randomly shuffling the data
        '''
        raise("This function has to be reconstructed to avoid  remove PDBs randomly from the TSG group")

        
        import random 
        list_of_ids_temp=self.OG_train_PDBs+self.TSG_train_PDBs+self.Fusion_train_PDBs+self.overlapped_PDBs_ONGO+self.overlapped_PDBs_TSG
        
        removed_list_tsg=[]
        if (list_of_ids_temp%self.batch_size)>0:
            chk=list(range(len(self.OG_train_PDBs),len(self.OG_train_PDBs)+len(self.TSG_train_PDBs)))
            random.shuffle(chk)
            for i in range(0,list_of_ids_temp%self.batch_size):
                removed_list_tsg.append(list_of_ids_temp.pop(chk[i]))#the removal of TSG PDBs
        
        list_of_ids_classes_temp=[]
        for i in range(0,len(self.OG_train_PDBs)):
            list_of_ids_classes_temp.append(np.array([1,0,0],dtype=float))
        #the removal of TSG PDBs
        for i in range(0,(len(self.TSG_train_PDBs)-len(removed_list_tsg))):
            list_of_ids_classes_temp.append(np.array([0,1,0],dtype=float))       
        for i in range(0,len(self.Fusion_train_PDBs)):
            list_of_ids_classes_temp.append(np.array([0,0,1],dtype=float))   
        for i in range(0,len(self.overlapped_PDBs_ONGO)):
            list_of_ids_classes_temp.append(np.array([0.5,0,0.5]))   
        for i in range(0,len(self.overlapped_PDBs_TSG)):
            list_of_ids_classes_temp.append(np.array([0,0.5,0.5]))   
            
        list_of_ids=[]
        list_of_id_classes=[]
        
        if self.shuffle:
            mapIndexPosition = list(zip(list_of_ids_temp, list_of_ids_classes_temp))
            random.shuffle(mapIndexPosition)
            list_of_ids_temp, list_of_ids_classes_temp = zip(*mapIndexPosition)   

        for i in range(0,self.iter_train//len(list_of_ids_temp)):
            list_of_ids = list_of_ids + deepcopy(list_of_ids_temp)
            list_of_id_classes = list_of_id_classes + deepcopy(list_of_ids_classes_temp)
        
        list_of_ids = list_of_ids + deepcopy(list_of_ids_temp[0:self.iter_train%len(list_of_ids_classes_temp)])           
        list_of_id_classes = list_of_id_classes + deepcopy(list_of_ids_classes_temp[0:self.iter_train%len(list_of_ids_classes_temp)])
        
        self.list_of_ids = deepcopy(list_of_ids)
        self.list_of_id_classes=deepcopy(list_of_id_classes)
    
        os.chdir('/')
        os.chdir(self.main_directory)
        if self.validation_split>0:
            pickle.dump(self.list_of_ids, open("V_list_of_ids.p", "wb"))  
            pickle.dump(self.list_of_id_classes, open("V_list_of_id_classes.p", "wb"))
            pickle.dump(removed_list_tsg, open("V_removed_list_tsg_unbalanced.p", "wb"))  
        else:
            pickle.dump(self.list_of_ids, open("list_of_ids.p", "wb"))  
            pickle.dump(self.list_of_id_classes, open("list_of_id_classes.p", "wb"))
            pickle.dump(removed_list_tsg, open("removed_list_tsg_unbalanced.p", "wb"))  

    def adding(self,Fusion_chk,index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion,OG_IN=True):
        '''
        This function add the elements of PDBs from the stack
        Fusion_chk: True;means add the Fusion PDB in the list
        
        This check the Fusion class if it's empty only move on to the overlapped PDBs
        Where the weightage is splited equally
        '''
        list_of_ids=self.list_of_ids
        list_of_id_classes=self.list_of_id_classes
        # adding the ONGO elements
        if OG_IN:
            if len(index_ongo)>0: 
                list_of_ids.append(self.OG_train_PDBs[index_ongo.pop()])
                list_of_id_classes.append(np.array([1,0,0],dtype=float))
            else:
                print("ONGO idexes reset")
                index_ongo = self.index_stack_chk(index_ongo_chk=True,index_tsg_chk=False,index_fusion_chk=False)
                list_of_ids.append(self.OG_train_PDBs[index_ongo.pop()])
                list_of_id_classes.append(np.array([1,0,0],dtype=float))
            
        if Fusion_chk:
            # adding the Fusion elements
            # print("Fusion element added")
            if len(index_fusion)>0:
                list_of_ids.append(self.Fusion_train_PDBs[index_fusion.pop()])
                list_of_id_classes.append(np.array([0,0,1],dtype=float))
                overlap_chk=False
            else:
                overlap_chk=True
        # adding the TSG elements
        if len(index_tsg)>0:
            list_of_ids.append(self.TSG_train_PDBs[index_tsg.pop()])
            list_of_id_classes.append(np.array([0,1,0],dtype=float))
        else:
            print("TSG idexes reset")
            index_tsg = self.index_stack_chk(index_ongo_chk=False,index_tsg_chk=True,index_fusion_chk=False)
            list_of_ids.append(self.TSG_train_PDBs[index_tsg.pop()])
            list_of_id_classes.append(np.array([0,1,0],dtype=float))
        

        if Fusion_chk and overlap_chk:
            if not len(index_ongo_fusion)>0:
                index_ongo_fusion = self.index_stack_overlapped_chk(True,False)
                if not len(index_tsg_fusion)>0:
                    print("Since both overlapped done thus Fusion idexes are reseten")
                    index_fusion = self.index_stack_chk(index_ongo_chk=False,index_tsg_chk=False,index_fusion_chk=True)
                    index_tsg_fusion = self.index_stack_overlapped_chk(False,True)
                else:
                    list_of_ids.append(self.overlapped_PDBs_TSG[index_tsg_fusion.pop()])
                    list_of_id_classes.append(np.array([0,0.5,0.5],dtype=float))
            else:
                list_of_ids.append(self.overlapped_PDBs_ONGO[index_ongo_fusion.pop()])
                list_of_id_classes.append(np.array([0.5,0,0.5],dtype=float))
                            
        
        self.list_of_ids = list_of_ids
        self.list_of_id_classes = list_of_id_classes
       
        return index_ongo,index_tsg,index_fusion,index_ongo_fusion,index_tsg_fusion



    def adding_fully_clean(self,Fusion_chk,index_ongo,index_tsg,index_fusion,OG_IN=True):
        '''
        This function add the elements of PDBs from the stack
        Fusion_chk: True;means add the Fusion PDB in the list
        '''
        clean_list_of_ids=self.clean_list_of_ids
        clean_list_of_id_classes=self.clean_list_of_id_classes
        # adding the ONGO elements
        if OG_IN:
            if len(index_ongo)>0: 
                clean_list_of_ids.append(self.OG_train_PDBs[index_ongo.pop()])
                clean_list_of_id_classes.append(0)
            else:
                print("ONGO idexes reset")
                index_ongo = self.index_stack_chk(index_ongo_chk=True,index_tsg_chk=False,index_fusion_chk=False)
                clean_list_of_ids.append(self.OG_train_PDBs[index_ongo.pop()])
                clean_list_of_id_classes.append(0)
        
        if Fusion_chk:
            # adding the Fusion elements
            # print("Fusion element added")
            if len(index_fusion)>0:
                clean_list_of_ids.append(self.Fusion_train_PDBs[index_fusion.pop()])
                clean_list_of_id_classes.append(2)
            else:
                print("Fusion idexes reset")
                index_fusion = self.index_stack_chk(index_ongo_chk=False,index_tsg_chk=False,index_fusion_chk=True)
                clean_list_of_ids.append(self.Fusion_train_PDBs[index_fusion.pop()])
                clean_list_of_id_classes.append(2)
        # adding the TSG elements
        if len(index_tsg)>0:
            clean_list_of_ids.append(self.TSG_train_PDBs[index_tsg.pop()])
            clean_list_of_id_classes.append(1)
        else:
            print("TSG idexes reset")
            index_tsg = self.index_stack_chk(index_ongo_chk=False,index_tsg_chk=True,index_fusion_chk=False)
            clean_list_of_ids.append(self.TSG_train_PDBs[index_tsg.pop()])
            clean_list_of_id_classes.append(1)
        
        self.clean_list_of_ids = clean_list_of_ids
        self.clean_list_of_id_classes = clean_list_of_id_classes
        return index_ongo,index_tsg,index_fusion

    def index_stack_overlapped_chk(self,index_ongo_fusion_chk,index_tsg_fusion_chk):
        '''
        This function is used for mainintaining balanced data set 
        thus it shuffled each time, while creating new data
        
        check the stack if any of them is empty then creat a new stack
        This way always change the train data when it cycles
        
        index_ongo_fusion_chk,index_tsg_fusion_chk if one of them is true then take the stack shuffle and return it
        return the newly stacks index_ongo_fusion index_tsg or index_fusion
        '''
        shuffle = self.shuffle
        if index_ongo_fusion_chk:
            print("ONGO Fusion overlapped idexes reset")
            index_ongo_fusion= np.arange(len(self.overlapped_PDBs_ONGO))
            if shuffle:
                np.random.shuffle(index_ongo_fusion)
            return list(index_ongo_fusion)
        if index_tsg_fusion_chk:
            print("TSG Fusion overlapped idexes reset")
            index_tsg_fusion = np.arange(len(self.overlapped_PDBs_TSG))
            if  shuffle:
                np.random.shuffle(index_tsg_fusion)
            return list(index_tsg_fusion)


    def index_stack_chk(self,index_ongo_chk,index_tsg_chk,index_fusion_chk):
        '''
        This function is used for mainintaining balanced data set 
        thus it shuffled each time, while creating new data
        
        This function check the stack if any of them is empty then creat a new stack
        This way always change the train data when it cycles
        
        index_ongo_chk,index_tsg_chk,index_fusion_chk=If one of them is true then take the stack shuffle and return it
        return the newly stacks index_ongo or index_tsg or index_fusion
        '''
        shuffle =self.shuffle
        if index_ongo_chk:
            print("ONGO idexes reset")
            index_ongo= np.arange(len(self.OG_train_PDBs))
            if shuffle:
                np.random.shuffle(index_ongo)
            return list(index_ongo)
        if index_tsg_chk:
            print("TSG idexes reset")
            index_tsg = np.arange(len(self.TSG_train_PDBs))
            if  shuffle:
                np.random.shuffle(index_tsg)
            return list(index_tsg)
        if index_fusion_chk:
            print("Fusion idexes reset")
            index_fusion = np.arange(len(self.Fusion_train_PDBs))
            if  shuffle:
                np.random.shuffle(index_fusion)
            return list(index_fusion)


    def __data_generation_full_clean(self, list_IDs_temp,list_IDs_class_temp):
        'Generates data containing batch_size samples only this part iws editted' # X : (n_samples, *dim, n_channels)
        # Initialization
        if len(list_IDs_temp)<self.batch_size:
            size_data=len(list_IDs_temp)
        else:
           size_data = self.batch_size
            
        q0_x1= np.empty((size_data,*self.dim, self.n_channels))
        q0_x2= np.empty((size_data,*self.dim, self.n_channels))
        q0_x3= np.empty((size_data,*self.dim, self.n_channels))
        
        q1_x1= np.empty((size_data,*self.dim, self.n_channels))
        q1_x2= np.empty((size_data,*self.dim, self.n_channels))
        q1_x3= np.empty((size_data,*self.dim, self.n_channels))
        
        q2_x1= np.empty((size_data,*self.dim, self.n_channels))
        q2_x2= np.empty((size_data,*self.dim, self.n_channels))
        q2_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q3_x1= np.empty((size_data,*self.dim, self.n_channels))
        q3_x2= np.empty((size_data,*self.dim, self.n_channels))
        q3_x3= np.empty((size_data,*self.dim, self.n_channels))

        q4_x1= np.empty((size_data,*self.dim, self.n_channels))
        q4_x2= np.empty((size_data,*self.dim, self.n_channels))
        q4_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q5_x1= np.empty((size_data,*self.dim, self.n_channels))
        q5_x2= np.empty((size_data,*self.dim, self.n_channels))
        q5_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q6_x1= np.empty((size_data,*self.dim, self.n_channels))
        q6_x2= np.empty((size_data,*self.dim, self.n_channels))
        q6_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q7_x1= np.empty((size_data,*self.dim, self.n_channels))
        q7_x2= np.empty((size_data,*self.dim, self.n_channels))
        q7_x3= np.empty((size_data,*self.dim, self.n_channels))

        
        if self.Full_clean:
            y = np.empty((size_data), dtype=int)
        else:
            raise ("Miss use of Function __data_generation_full_clean")
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data_temp = np.load(ID)
            	
            q0_x1[i,] = deepcopy(data_temp[0,0,:,:,:])
            q0_x2[i,] = deepcopy(data_temp[0,1,:,:,:])
            q0_x3[i,] = deepcopy(data_temp[0,2,:,:,:])
            
            q1_x1[i,] = deepcopy(data_temp[1,0,:,:,:])
            q1_x2[i,] = deepcopy(data_temp[1,1,:,:,:])
            q1_x3[i,] = deepcopy(data_temp[1,2,:,:,:])

            q2_x1[i,] = deepcopy(data_temp[2,0,:,:,:])
            q2_x2[i,] = deepcopy(data_temp[2,1,:,:,:])
            q2_x3[i,] = deepcopy(data_temp[2,2,:,:,:])
			
            q3_x1[i,] = deepcopy(data_temp[3,0,:,:,:])
            q3_x2[i,] = deepcopy(data_temp[3,1,:,:,:])
            q3_x3[i,] = deepcopy(data_temp[3,2,:,:,:])
			
            q4_x1[i,] = deepcopy(data_temp[4,0,:,:,:])
            q4_x2[i,] = deepcopy(data_temp[4,1,:,:,:])
            q4_x3[i,] = deepcopy(data_temp[4,2,:,:,:])
			
            q5_x1[i,] = deepcopy(data_temp[5,0,:,:,:])
            q5_x2[i,] = deepcopy(data_temp[5,1,:,:,:])
            q5_x3[i,] = deepcopy(data_temp[5,2,:,:,:])
			
            q6_x1[i,] = deepcopy(data_temp[6,0,:,:,:])
            q6_x2[i,] = deepcopy(data_temp[6,1,:,:,:])
            q6_x3[i,] = deepcopy(data_temp[6,2,:,:,:])
			
            q7_x1[i,] = deepcopy(data_temp[7,0,:,:,:])
            q7_x2[i,] = deepcopy(data_temp[7,1,:,:,:])
            q7_x3[i,] = deepcopy(data_temp[7,2,:,:,:])
            
            y[i] = self.labels[ID]
            if  self.labels[ID]>2:
                print('Wrongly placed id: ',ID)
        
        if len(list_IDs_temp)>self.batch_size:
            print('-----exceed batch size')
        X=[q0_x1,q0_x2,q0_x3,q1_x1,q1_x2,q1_x3,q2_x1,q2_x2,q2_x3,q3_x1,q3_x2,q3_x3,q4_x1,q4_x2,q4_x3,q5_x1,q5_x2,q5_x3,q6_x1,q6_x2,q6_x3,q7_x1,q7_x2,q7_x3]
        return X,keras.utils.to_categorical(y, num_classes=self.n_classes)
  
    def __data_generation(self, list_IDs_temp,list_IDs_class_temp):
        'Generates data containing batch_size samples only this part iws editted' # X : (n_samples, *dim, n_channels)
        # Initialization
        if len(list_IDs_temp)<self.batch_size:
            size_data=len(list_IDs_temp)
        else:
           size_data = self.batch_size
            
        q0_x1= np.empty((size_data,*self.dim, self.n_channels))
        q0_x2= np.empty((size_data,*self.dim, self.n_channels))
        q0_x3= np.empty((size_data,*self.dim, self.n_channels))
        
        q1_x1= np.empty((size_data,*self.dim, self.n_channels))
        q1_x2= np.empty((size_data,*self.dim, self.n_channels))
        q1_x3= np.empty((size_data,*self.dim, self.n_channels))
        
        q2_x1= np.empty((size_data,*self.dim, self.n_channels))
        q2_x2= np.empty((size_data,*self.dim, self.n_channels))
        q2_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q3_x1= np.empty((size_data,*self.dim, self.n_channels))
        q3_x2= np.empty((size_data,*self.dim, self.n_channels))
        q3_x3= np.empty((size_data,*self.dim, self.n_channels))

        q4_x1= np.empty((size_data,*self.dim, self.n_channels))
        q4_x2= np.empty((size_data,*self.dim, self.n_channels))
        q4_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q5_x1= np.empty((size_data,*self.dim, self.n_channels))
        q5_x2= np.empty((size_data,*self.dim, self.n_channels))
        q5_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q6_x1= np.empty((size_data,*self.dim, self.n_channels))
        q6_x2= np.empty((size_data,*self.dim, self.n_channels))
        q6_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q7_x1= np.empty((size_data,*self.dim, self.n_channels))
        q7_x2= np.empty((size_data,*self.dim, self.n_channels))
        q7_x3= np.empty((size_data,*self.dim, self.n_channels))
        
        if not self.Full_clean:
          y = np.empty((size_data,self.n_classes), dtype=float)
        else:
          raise ("Miss use of Function __data_generation")
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data_temp = np.load(ID)
            	
            q0_x1[i,] = deepcopy(data_temp[0,0,:,:,:])
            q0_x2[i,] = deepcopy(data_temp[0,1,:,:,:])
            q0_x3[i,] = deepcopy(data_temp[0,2,:,:,:])
            
            q1_x1[i,] = deepcopy(data_temp[1,0,:,:,:])
            q1_x2[i,] = deepcopy(data_temp[1,1,:,:,:])
            q1_x3[i,] = deepcopy(data_temp[1,2,:,:,:])

            q2_x1[i,] = deepcopy(data_temp[2,0,:,:,:])
            q2_x2[i,] = deepcopy(data_temp[2,1,:,:,:])
            q2_x3[i,] = deepcopy(data_temp[2,2,:,:,:])
			
            q3_x1[i,] = deepcopy(data_temp[3,0,:,:,:])
            q3_x2[i,] = deepcopy(data_temp[3,1,:,:,:])
            q3_x3[i,] = deepcopy(data_temp[3,2,:,:,:])
			
            q4_x1[i,] = deepcopy(data_temp[4,0,:,:,:])
            q4_x2[i,] = deepcopy(data_temp[4,1,:,:,:])
            q4_x3[i,] = deepcopy(data_temp[4,2,:,:,:])
			
            q5_x1[i,] = deepcopy(data_temp[5,0,:,:,:])
            q5_x2[i,] = deepcopy(data_temp[5,1,:,:,:])
            q5_x3[i,] = deepcopy(data_temp[5,2,:,:,:])
			
            q6_x1[i,] = deepcopy(data_temp[6,0,:,:,:])
            q6_x2[i,] = deepcopy(data_temp[6,1,:,:,:])
            q6_x3[i,] = deepcopy(data_temp[6,2,:,:,:])
			
            q7_x1[i,] = deepcopy(data_temp[7,0,:,:,:])
            q7_x2[i,] = deepcopy(data_temp[7,1,:,:,:])
            q7_x3[i,] = deepcopy(data_temp[7,2,:,:,:])
            
            y[i,:] = deepcopy(list_IDs_class_temp[i])
    
        X=[q0_x1,q0_x2,q0_x3,q1_x1,q1_x2,q1_x3,q2_x1,q2_x2,q2_x3,q3_x1,q3_x2,q3_x3,q4_x1,q4_x2,q4_x3,q5_x1,q5_x2,q5_x3,q6_x1,q6_x2,q6_x3,q7_x1,q7_x2,q7_x3]
        return X,y
          
  
    def __getitem__(self, index):
      'Generate one batch of data'
      if self.Full_clean:
          # Generate indexes of the batch
          if len(self.clean_list_of_ids)>(index+1)*self.batch_size:
              indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
          else:
              print("One batch out of size")
              indexes = self.indexes[index*self.batch_size-(self.batch_size-(len(self.clean_list_of_ids)-index*self.batch_size)):len(self.clean_list_of_ids)]
          # Find list of IDs
          list_IDs_temp = [self.clean_list_of_ids[k] for k in indexes]
          list_IDs_class_temp= [self.labels[ID] for ID in list_IDs_temp]
          
          # Generate data
          if index==0:
              print('len(self.clean_list_of_ids: ', len(self.clean_list_of_ids))
              print('(index+1)*self.batch_size: ',(index+1)*self.batch_size)
              print('len(self.indexes): ',len(self.indexes))
              print('len(indexes): ',len(indexes))
              # print('__getitem__: ',index)
              # print(list_IDs_temp)
              print("")
              print("")

          X, y = self.__data_generation_full_clean(list_IDs_temp,list_IDs_class_temp)
      else:
          if len(self.list_of_ids)>(index+1)*self.batch_size:
              indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    #          print('here01')
          else:
              indexes = self.indexes[index*self.batch_size-(self.batch_size-(len(self.list_of_ids)-index*self.batch_size)):len(self.list_of_ids)]
    #          print(len(indexes))
              # Find list of IDs
          list_IDs_temp = [self.list_of_ids[k] for k in indexes]
          list_IDs_class_temp= [self.list_of_id_classes[k] for k in indexes]

          X, y = self.__data_generation(list_IDs_temp,list_IDs_class_temp)
      return X, y


class Data_test:
    'Generates 2D projected data for Keras'
    def __init__(self, list_IDs, labels, batch_size=10, dim = (128,128), n_channels=21 ,n_classes=3):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dim = dim
        self.indexes = np.arange(len(self.list_IDs))

    def getitem_test(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    
      # Find list of IDs
      list_IDs_temp = [self.list_IDs[k] for k in indexes]
    
      # Generate data
      X, y = self.data_generation_test(list_IDs_temp)
      return X, y      
    
    def data_generation_test(self, list_IDs_temp):
        'Generates data containing batch_size samples only this part iws editted' # X : (n_samples, *dim, n_channels)
        # Initialization
        if len(list_IDs_temp)<self.batch_size:
            size_data=len(list_IDs_temp)
        else:
           size_data = self.batch_size
            
        q0_x1= np.empty((size_data,*self.dim, self.n_channels))
        q0_x2= np.empty((size_data,*self.dim, self.n_channels))
        q0_x3= np.empty((size_data,*self.dim, self.n_channels))
        
        q1_x1= np.empty((size_data,*self.dim, self.n_channels))
        q1_x2= np.empty((size_data,*self.dim, self.n_channels))
        q1_x3= np.empty((size_data,*self.dim, self.n_channels))
        
        q2_x1= np.empty((size_data,*self.dim, self.n_channels))
        q2_x2= np.empty((size_data,*self.dim, self.n_channels))
        q2_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q3_x1= np.empty((size_data,*self.dim, self.n_channels))
        q3_x2= np.empty((size_data,*self.dim, self.n_channels))
        q3_x3= np.empty((size_data,*self.dim, self.n_channels))

        q4_x1= np.empty((size_data,*self.dim, self.n_channels))
        q4_x2= np.empty((size_data,*self.dim, self.n_channels))
        q4_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q5_x1= np.empty((size_data,*self.dim, self.n_channels))
        q5_x2= np.empty((size_data,*self.dim, self.n_channels))
        q5_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q6_x1= np.empty((size_data,*self.dim, self.n_channels))
        q6_x2= np.empty((size_data,*self.dim, self.n_channels))
        q6_x3= np.empty((size_data,*self.dim, self.n_channels))
		
        q7_x1= np.empty((size_data,*self.dim, self.n_channels))
        q7_x2= np.empty((size_data,*self.dim, self.n_channels))
        q7_x3= np.empty((size_data,*self.dim, self.n_channels))
        
        y = np.empty((size_data), dtype=int)
     
        X=[]
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data_temp = np.load(ID)
            
            q0_x1[i,] = deepcopy(data_temp[0,0,:,:,:])
            q0_x2[i,] = deepcopy(data_temp[0,1,:,:,:])
            q0_x3[i,] = deepcopy(data_temp[0,2,:,:,:])
            
            q1_x1[i,] = deepcopy(data_temp[1,0,:,:,:])
            q1_x2[i,] = deepcopy(data_temp[1,1,:,:,:])
            q1_x3[i,] = deepcopy(data_temp[1,2,:,:,:])

            q2_x1[i,] = deepcopy(data_temp[2,0,:,:,:])
            q2_x2[i,] = deepcopy(data_temp[2,1,:,:,:])
            q2_x3[i,] = deepcopy(data_temp[2,2,:,:,:])
			
            q3_x1[i,] = deepcopy(data_temp[3,0,:,:,:])
            q3_x2[i,] = deepcopy(data_temp[3,1,:,:,:])
            q3_x3[i,] = deepcopy(data_temp[3,2,:,:,:])
			
            q4_x1[i,] = deepcopy(data_temp[4,0,:,:,:])
            q4_x2[i,] = deepcopy(data_temp[4,1,:,:,:])
            q4_x3[i,] = deepcopy(data_temp[4,2,:,:,:])
			
            q5_x1[i,] = deepcopy(data_temp[5,0,:,:,:])
            q5_x2[i,] = deepcopy(data_temp[5,1,:,:,:])
            q5_x3[i,] = deepcopy(data_temp[5,2,:,:,:])
			
            q6_x1[i,] = deepcopy(data_temp[6,0,:,:,:])
            q6_x2[i,] = deepcopy(data_temp[6,1,:,:,:])
            q6_x3[i,] = deepcopy(data_temp[6,2,:,:,:])
			
            q7_x1[i,] = deepcopy(data_temp[7,0,:,:,:])
            q7_x2[i,] = deepcopy(data_temp[7,1,:,:,:])
            q7_x3[i,] = deepcopy(data_temp[7,2,:,:,:])
            
            # Store class
            y[i] = self.labels[ID]
       
        X=[q0_x1,q0_x2,q0_x3,q1_x1,q1_x2,q1_x3,q2_x1,q2_x2,q2_x3,q3_x1,q3_x2,q3_x3,q4_x1,q4_x2,q4_x3,q5_x1,q5_x2,q5_x3,q6_x1,q6_x2,q6_x3,q7_x1,q7_x2,q7_x3]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
  
    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem_list_id__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      if len(self.list_IDs)>(index+1)*self.batch_size:
          indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#          print('here01')
      else:
          indexes = self.indexes[index*self.batch_size-(self.batch_size-(len(self.list_IDs)-index*self.batch_size)):len(self.list_IDs)]
#          print(len(indexes))
      # Find list of IDs
      list_IDs_temp = [self.list_IDs[k] for k in indexes]
      return list_IDs_temp
  
