# -*- coding: utf-8 -*-
"""
Created on %30-Jan-2020 at 11.00A.m
changed_to class format on 04-02-2020 at 1.30pm
reditted T1_and T2 on 10-May-2020
@author: %A.Nishanth C00294860
"""
import os
import numpy as np
from copy import deepcopy
import pickle


class retrieve_results_10_fold_NN:
    """
    This class calculte the ROCcurve and cnfusion matrix for the given probabilities
    """
    
    def __init__(self,main_dir,nn_probability,saving_dir):
  
        '''Load the results'''
        os.chdir('/')
        os.chdir(main_dir)
        #for organise the results or performance
        overlapped_PDBs_ONGO= pickle.load(open("overlapped_PDBs_ONGO.p", "rb"))
        overlapped_PDBs_TSG= pickle.load(open("overlapped_PDBs_TSG.p", "rb"))
        test_labels_dic= pickle.load(open("train_labels_dic.p", "rb"))


        '''go through each fold classification and extract the results'''
        print("")
        print("")
        print("")
        
        print("start from 0 and end at 10")
        print("")
        print("")
        OG_test_probs=[]
        OG_test_PDBs=[]
        TSG_test_probs=[]
        TSG_test_PDBs=[]
        Fusion_test_probs=[]
        Fusion_test_PDBs=[]
        test_OG_len=0
        test_TSG_len=0
        test_Fusion_len=0
        for nth_fold in range(0,10):
        #for nth_fold in range(2,7):
            os.chdir('/')
            os.chdir(nn_probability)
            
            #splited_clean_Fusion =  pickle.load(open("10_splited_clean_Fusion.p", "rb"))
            #splited_clean_ONGO =  pickle.load(open("10_splited_clean_ONGO.p", "rb"))
            #splited_clean_TSG =  pickle.load(open("10_splited_clean_TSG.p", "rb"))
        
            test_PDBs= pickle.load(open(''.join(["pdbs_valid_fold_",str(nth_fold),".p"]), "rb"))
            test_probabilities = pickle.load(open(''.join(["valid_probabilities_fold_",str(nth_fold),".p"]), "rb"))
            
            #% Finalise the Test results
            #sum(a, [])#to make as one list
            '''Extract the test results'''
            OG_test_probs_t=[]
            OG_test_PDBs_t=[]
            TSG_test_probs_t=[]
            TSG_test_PDBs_t=[]
            Fusion_test_probs_t=[]
            Fusion_test_PDBs_t=[]
            for i in range(0,len(test_PDBs)):
                for j in range(0,len(test_PDBs[i])):
                    if test_labels_dic[test_PDBs[i][j]]==0 and (test_PDBs[i][j] not in overlapped_PDBs_ONGO):
                        OG_test_probs_t.append(self.prob_fix_func(deepcopy(test_probabilities[i][j,:])))
                        OG_test_PDBs_t.append(test_PDBs[i][j])
                    elif test_labels_dic[test_PDBs[i][j]]==1 and (test_PDBs[i][j] not in overlapped_PDBs_TSG):
                        TSG_test_probs_t.append(self.prob_fix_func(deepcopy(test_probabilities[i][j,:])))
                        TSG_test_PDBs_t.append(test_PDBs[i][j])
                    elif test_labels_dic[test_PDBs[i][j]]==2:
                        Fusion_test_probs_t.append(self.prob_fix_func(deepcopy(test_probabilities[i][j,:])))
                        Fusion_test_PDBs_t.append(test_PDBs[i][j])
                    else:
                        raise("Some thing wrong")
                        
            test_OG_len=test_OG_len + len(OG_test_probs_t)
            test_TSG_len=test_TSG_len + len(TSG_test_probs_t)
            test_Fusion_len=test_Fusion_len + len(Fusion_test_probs_t)
               
            OG_test_probs.append(deepcopy(OG_test_probs_t))
            OG_test_PDBs.append(deepcopy(OG_test_PDBs_t))
            TSG_test_probs.append(deepcopy(TSG_test_probs_t))
            TSG_test_PDBs.append(deepcopy(TSG_test_PDBs_t))
            Fusion_test_probs.append(deepcopy(Fusion_test_probs_t))
            Fusion_test_PDBs.append(deepcopy(Fusion_test_PDBs_t))
            del test_probabilities
            del test_PDBs
        
        
        '''then finalise the probabilities to save them and get the accuracy percenatge while doing it'''
        OG_PDBs_validated=[]
        OG_test_probs_numpy=np.empty((test_OG_len,3))
        OG_validation_correct=0
        k=0
        for j in range(0,len(OG_test_probs)):
            OG_test_probs_t = deepcopy(OG_test_probs[j])
            OG_test_PDBs_t = deepcopy(OG_test_PDBs[j])
            for i in range(0,len(OG_test_probs_t)):
                OG_test_probs_numpy[k+i,:]=deepcopy(OG_test_probs_t[i])
                if OG_test_probs_t[i][0]>= OG_test_probs_t[i][1] and OG_test_probs_t[i][0]>= OG_test_probs_t[i][2]:
                    OG_validation_correct=OG_validation_correct+1
                OG_PDBs_validated.append(OG_test_PDBs_t[i])
            k=k+ len(OG_test_probs_t)
        ONGO_acc=round(100*OG_validation_correct/test_OG_len,2)
        print("Overall ONGO 10-fold cross validation accuracy: ",ONGO_acc,' %')
        
        TSG_PDBs_validated=[]
        TSG_test_probs_numpy=np.empty((test_TSG_len,3))
        TSG_validation_correct=0
        k=0
        for j in range(0,len(TSG_test_probs)):
            TSG_test_probs_t = deepcopy(TSG_test_probs[j])
            TSG_test_PDBs_t = deepcopy(TSG_test_PDBs[j])
            for i in range(0,len(TSG_test_probs_t)):
                TSG_test_probs_numpy[k+i,:]=deepcopy(TSG_test_probs_t[i])
                if TSG_test_probs_t[i][0]<= TSG_test_probs_t[i][1] and TSG_test_probs_t[i][1]>= TSG_test_probs_t[i][2]:
                    TSG_validation_correct=TSG_validation_correct+1
                TSG_PDBs_validated.append(TSG_test_PDBs_t[i])
            k=k+ len(TSG_test_probs_t)
        TSG_acc= round(100*TSG_validation_correct/test_TSG_len,2)
        print("Overall TSG 10-fold cross validation accuracy: ",TSG_acc,' %')
        
        Fusion_PDBs_validated=[]
        Fusion_test_probs_numpy=np.empty((test_Fusion_len,3))
        Fusion_validation_correct=0
        k=0
        for j in range(0,len(Fusion_test_probs)):
            Fusion_test_probs_t = deepcopy(Fusion_test_probs[j])
            Fusion_test_PDBs_t = deepcopy(Fusion_test_PDBs[j])
            for i in range(0,len(Fusion_test_probs_t)):
                Fusion_test_probs_numpy[k+i,:]=deepcopy(Fusion_test_probs_t[i])
                if Fusion_test_probs_t[i][2]>= Fusion_test_probs_t[i][1] and Fusion_test_probs_t[i][0]<= Fusion_test_probs_t[i][2]:
                    Fusion_validation_correct=Fusion_validation_correct+1
                Fusion_PDBs_validated.append(Fusion_test_PDBs_t[i])
            k=k+ len(Fusion_test_probs_t)
        Fusion_acc=round(100*Fusion_validation_correct/test_Fusion_len,2)
        print("Overall Fusion 10-fold cross validation accuracy: ",Fusion_acc,' %')
        Overall_acc= round(100*(OG_validation_correct+TSG_validation_correct+Fusion_validation_correct)/(test_OG_len+test_TSG_len+test_Fusion_len),2)
        print("Overall combined 10-fold cross validation accuracy: ",Overall_acc,' %')

        '''save the results'''
        os.chdir('/')
        if not os.path.isdir(saving_dir):
            os.makedirs(saving_dir)
        os.chdir('/')
        os.chdir(saving_dir)
        
        np.save('OG_10_fold_probs.npy', OG_test_probs_numpy)
        np.save('TSG_10_fold_probs.npy', TSG_test_probs_numpy)
        np.save('Fusion_10_fold_probs.npy', Fusion_test_probs_numpy)
        
        pickle.dump(OG_PDBs_validated, open("OG_10_fold_PDBs.p", "wb"))  
        pickle.dump(TSG_PDBs_validated, open("TSG_10_fold_PDBs.p", "wb")) 
        pickle.dump(Fusion_PDBs_validated, open("Fusion_10_fold_PDBs.p", "wb"))  
        self.Accuracies= np.array([Overall_acc,ONGO_acc,TSG_acc,Fusion_acc])
        

    def prob_fix_func(self,prob,round_fac=2):
        '''
        This function fix the probability added to one and round up the probability
        '''
        prob = np.round(prob,2)
        if sum(prob)==1:
            return prob
        else:
            all_sum=sum(prob)
            prob[0] = prob[0]/all_sum
            prob[1] = prob[1]/all_sum
            prob[2] = prob[2]/all_sum
            return np.round(prob,2)
        
