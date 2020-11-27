# -*- coding: utf-8 -*-
"""
Created on %14-May-2020 at 04.43p.m
@author: %A.Nishanth C00294860
"""
import os
import numpy as np
import pickle
from copy import deepcopy

class retrieve_comp_results_not_valid_10_fold_NN:
    """
    This class retrieve the 10-fold results of the ensamble metghod if the PDB not in validation set else return the validation probability
    """
    
    def __init__(self,model_name,nn_probability,nn_probability_left_pdb,saving_dir,ear_prob=True):
        self.saving_dir=saving_dir
        test_prob_dic={}
        for nn_probability_sel in [nn_probability,nn_probability_left_pdb]:
            os.chdir('/')
            os.chdir(nn_probability_sel)
            for nth_fold in range(0,10):
            #nth_fold=1
                test_PDBs= pickle.load(open(''.join(["pdbs_test_",str(nth_fold),".p"]), "rb"))
                test_probabilities = pickle.load(open(''.join(["test_probabilities_",str(nth_fold),".p"]), "rb"))
                #go through the PDBs and retrie the results
                for i in range(0,len(test_PDBs)):
                    for j in range(0,len(test_probabilities[i])):
                        if nth_fold==0:
                            if i==302 and ear_prob:
                                test_prob_dic[test_PDBs[i][3+j]]=self.prob_fix_func(test_probabilities[i][j,:])
                            else:
                                test_prob_dic[test_PDBs[i][j]]=self.prob_fix_func(test_probabilities[i][j,:])
                        else:
                            if i==302 and ear_prob:
                                test_prob_dic[test_PDBs[i][3+j]]=test_prob_dic[test_PDBs[i][3+j]]+self.prob_fix_func(test_probabilities[i][j,:])
                            else:
                                test_prob_dic[test_PDBs[i][j]]=test_prob_dic[test_PDBs[i][j]]+self.prob_fix_func(test_probabilities[i][j,:])
            
        test_PDBs=[]
        for key in test_prob_dic.keys(): 
                test_PDBs.append(key) 
                
        for pdbs in test_PDBs:
            test_prob_dic[pdbs]=self.prob_fix_func(test_prob_dic[pdbs]/10)
    
        '''save the results'''
        os.chdir('/')
        os.chdir(saving_dir)      

        OG_test_probs_numpy = np.load('OG_10_fold_probs.npy')
        TSG_test_probs_numpy = np.load('TSG_10_fold_probs.npy')
        Fusion_test_probs_numpy =np.load('Fusion_10_fold_probs.npy')
        
        OG_test_pdbs = pickle.load(open("OG_10_fold_PDBs.p", "rb"))  
        TSG_test_pdbs = pickle.load(open("TSG_10_fold_PDBs.p", "rb"))  
        Fusion_test_pdbs = pickle.load(open("Fusion_10_fold_PDBs.p", "rb"))  
        
        test_prob_dic=self.update_or_extend_dic(OG_test_pdbs,OG_test_probs_numpy,test_prob_dic)
        test_prob_dic=self.update_or_extend_dic(TSG_test_pdbs,TSG_test_probs_numpy,test_prob_dic)
        test_prob_dic=self.update_or_extend_dic(Fusion_test_pdbs,Fusion_test_probs_numpy,test_prob_dic)

        test_PDBs=list(set(test_PDBs+OG_test_pdbs+TSG_test_pdbs+Fusion_test_pdbs))

        pickle.dump(test_prob_dic, open(''.join([model_name,"test_prob_dic.p"]), "wb"))  
        self.test_prob_dic=test_prob_dic
        self.test_PDBs=test_PDBs
        
        
        
    def Helper_to_get_T_1_ROC_for_differ_version(self,satisfied_dir,PDB_list_icluded=False):
        '''
        This function helps to retrieve the results for Tier-1 ONGO, TSG, Fusion results
        '''
        os.chdir('/')    
        os.chdir(satisfied_dir)        
        OG_V2_pdbs = list(set(sum(pickle.load(open("ONGO_satisfied_PDBs.p", "rb")),[])))  
        TSG_V2_pdbs = list(set(sum(pickle.load(open("TSG_satisfied_PDBs.p", "rb")),[])))  
        Fusion_V2_pdbs = list(set(sum(pickle.load(open("Fusion_satisfied_PDBs.p", "rb")),[])))  
                 
        ONGO_test_probs_numpy=self.given_PDBs_prob(OG_V2_pdbs)
        TSG_test_probs_numpy =self.given_PDBs_prob(TSG_V2_pdbs)
        Fusion_test_probs_numpy =self.given_PDBs_prob(Fusion_V2_pdbs)
        
        os.chdir('/')
        os.chdir(self.saving_dir)   
        pickle.dump(ONGO_test_probs_numpy, open("T1_all_ONGO_probs_numpy.p", "wb"))  
        pickle.dump(TSG_test_probs_numpy, open("T1_all_TSG_probs_numpy.p", "wb"))  
        pickle.dump(Fusion_test_probs_numpy, open("T1_all_Fusion_probs_numpy.p", "wb"))  
        if PDB_list_icluded: 
            return OG_V2_pdbs,TSG_V2_pdbs,Fusion_V2_pdbs,ONGO_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy
        else:
            return ONGO_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy
    
    def update_or_extend_dic(self,given_pdb,given_prob,test_prob_dic):
        print("Given Dictionary length: ",len(test_prob_dic))
        for i in range(0,len(given_pdb)):
            test_prob_dic[given_pdb[i]]=given_prob[i,:]
        print("Updated Dictioynary length: ",len(test_prob_dic))

        return test_prob_dic    
    
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
        
    def given_PDBs_prob(self,given_pdbs): 
        given_pdbs_probs_numpy =np.empty((len(given_pdbs),3))    
        for i in range(0,len(given_pdbs)):
            chk_pdb=''.join([given_pdbs[i],'.npy'])
            given_pdbs_probs_numpy[i,:]=deepcopy(self.test_prob_dic[chk_pdb])  
            if given_pdbs[i]=='5B5W':
                print('5B5W updated: ',self.test_prob_dic[chk_pdb])
            elif given_pdbs[i]=='5BRK':
                print('5BRK updated: ',self.test_prob_dic[chk_pdb])
        return given_pdbs_probs_numpy
             
    
    def update_the_dic_keys(self):
        for old_key in self.test_PDBs:
            self.test_prob_dic[old_key[0:-4]] = self.test_prob_dic.pop(old_key)
			
class load_comp_results_not_valid_10_fold_NN:
    """
    This class retrieve the 10-fold results of the ensamble metghod if the PDB not in validation set else return the validation probability
    """
    def __init__(self,model_name,nn_probability,nn_probability_left_pdb,saving_dir,ear_prob=True):
        self.saving_dir=saving_dir
        test_prob_dic={}
        for nn_probability_sel in [nn_probability,nn_probability_left_pdb]:
            os.chdir('/')
            os.chdir(nn_probability_sel)
            for nth_fold in range(0,10):
            #nth_fold=1
                test_PDBs= pickle.load(open(''.join(["pdbs_test_",str(nth_fold),".p"]), "rb"))
                test_probabilities = pickle.load(open(''.join(["test_probabilities_",str(nth_fold),".p"]), "rb"))
                #go through the PDBs and retrie the results
                for i in range(0,len(test_PDBs)):
                    for j in range(0,len(test_probabilities[i])):
                        if nth_fold==0:
                            if i==302 and ear_prob:
                                test_prob_dic[test_PDBs[i][3+j]]=self.prob_fix_func(test_probabilities[i][j,:])
                            else:
                                test_prob_dic[test_PDBs[i][j]]=self.prob_fix_func(test_probabilities[i][j,:])
                        else:
                            if i==302 and ear_prob:
                                test_prob_dic[test_PDBs[i][3+j]]=test_prob_dic[test_PDBs[i][3+j]]+self.prob_fix_func(test_probabilities[i][j,:])
                            else:
                                test_prob_dic[test_PDBs[i][j]]=test_prob_dic[test_PDBs[i][j]]+self.prob_fix_func(test_probabilities[i][j,:])
            
        test_PDBs=[]
        for key in test_prob_dic.keys(): 
                test_PDBs.append(key) 
                
        for pdbs in test_PDBs:
            test_prob_dic[pdbs]=self.prob_fix_func(test_prob_dic[pdbs]/10)
        self.test_prob_dic=test_prob_dic
        self.test_PDBs=test_PDBs


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