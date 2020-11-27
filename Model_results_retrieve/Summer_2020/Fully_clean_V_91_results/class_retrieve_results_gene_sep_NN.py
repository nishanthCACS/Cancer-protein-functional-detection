# -*- coding: utf-8 -*-
"""
Created on %03-Feb-2020s

@author: %A.Nishanth C00294860
"""
import os
import numpy as np
from copy import deepcopy
import pickle

class retrieve_results_gene_sep_NN:
    """
    This class calculte the ROCcurve and cnfusion matrix for the given probabilities
    """
    
    def __init__(self,main_dir,nn_probability,saving_dir,model_name='giv'):
        
        '''First extract the Trian, Validation and Test set details'''
        self.saving_dir=saving_dir
        os.chdir('/')
        os.chdir(nn_probability)
        if model_name=='giv':
            train_PDBs= pickle.load(open("pdbs_train_test.p", "rb"))
            valid_ids= pickle.load(open("valid_ids.p", "rb"))
            test_PDBs= pickle.load(open("pdbs_test.p", "rb"))
            train_probabilities = pickle.load(open("train_test_probabilities.p", "rb"))
            test_probabilities = pickle.load(open("test_probabilities.p", "rb"))
        else:
            train_PDBs= pickle.load(open(''.join([model_name,"_pdbs_train_test.p"]), "rb"))
            valid_ids= pickle.load(open(''.join([model_name,"_valid_ids.p"]), "rb"))
            test_PDBs= pickle.load(open(''.join([model_name,"_pdbs_test.p"]), "rb"))
            train_probabilities = pickle.load(open(''.join([model_name,"_train_test_probabilities.p"]), "rb"))
            test_probabilities = pickle.load(open(''.join([model_name,"_test_probabilities.p"]), "rb"))
        os.chdir('/')
        os.chdir(main_dir)
        #for organise the results or performance
        OG_train_PDBs=  pickle.load(open("OG_train_PDBs.p", "rb"))
        TSG_train_PDBs=  pickle.load(open("TSG_train_PDBs.p", "rb"))    
        Fusion_train_PDBs=  pickle.load(open("Fusion_train_PDBs.p", "rb"))
        
        overlapped_PDBs_ONGO= pickle.load(open("overlapped_PDBs_ONGO.p", "rb"))
        overlapped_PDBs_TSG= pickle.load(open("overlapped_PDBs_TSG.p", "rb"))
        test_labels_dic= pickle.load(open("test_labels_dic.p", "rb"))
        
        OG_test_probs=[]
        OG_test_PDBs=[]
        TSG_test_probs=[]
        TSG_test_PDBs=[]
        Fusion_test_probs=[]
        Fusion_test_PDBs=[]
        '''Extract the test results'''
        for i in range(0,len(test_PDBs)):
            for j in range(0,len(test_PDBs[i])):
#                if test_labels_dic[test_PDBs[i][j]]==0:# and (test_PDBs[i][j] not in overlapped_PDBs_ONGO):
                if test_labels_dic[test_PDBs[i][j]]==0 and (test_PDBs[i][j] not in overlapped_PDBs_ONGO):
                    OG_test_probs.append(self.prob_fix_func(deepcopy(test_probabilities[i][j,:])))
                    OG_test_PDBs.append(test_PDBs[i][j])
#                elif test_labels_dic[test_PDBs[i][j]]==1:# and (test_PDBs[i][j] not in overlapped_PDBs_TSG):
                elif test_labels_dic[test_PDBs[i][j]]==1 and (test_PDBs[i][j] not in overlapped_PDBs_TSG):
                    TSG_test_probs.append(self.prob_fix_func(deepcopy(test_probabilities[i][j,:])))
                    TSG_test_PDBs.append(test_PDBs[i][j])
#                elif test_labels_dic[test_PDBs[i][j]]==2:# and (test_PDBs[i][j] not in overlapped_PDBs_ONGO) and (test_PDBs[i][j] not in overlapped_PDBs_TSG):
                elif test_labels_dic[test_PDBs[i][j]]==2 and (test_PDBs[i][j] not in overlapped_PDBs_ONGO) and (test_PDBs[i][j] not in overlapped_PDBs_TSG):
                    Fusion_test_probs.append(self.prob_fix_func(deepcopy(test_probabilities[i][j,:])))
                    Fusion_test_PDBs.append(test_PDBs[i][j])
                else:
                    raise("Some thing wrong")
        print("OG test PDBs length: ",len(OG_test_PDBs))
        print("TSG test PDBs length: ",len(TSG_test_PDBs))
        print("Fusion test PDBs length: ",len(Fusion_test_PDBs))

        del test_probabilities
        OG_test_probs_numpy=np.empty((len(OG_test_probs),3))
        for i in range(0,len(OG_test_probs)):
            OG_test_probs_numpy[i,:]=deepcopy(OG_test_probs[i])
        del OG_test_probs
        
        TSG_test_probs_numpy=np.empty((len(TSG_test_probs),3))
        for i in range(0,len(TSG_test_probs)):
            TSG_test_probs_numpy[i,:]=deepcopy(TSG_test_probs[i])
        del TSG_test_probs
        
        Fusion_test_probs_numpy=np.empty((len(Fusion_test_probs),3))
        for i in range(0,len(Fusion_test_probs)):
            Fusion_test_probs_numpy[i,:]=deepcopy(Fusion_test_probs[i])
        del Fusion_test_probs
        
        os.chdir('/')
        if not os.path.isdir(saving_dir):
            os.makedirs(saving_dir)
            print('')
            print(saving_dir)
            print('')
            print('Path created success fully')
            print('')

        ONGO_acc = self.acc_helper(OG_test_probs_numpy,0)
        TSG_acc = self.acc_helper(TSG_test_probs_numpy,1)
        Fusion_acc = self.acc_helper(Fusion_test_probs_numpy,2)
        over_all_acc= ((ONGO_acc*len(OG_test_probs_numpy))+(TSG_acc*len(TSG_test_probs_numpy))+(Fusion_acc*len(Fusion_test_probs_numpy)))/(len(OG_test_probs_numpy)+len(TSG_test_probs_numpy)+len(Fusion_test_probs_numpy))
        self.Accuracies=np.array([round(over_all_acc,2),round(ONGO_acc,2),round(TSG_acc,2),round(Fusion_acc,2)])


        os.chdir('/')
        os.chdir(saving_dir)
        
        np.save('OG_test_probs.npy', OG_test_probs_numpy)
        np.save('TSG_test_probs.npy', TSG_test_probs_numpy)
        np.save('Fusion_test_probs.npy', Fusion_test_probs_numpy)
        
        pickle.dump(OG_test_PDBs, open("OG_test_PDBs.p", "wb"))  
        pickle.dump(TSG_test_PDBs, open("TSG_test_PDBs.p", "wb")) 
        pickle.dump(Fusion_test_PDBs, open("Fusion_test_PDBs.p", "wb"))
        print("Test set retreval done")
        
        self.OG_test_PDBs=OG_test_PDBs
        self.TSG_test_PDBs=TSG_test_PDBs
        self.Fusion_test_PDBs=Fusion_test_PDBs
        
        OG_train_probs=[]
        OG_train_PDBs_fin=[]
        TSG_train_probs=[]
        TSG_train_PDBs_fin=[]
        Fusion_train_probs=[]
        Fusion_train_PDBs_fin=[]
        OG_Fusion_overlap_train_PDBs=[]
        TSG_Fusion_overlap_train_PDBs=[]
        OG_Fusion_overlap_train_prob=[]
        TSG_Fusion_overlap_train_prob=[]
        
        OG_valid_probs=[]
        OG_valid_PDBs_fin=[]
        TSG_valid_probs=[]
        TSG_valid_PDBs_fin=[]
        Fusion_valid_probs=[]
        Fusion_valid_PDBs_fin=[]
        OG_Fusion_overlap_valid_PDBs=[]
        TSG_Fusion_overlap_valid_PDBs=[]
        OG_Fusion_overlap_valid_prob=[]
        TSG_Fusion_overlap_valid_prob=[]
        for i in range(0,len(train_PDBs)):
            for j in range(0,len(train_PDBs[i])):
                if train_PDBs[i][j] in valid_ids:
                    if (train_PDBs[i][j] in OG_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_ONGO):
                        OG_valid_probs.append(self.prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                        OG_valid_PDBs_fin.append(train_PDBs[i][j])
                    elif (train_PDBs[i][j] in TSG_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_TSG):
                        TSG_valid_probs.append(self.prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                        TSG_valid_PDBs_fin.append(train_PDBs[i][j])
                    elif  (train_PDBs[i][j] in Fusion_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_ONGO) and (train_PDBs[i][j] not in overlapped_PDBs_TSG) :
                        Fusion_valid_probs.append(self.prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                        Fusion_valid_PDBs_fin.append(train_PDBs[i][j])
                    elif (train_PDBs[i][j] in overlapped_PDBs_ONGO):
                        OG_Fusion_overlap_valid_prob.append(self.prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                        OG_Fusion_overlap_valid_PDBs.append(train_PDBs[i][j])
                    elif (train_PDBs[i][j] in overlapped_PDBs_TSG):
                        TSG_Fusion_overlap_valid_prob.append(self.prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                        TSG_Fusion_overlap_valid_PDBs.append(train_PDBs[i][j])
                    else:
                        if train_PDBs[i][j] in test_PDBs:
                            print(train_PDBs[i][j])
                            raise('problem encountered: ',train_PDBs[i][j], ' Not fit in training categorisation')
                        print(train_PDBs[i][j]," again")
                else:
                    if (train_PDBs[i][j] in OG_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_ONGO) and (train_PDBs[i][j] not in OG_train_PDBs_fin):
                        OG_train_probs.append(self.prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                        OG_train_PDBs_fin.append(train_PDBs[i][j])
                    elif (train_PDBs[i][j] in TSG_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_TSG) and (train_PDBs[i][j] not in TSG_train_PDBs_fin):
                        TSG_train_probs.append(self.prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                        TSG_train_PDBs_fin.append(train_PDBs[i][j])
                    elif  (train_PDBs[i][j] in Fusion_train_PDBs) and (train_PDBs[i][j] not in overlapped_PDBs_ONGO) and (train_PDBs[i][j] not in overlapped_PDBs_TSG) and (train_PDBs[i][j] not in Fusion_train_PDBs_fin) :
                        Fusion_train_probs.append(self.prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                        Fusion_train_PDBs_fin.append(train_PDBs[i][j])
                    elif (train_PDBs[i][j] in overlapped_PDBs_ONGO) and (train_PDBs[i][j] not in OG_Fusion_overlap_train_PDBs):
                        OG_Fusion_overlap_train_prob.append(self.prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                        OG_Fusion_overlap_train_PDBs.append(train_PDBs[i][j])
                    elif (train_PDBs[i][j] in overlapped_PDBs_TSG) and (train_PDBs[i][j] not in TSG_Fusion_overlap_train_PDBs):
                        TSG_Fusion_overlap_train_prob.append(self.prob_fix_func(deepcopy(train_probabilities[i][j,:])))
                        TSG_Fusion_overlap_train_PDBs.append(train_PDBs[i][j])
                    else:
                        if train_PDBs[i][j] in test_PDBs:
                            print(train_PDBs[i][j])
                            raise('problem encountered: ',train_PDBs[i][j], ' Not fit in training categorisation')
                        print(train_PDBs[i][j]," again")
#                        raise('small problem encountered: ',train_PDBs[i][j], ' Not fit in training categorisation')     
        
        del train_probabilities
        OG_train_probs_numpy=np.empty((len(OG_train_probs),3))
        for i in range(0,len(OG_train_probs)):
            OG_train_probs_numpy[i,:]=deepcopy(OG_train_probs[i])
        del OG_train_probs
        
        TSG_train_probs_numpy=np.empty((len(TSG_train_probs),3))
        for i in range(0,len(TSG_train_probs)):
            TSG_train_probs_numpy[i,:]=deepcopy(TSG_train_probs[i])
        del TSG_train_probs
        
        Fusion_train_probs_numpy=np.empty((len(Fusion_train_probs),3))
        for i in range(0,len(Fusion_train_probs)):
            Fusion_train_probs_numpy[i,:]=deepcopy(Fusion_train_probs[i])
        del Fusion_train_probs
        
        OG_Fusion_overlap_train_probs_numpy=np.empty((len(OG_Fusion_overlap_train_prob),3))
        for i in range(0,len(OG_Fusion_overlap_train_prob)):
            OG_Fusion_overlap_train_probs_numpy[i,:]=deepcopy(OG_Fusion_overlap_train_prob[i])
        del OG_Fusion_overlap_train_prob
        
        TSG_Fusion_overlap_train_probs_numpy=np.empty((len(TSG_Fusion_overlap_train_prob),3))
        for i in range(0,len(TSG_Fusion_overlap_train_prob)):
            TSG_Fusion_overlap_train_probs_numpy[i,:]=deepcopy(TSG_Fusion_overlap_train_prob[i])
        del TSG_Fusion_overlap_train_prob
      
        '''validation set formation '''

        OG_valid_probs_numpy=np.empty((len(OG_valid_probs),3))
        for i in range(0,len(OG_valid_probs)):
            OG_valid_probs_numpy[i,:]=deepcopy(OG_valid_probs[i])
        del OG_valid_probs
        
        TSG_valid_probs_numpy=np.empty((len(TSG_valid_probs),3))
        for i in range(0,len(TSG_valid_probs)):
            TSG_valid_probs_numpy[i,:]=deepcopy(TSG_valid_probs[i])
        del TSG_valid_probs
        
        Fusion_valid_probs_numpy=np.empty((len(Fusion_valid_probs),3))
        for i in range(0,len(Fusion_valid_probs)):
            Fusion_valid_probs_numpy[i,:]=deepcopy(Fusion_valid_probs[i])
        del Fusion_valid_probs
        
        OG_Fusion_overlap_valid_probs_numpy=np.empty((len(OG_Fusion_overlap_valid_prob),3))
        for i in range(0,len(OG_Fusion_overlap_valid_prob)):
            OG_Fusion_overlap_valid_probs_numpy[i,:]=deepcopy(OG_Fusion_overlap_valid_prob[i])
        del OG_Fusion_overlap_valid_prob
        
        TSG_Fusion_overlap_valid_probs_numpy=np.empty((len(TSG_Fusion_overlap_valid_prob),3))
        for i in range(0,len(TSG_Fusion_overlap_valid_prob)):
            TSG_Fusion_overlap_valid_probs_numpy[i,:]=deepcopy(TSG_Fusion_overlap_valid_prob[i])
        del TSG_Fusion_overlap_valid_prob
        
        '''saving of training set'''
        
        np.save('OG_valid_probs.npy', OG_valid_probs_numpy)
        np.save('TSG_valid_probs.npy', TSG_valid_probs_numpy)
        np.save('Fusion_valid_probs.npy', Fusion_valid_probs_numpy)
        np.save('OG_Fusion_overlap_valid_probs.npy', OG_Fusion_overlap_valid_probs_numpy)
        np.save('TSG_Fusion_overlap_valid_probs.npy', TSG_Fusion_overlap_valid_probs_numpy)
        pickle.dump(OG_valid_PDBs_fin, open("OG_valid_PDBs.p", "wb"))  
        pickle.dump(TSG_valid_PDBs_fin, open("TSG_valid_PDBs.p", "wb")) 
        pickle.dump(Fusion_valid_PDBs_fin, open("Fusion_valid_PDBs.p", "wb"))  
        pickle.dump(OG_Fusion_overlap_valid_PDBs, open("OG_Fusion_overlap_valid_PDBs.p", "wb")) 
        pickle.dump(TSG_Fusion_overlap_valid_PDBs, open("TSG_Fusion_overlap_valid_PDBs.p", "wb"))  
        
        np.save('OG_train_probs.npy', OG_train_probs_numpy)
        np.save('TSG_train_probs.npy', TSG_train_probs_numpy)
        np.save('Fusion_train_probs.npy', Fusion_train_probs_numpy)
        np.save('OG_Fusion_overlap_train_probs.npy', OG_Fusion_overlap_train_probs_numpy)
        np.save('TSG_Fusion_overlap_train_probs.npy', TSG_Fusion_overlap_train_probs_numpy)
        
        pickle.dump(OG_train_PDBs_fin, open("OG_train_PDBs.p", "wb"))  
        pickle.dump(TSG_train_PDBs_fin, open("TSG_train_PDBs.p", "wb")) 
        pickle.dump(Fusion_train_PDBs_fin, open("Fusion_train_PDBs.p", "wb"))  
        pickle.dump(OG_Fusion_overlap_train_PDBs, open("OG_Fusion_overlap_train_PDBs.p", "wb")) 
        pickle.dump(TSG_Fusion_overlap_train_PDBs, open("TSG_Fusion_overlap_train_PDBs.p", "wb"))    
        print("Train retrieval done")
    
    def acc_helper(self,np_list,class_given): 
        correct=0
        for i in range(0,len(np_list)):
#            if class_given==0:
#                if np_list[i][0]>= np_list[i][1] and np_list[i][0]>= np_list[i][2]:
#                    correct = correct+1  
#            elif class_given==1:
#                if np_list[i][1]>= np_list[i][0] and np_list[i][1]>= np_list[i][2]:
#                    correct = correct+1
#            elif class_given==2:
#                if np_list[i][2]>= np_list[i][0] and np_list[i][2]>= np_list[i][1]:
#                    correct = correct+1
#                    
            if class_given==0:
                if np_list[i][0]>= np_list[i][1] and np_list[i][0]>= np_list[i][2]:
                    correct = correct+1  
            elif class_given==1:
                if np_list[i][1]> np_list[i][0] and np_list[i][1]>= np_list[i][2]:
                    correct = correct+1
            elif class_given==2:
                if np_list[i][2]> np_list[i][0] and np_list[i][2]> np_list[i][1]:
                    correct = correct+1
                    

        return  100*correct/len(np_list)
    
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


    def  clean_test_PDB_results(self,unigene_load_dir):
        '''
        This function remove the ambiguity PDBs to find the correct results without overlapping genes PDBs
        '''
        os.chdir('/')
        os.chdir(unigene_load_dir)
        OG_gene_overlap_PDBs=  pickle.load(open("ONGO_overlapped_gene_PDBs.p", "rb"))
        TSG_gene_overlap_PDBs=  pickle.load(open("TSG_overlapped_gene_PDBs.p", "rb"))
        
        for pdb in OG_gene_overlap_PDBs:
            if pdb in self.OG_test_PDBs:
                print(pdb,  " in OG TESt overlapped OG_gene_overlap_PDBs" )
            elif pdb in self.TSG_test_PDBs:
                print(pdb,  " in TSG TESt overlapped OG_gene_overlap_PDBs" )
            elif pdb in self.Fusion_test_PDBs:
                print(pdb,  " in Fusion TESt overlapped OG_gene_overlap_PDBs" )
                
                
        for pdb in TSG_gene_overlap_PDBs:
            if pdb in self.OG_test_PDBs:
                print(pdb,  " in OG TESt overlapped TSG_gene_overlap_PDBs" )
            elif pdb in self.TSG_test_PDBs:
                print(pdb,  " in TSG TESt overlapped TSG_gene_overlap_PDBs" )
            elif pdb in self.Fusion_test_PDBs:
                print(pdb,  " in Fusion TESt overlapped TSG_gene_overlap_PDBs" )
    #%%

