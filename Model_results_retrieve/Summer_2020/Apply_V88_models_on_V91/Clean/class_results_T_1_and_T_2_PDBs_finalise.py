# -*- coding: utf-8 -*-
"""
Created on %28-05-2020 at 4.08pm

@author: %A.Nishanth C00294860
"""
import os


class results_T_1_and_T_2_PDBs_finalise:
    """
    This class finalise the document the PDBs' probabilities
    """
    
    def __init__(self,T1_T2_test_prob_dict, saving_dir,minimum_prob_decide= 0.25,Test_set=True,straight_save=True,return_out=False):
        
        self.minimum_prob_decide = minimum_prob_decide
        self.T1_T2_test_prob_dict=T1_T2_test_prob_dict
        
        if straight_save:
            self.saving_dir=saving_dir
        else:
            if Test_set:
                self.saving_dir = "".join([saving_dir,'/test/'])
            else:
                self.saving_dir = "".join([saving_dir,'/train/'])
                
    def decider_prob(self,f, pdb):
            """
            Helper function for documentation for decide calss
            """ 
            minimum_prob_decide= self.minimum_prob_decide 
            T1_T2_test_prob_dic=self.T1_T2_test_prob_dict
            if T1_T2_test_prob_dic[pdb][0] >= minimum_prob_decide and T1_T2_test_prob_dic[pdb][1] >= minimum_prob_decide and T1_T2_test_prob_dic[pdb][2] >= minimum_prob_decide:
               f.write( '\t' +', '+ "ONGO_TSG_Fusion" + '\n') 
               return "ONGO_TSG_Fusion"
            elif T1_T2_test_prob_dic[pdb][0] >= minimum_prob_decide and T1_T2_test_prob_dic[pdb][1] >= minimum_prob_decide and T1_T2_test_prob_dic[pdb][2] < minimum_prob_decide:
               f.write( '\t' +', '+ "ONGO_TSG" + '\n') 
               return "ONGO_TSG"
            elif T1_T2_test_prob_dic[pdb][0] >= minimum_prob_decide and T1_T2_test_prob_dic[pdb][1] < minimum_prob_decide and T1_T2_test_prob_dic[pdb][2] >= minimum_prob_decide:
               f.write( '\t' +', '+ "ONGO_Fusion" + '\n') 
               return  "ONGO_Fusion"
            elif T1_T2_test_prob_dic[pdb][0] < minimum_prob_decide and T1_T2_test_prob_dic[pdb][1] >= minimum_prob_decide and T1_T2_test_prob_dic[pdb][2] >= minimum_prob_decide:
               f.write( '\t' +', '+ "TSG_Fusion" + '\n') 
               return "TSG_Fusion"
            elif T1_T2_test_prob_dic[pdb][0] >= minimum_prob_decide*2:
               f.write( '\t' +', '+ "ONGO" + '\n') 
               return "ONGO" 
            elif T1_T2_test_prob_dic[pdb][1] >= minimum_prob_decide*2:
               f.write( '\t' +', '+ "TSG" + '\n')  
               return "TSG"
            elif T1_T2_test_prob_dic[pdb][2] >= minimum_prob_decide*2:
               f.write( '\t' +', '+ "Fusion" + '\n')  
               return "Fusion"
            else:
               print(" T1_T2_test_prob_dic[pdb][2]: ", T1_T2_test_prob_dic[pdb][2])
               raise ValueError("You needed to change the minimum_prob_decide because it doesn't satisfy all cases")

    def documentation_prob_all(self):
            """
            This function documents the probabilities
            """
            if not os.path.exists(self.saving_dir):
                os.makedirs(self.saving_dir)
            os.chdir('/')
            os.chdir(self.saving_dir)
            pdbs=self.T1_T2_test_prob_dict.keys() 

            
            f= open('T_1_T_2_PDBs_all.txt',"w+")
            f.write('\t\t'+'Probabilities of direct pikles'+'\n')
            f.write('PDBs'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + 'Class' + '\n')
            
            for pdb in pdbs:
                #insert the group details
                f.write(pdb + '\t' +',' + str(round(self.T1_T2_test_prob_dict[pdb][0],2))  + '\t' +',' + str(round(self.T1_T2_test_prob_dict[pdb][1],2)) + '\t' +',' + str(round(self.T1_T2_test_prob_dict[pdb][2],2)))
                _ = self.decider_prob(f, pdb)
                    
            f.close() 
            
    def documentation_prob_only_given(self,name,pdbs):
        """
        This function documents the probabilities given PDB lists
        """
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)
        os.chdir('/')
        os.chdir(self.saving_dir)
        
        f= open(''.join([name,'_T_1_PDBs_clean.txt']),"w+")
        f.write('\t\t'+'Probabilities of direct pikles'+'\n')
        f.write('PDBs'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + 'Class' + '\n')
        
        for pdb in pdbs:
            #insert the group details
            f.write(pdb + '\t' +',' + str(round(self.T1_T2_test_prob_dict[pdb][0],2))  + '\t' +',' + str(round(self.T1_T2_test_prob_dict[pdb][1],2)) + '\t' +',' + str(round(self.T1_T2_test_prob_dict[pdb][2],2)))
            _ = self.decider_prob(f, pdb)
                
        f.close() 