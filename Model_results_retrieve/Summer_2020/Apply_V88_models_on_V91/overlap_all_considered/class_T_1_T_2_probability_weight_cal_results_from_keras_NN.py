# -*- coding: utf-8 -*-
"""
Created on %(03-Sep-2018) 13.35Pm
Editted on 19-May-2020 at 4.22p.m to fit with dictionary of Probability
@author: %A.Nishanth C00294860

updated to match with Keras probabilities from pikles form on 02-02-2020 at 9.25 A.m
"""
import os
import pickle
from copy import deepcopy
import numpy as np

class probability_weight_cal_results_from_keras_NN_for_T_1_T_2:
    """
    This class calculates the probabilities from the pikle results created by 
    Deeplearning model(results represented change should be fisrt coloumn only contain PDB_names)
  
    Eg: Protein
        1A31
        1A35
    
    And the results created by 
    class_probabilty_weigtht_initialize_gene_from_PDB_seq_length
    """
    
    def __init__(self,test_gene_list,test_prob_dic, working_dir_fraction,saving_dir,name, minimum_prob_decide= 0.25,Test_set=True,straight_save=True,return_out=False):
        
        self.working_dir_fraction = working_dir_fraction # to load the results of fractional results
        self.minimum_prob_decide = minimum_prob_decide
        self.name = name
        self.Test_set = Test_set
        self.return_out=return_out

        # to save the results
        if straight_save:
            self.saving_dir=saving_dir
        else:
            if Test_set:
                self.saving_dir = "".join([saving_dir,'/test/'])
            else:
                self.saving_dir = "".join([saving_dir,'/train/'])

        # Assign the deeplearning model results
        self.test_gene_list = test_gene_list
        self.test_prob_dic = test_prob_dic
    
    def load_method_results(self):  
        """
        This is the main function difference from the 
        class_probabilty_weigtht_cal_results_from_NN_test
        to
        class_probabilty_weigtht_cal_results_from_NN
        
        Here this function only take the genes in the pikle of test files
        """
        name = self.name
        # to read the 
        test_gene_list=self.test_gene_list
        #% then read the fraction values for that class
        os.chdir('/')
        os.chdir(self.working_dir_fraction)
        # first choose the direct hit for the Genes with PDB's and findout the probabilities
        weight_initialize_gene_from_PDB_seq_length = os.listdir()
        
        direct_pikles_UniGene = []
        method_pikles_UniGene = []

        direct_pikles = []
        method_1_pikles = []
        method_2_pikles = []
        method_3_pikles = []
        for n in weight_initialize_gene_from_PDB_seq_length: 
            if n[len(name)+1:len(name)+7] == 'direct':
                direct_pikles.append(n)
#                if name == "ONGO":
#                   direct_pikles_UniGene.append(n[12:-2])
#                elif name == "TSG":
#                    direct_pikles_UniGene.append(n[11:-2])
#                elif name == "Fusion":
#                    direct_pikles_UniGene.append(n[14:-2])
                direct_pikles_UniGene.append(n[len(name)+8:-2])

            elif n[len(name)+1:len(name)+9] == 'method_1':
                method_1_pikles.append(n)
            elif n[len(name)+1:len(name)+9] == 'method_2':
                method_2_pikles.append(n)
            elif n[len(name)+1:len(name)+9] == 'method_3':
                method_pikles_UniGene.append(n[10+len(name):-2])
                method_3_pikles.append(n)
#                if name == "ONGO":
#                    method_pikles_UniGene.append(n[14:-2])
#                elif name == "TSG":
#                    method_pikles_UniGene.append(n[13:-2])
#                elif name == "Fusion":
#                    method_pikles_UniGene.append(n[16:-2])
                    
        wanted_direct_pikles = []
        wanted_method_1_pikles = []
        wanted_method_2_pikles = []
        wanted_method_3_pikles = []
        for i in range(0,len(method_pikles_UniGene)):
            if method_pikles_UniGene[i] in test_gene_list:
                wanted_method_1_pikles.append(method_1_pikles[i])
                wanted_method_2_pikles.append(method_2_pikles[i])
                wanted_method_3_pikles.append(method_3_pikles[i])
        for i in range(0,len(direct_pikles_UniGene)):
            if direct_pikles_UniGene[i] in test_gene_list:        
                wanted_direct_pikles.append(direct_pikles[i])
                
        return wanted_direct_pikles, wanted_method_1_pikles, wanted_method_2_pikles, wanted_method_3_pikles         
   

    def probability_from_method(self,method_name):
        """
        Calculate the probability for the genes only has maore than one PDBs
        This function caculate the final probability for give pikle set
        """
        os.chdir('/')
        os.chdir(self.working_dir_fraction)
        whole = pickle.load(open(method_name,"rb"))
        # go through all available coverages
        # and calculate the overall fraction first for all PDBs
        pdb_temp = []
        sum_length = 0
        for j in range(0,len(whole[1])):
            sum_length = sum_length + whole[1][j]
            for k in range(0,len(whole[3])):
                #intially needed to append the fractions 
                if j==0:
                    pdb_temp.append(whole[1][j]*whole[2][j][k])
                else:
                    pdb_temp[k] = pdb_temp[k] + whole[1][j]*whole[2][j][k]
         
        for k in range(0,len(whole[3])):            
            pdb_temp[k] = pdb_temp[k]/sum_length
            
        # then gothrough probabilities and caculate the final
        p_OG = 0
        p_TSG = 0
        p_Fusion = 0
        for k in range(0,len(whole[3])): 
            if pdb_temp[k] > 0:
                    p_OG = p_OG + pdb_temp[k]*self.test_prob_dic[whole[3][k]][0]
                    p_TSG = p_TSG + pdb_temp[k]*self.test_prob_dic[whole[3][k]][1]
                    p_Fusion = p_Fusion + pdb_temp[k]*self.test_prob_dic[whole[3][k]][2]
                
        # finalize the probabilities
        prob = [p_OG/(p_OG + p_TSG + p_Fusion), p_TSG/(p_OG + p_TSG + p_Fusion),  p_Fusion/(p_OG + p_TSG + p_Fusion)]
        return  prob, whole[0]

    def calculate_prob(self):                
        """
        This function finaly calcuate the probabilities
        """

        direct_pikles, method_1_pikles, method_2_pikles, method_3_pikles = self.load_method_results()  
        
        #% calculate the probability for the genes only has one PDBs
        gene_direct_prob = []
        for i in range(0,len(direct_pikles)):
            dir_1 = pickle.load(open(direct_pikles[i], "rb" ))
            gene_direct_prob.append([dir_1[0], self.test_prob_dic[dir_1[1][0]]])
          
        # start with save the finalised details with gene name has more than one PDB
        fin_gene_prob = []
        Ensamble_gene_prob = []
        for i in range(0,len(method_1_pikles)):
            print("running pikle: ",method_1_pikles[i])
            prob_1, gene_name_1 = self.probability_from_method(method_1_pikles[i])
            prob_2, gene_name_2 = self.probability_from_method(method_2_pikles[i])
            prob_3, gene_name_3 = self.probability_from_method(method_3_pikles[i])
            prob_ens = [(prob_1[0] + prob_2[0] + prob_3[0])/3, (prob_1[1] + prob_2[1] + prob_3[1])/3, (prob_1[2] + prob_2[2] + prob_3[2])/3]
            if gene_name_1 == gene_name_2 == gene_name_3:
                 fin_gene_prob.append([gene_name_1, prob_1, prob_2, prob_3])
                 Ensamble_gene_prob.append([gene_name_1, prob_ens])
                 
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)
        os.chdir('/')
        os.chdir(self.saving_dir)
        pickle.dump(Ensamble_gene_prob, open(''.join([self.name,"_ensamble_gene_name_m_1_2_3_prob.p"]), "wb"))  
        pickle.dump(fin_gene_prob, open(''.join([self.name,"_fin_gene_name_m_1_2_3_prob.p"]), "wb"))  
        
        return  gene_direct_prob, fin_gene_prob, Ensamble_gene_prob 

    def decider_prob(self,f,ids, minimum_prob_decide,i):
        """
        Helper function for documentation for decide calss
        """ 
        if ids[i][0] >= minimum_prob_decide and ids[i][1] >= minimum_prob_decide and ids[i][2] >= minimum_prob_decide:
           f.write( '\t' +', '+ "ONGO_TSG_Fusion" + '\n') 
           return "ONGO_TSG_Fusion"
        elif ids[i][0] >= minimum_prob_decide and ids[i][1] >= minimum_prob_decide and ids[i][2] < minimum_prob_decide:
           f.write( '\t' +', '+ "ONGO_TSG" + '\n') 
           return "ONGO_TSG"
        elif ids[i][0] >= minimum_prob_decide and ids[i][1] < minimum_prob_decide and ids[i][2] >= minimum_prob_decide:
           f.write( '\t' +', '+ "ONGO_Fusion" + '\n') 
           return  "ONGO_Fusion"
        elif ids[i][0] < minimum_prob_decide and ids[i][1] >= minimum_prob_decide and ids[i][2] >= minimum_prob_decide:
           f.write( '\t' +', '+ "TSG_Fusion" + '\n') 
           return "TSG_Fusion"
        elif ids[i][0] >= minimum_prob_decide*2:
           f.write( '\t' +', '+ "ONGO" + '\n') 
           return "ONGO" 
        elif ids[i][1] >= minimum_prob_decide*2:
           f.write( '\t' +', '+ "TSG" + '\n')  
           return "TSG"
        elif ids[i][2] >= minimum_prob_decide*2:
           f.write( '\t' +', '+ "Fusion" + '\n')  
           return "Fusion"
        else:
           print(" ids[i][2]: ", ids[i][2])
           raise ValueError("You needed to change the minimum_prob_decide because it doesn't satisfy all cases")

    def documentation_prob(self):
        """
        This function documents the probabilities
        """
        name = self.name
        minimum_prob_decide = self.minimum_prob_decide     
        gene_direct_prob, fin_gene_prob, Ensamble_gene_prob = self.calculate_prob()
  
        count_direct = 0
    
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir)
        os.chdir('/')
        os.chdir(self.saving_dir)

        
        f= open(''.join([name,'_results_gene_overlap_all_methods.txt']),"w+")
        f.write('\t\t'+'Probabilities of direct pikles'+'\n')
        f.write('Uni_gene_ID'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + ',Class' + '\n')
        
        for ids in gene_direct_prob:
            #insert the group details
            f.write(ids[0] + '\t' +',' + str(round(ids[1][0],2))  + '\t' +',' + str(round(ids[1][1],2)) + '\t' +',' + str(round(ids[1][2],2)))
            predicted = self.decider_prob(f, ids, minimum_prob_decide,1)
            if predicted == name:
                count_direct = count_direct + 1
                
        f.write('\n')
#        f.write('\t--//////////-----------------------------------************-------------------------------------///////----------\t'+'\n')
        f.write('\n')
        f.write('\t\t'+'Probabilities from Ensamble'+'\n')
        f.write('Uni_gene_ID'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + ',Class' + '\n')
        count_ensamble = 0
        for ids in Ensamble_gene_prob:
            #insert the group details
            f.write(ids[0] + '\t' +',' + str(round(ids[1][0],2))  + '\t' +',' + str(round(ids[1][1],2)) + '\t' +',' + str(round(ids[1][2],2)))
            predicted = self.decider_prob(f, ids, minimum_prob_decide,1)
            if predicted == name:
               count_ensamble  = count_ensamble  + 1
        #print("Done")
        f.close() 
        # for method_1
        count_method_1 = 0
        f= open(''.join([name,'_supplimentery_results_gene_overlap_method_1.txt']),"w+")
        f.write('\t\t'+'Probabilities of Method_1 pikles'+'\n')
        f.write('Uni_gene_ID'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + ',Class' + '\n')
        for ids in fin_gene_prob:
            #insert the group details
            f.write(ids[0] + '\t' +',' + str(round(ids[1][0],2))  + '\t' +',' + str(round(ids[1][1],2)) + '\t' +',' + str(round(ids[1][2],2)))
            predicted = self.decider_prob(f, ids, minimum_prob_decide,1)
            if predicted == name:
               count_method_1  = count_method_1  + 1
        f.close() 
        
        # for method_2
        count_method_2 = 0
        f= open(''.join([name,'_supplimentery_results_gene_overlap_method_2.txt']),"w+")
        f.write('\t\t'+'Probabilities of Method_2 pikles'+'\n')
        f.write('Uni_gene_ID'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + ',Class' + '\n')
        for ids in fin_gene_prob:
            #insert the group details
            f.write(ids[0] + '\t' +',' + str(round(ids[2][0],2))  + '\t' +',' + str(round(ids[2][1],2)) + '\t' +',' + str(round(ids[2][2],2)))
            predicted = self.decider_prob(f, ids, minimum_prob_decide,2)
            if predicted == name:
               count_method_2  = count_method_2  + 1
        f.close() 
        
        # for method_3
        count_method_3 = 0
        f= open(''.join([name,'_supplimentery_results_gene_overlap_method_3.txt']),"w+")
        f.write('\t\t'+'Probabilities of Method_3 pikles'+'\n')
        f.write('Uni_gene_ID'+'\t'+', OG'+'\t'+ ', TSG'+ '\t'+ ', Fusion'+ '\t' + ',Class' + '\n')
        for ids in fin_gene_prob:
            #insert the group details
            f.write(ids[0] + '\t' +',' + str(round(ids[3][0],2))  + '\t' +',' + str(round(ids[3][1],2)) + '\t' +',' + str(round(ids[3][2],2)))
            predicted = self.decider_prob(f, ids, minimum_prob_decide,3)
            if predicted == name:
               count_method_3  = count_method_3  + 1
        f.close() 
        
        if self.return_out:
           re_Ensamble_gene_prob = np.empty((len(Ensamble_gene_prob),3))
           re_gene_direct_prob= np.empty((len(gene_direct_prob),3))
           re_fin_gene_prob= np.empty((3,len(fin_gene_prob),3))   
           for i in range(0,len(Ensamble_gene_prob)):
               re_Ensamble_gene_prob[i,:]=deepcopy(Ensamble_gene_prob[i][1])
               
           for i in range(0,len(gene_direct_prob)):
               re_gene_direct_prob[i,:]=deepcopy(gene_direct_prob[i][1])
             
           for i in range(0,len(fin_gene_prob)):
               re_fin_gene_prob[0,i,:]=deepcopy(fin_gene_prob[i][1])
               re_fin_gene_prob[1,i,:]=deepcopy(fin_gene_prob[i][2])
               re_fin_gene_prob[2,i,:]=deepcopy(fin_gene_prob[i][3])

           return  re_gene_direct_prob, re_fin_gene_prob, re_Ensamble_gene_prob 
       
            