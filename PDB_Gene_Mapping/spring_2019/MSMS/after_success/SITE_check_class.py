# -*- coding: utf-8 -*-
"""
Created on  %(09-Sep-2018) 6.07Pm

@author: %A.Nishanth C00294860
"""
import os
import pickle
import csv

class SITE_check:
    """
    This class is basically written for create the pikle files for 
    length and Gene details of the geiven group of class
    
    Further check afterwards howmany of them(PDB_files) has SITE information
    """
    
    def __init__(self,working_dir,checking_dir_PDB,saving_dir,name, threshold_length = 81):
        self.working_dir = working_dir # this is used to load up the PDB_ids that satisfied PDB_ids
        self.checking_dir_PDB =  checking_dir_PDB # to access the PDB_ids
        self.saving_dir = saving_dir
        self.name = name #name of the class
        self.threshold_length = threshold_length# to check the PDB_id how far overlapped with gene

    def selecting_PDBs(self):
        """
        This function take the 
        name: ONGO or TSG or Fusion
        threshold_length: PDBid has atleast this length in gene to be selected 
        And also check which PDBs has SITE information and save as pikle files for further proceeds
        
        Ids_info_all: THIS ONLY CONTAIIN X-RAY DATA
        """
        name = self.name
        os.chdir('/')
        os.chdir(self.working_dir)
        #    omitted = pickle.load(open( ''.join([name,"_omitted.p"]), "rb" ) )
        Ids_info_all = pickle.load(open( ''.join([name,"_Ids_info_all.p"]), "rb" ) )
        uni_gene = pickle.load(open( ''.join([name,"_uni_gene.p"]), "rb" ) )
        
        PDBs_sel=[]
        start_end_position = []
        for ids in Ids_info_all:
            PDB_t =[]
            start_end_position_t =[]
            for id_t in ids:
                if len(id_t) == 3:
        #            resolution.append(float(id_t[1]))
                    for j in range(0,len(id_t[2])):
                            if id_t[2][j]=="-":
                               if (int(id_t[2][j+1:len(id_t[2])])-int(id_t[2][0:j]))+1 > self.threshold_length:
                                   PDB_t.append(id_t[0])
                                   start_end_position_t.append([int(id_t[2][0:j]),int(id_t[2][j+1:len(id_t[2])])])
            PDBs_sel.append(PDB_t)
            start_end_position.append(start_end_position_t)
            
        #%  after threshold satisfication tested the select gene and PDB details
        fin_PDB_ids = [] # finalised_PDB_ids 
        fin_PDB_ids_start_end_position = [] # finalised_PDB_ids_start_end_position
        fin_uni_gene = [] # finalised_uni_gene 
        for i in range(0,len(PDBs_sel)): 
            if len(PDBs_sel[i]) > 0:   
                fin_PDB_ids.append(PDBs_sel[i])
                fin_PDB_ids_start_end_position.append(start_end_position[i])
                fin_uni_gene.append(uni_gene[i])  
        #% then go through each genes with PDB_ids and check howmany of them contain SITE information
        os.chdir("/")
        os.chdir(self.checking_dir_PDB)
        gene_sat = []
        Experimental_type = []#to hold the way the SITE information gathered
        for id_list in fin_PDB_ids:
            pdb_sat_t =[]
            Experimental_type_t = []
            for id_t in id_list:
                f = open( ''.join([id_t,".pdb"]))
                csv_f = csv.reader(f)
                cond_sat = False
                stack=[]
                for row in csv_f:
                    for s in row:
                        stack.append(s)
                        if row[0][0:4] =="SITE":
                            cond_sat = True
                # then load the SITE details
                if cond_sat:
                    pdb_sat_t.append(id_t)
                    Experimental_type_temp = []
                    for row in stack:
                        if row[0:25] == 'REMARK 800 EVIDENCE_CODE:':
                            Experimental_type_temp.append(row[26:len(row)])
                    Experimental_type_t.append(Experimental_type_temp)
            gene_sat.append(pdb_sat_t)
            Experimental_type.append(Experimental_type_t)
        #% then count the number of PDB_ids has satisfiable PDB_info
        count = 0
        count_gene = 0
        for id_list in gene_sat:
            if len(id_list) > 0:
                count_gene = count_gene  + 1
                count = count + len(id_list)
        
        total_count = 0
        for id_list in fin_PDB_ids:        
             if len(id_list) > 0:
                total_count = total_count + len(id_list)
        print("# of genes in the ",self.name, "                                     : " , len(fin_PDB_ids))  
        print("# of genes atleast has one PDB_id that satisfing having SITE info: ", count_gene)        
        print("Total number of PDB_s satisfied thresh hold condition            : ", total_count)
        print("count of pdbs totally satisfied having SITE info                 : ", count)
        print("")  
        print("-------------///////////////////////////--------------------------------")
        print("")  
        os.chdir("/")
        os.chdir(self.saving_dir)
        pickle.dump(fin_PDB_ids, open( ''.join([self.name,"_gene_list_thres_sat_PDB_ids.p"]), "wb" ))    
        pickle.dump(fin_uni_gene, open( ''.join([self.name,"_thres_sat_gene_list.p"]), "wb" ))    
        pickle.dump(gene_sat, open( ''.join([self.name,"_SITE_satisfied.p"]), "wb" ))      
        pickle.dump(Experimental_type, open( ''.join([self.name,"_SITE_satisfied_experiment_type.p"]), "wb" ))