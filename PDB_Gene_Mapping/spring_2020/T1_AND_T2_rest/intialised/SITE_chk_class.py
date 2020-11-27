# -*- coding: utf-8 -*-
"""
Created on  %(26-Apr-2020) 5.03Pm

@author: %A.Nishanth C00294860

Little modified version 
%(09-Sep-2018) 6.07Pm
to check the SITE information to gether those PDBS satisfied Threhold condition
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
    
    def __init__(self,PDBs,checking_dir_PDB,threshold_length = 81):

        """
        This function take the 
        name: ONGO or TSG or Fusion
        threshold_length: PDBid has atleast this length in gene to be selected 
        And also check which PDBs has SITE information and save as pikle files for further proceeds
        
        Ids_info_all: THIS ONLY CONTAIIN X-RAY DATA
        """
   
        #% then go through each genes with PDB_ids and check howmany of them contain SITE information
        os.chdir("/")
        os.chdir(checking_dir_PDB)
        Experimental_type = []#to hold the way the SITE information gathered
        pdb_sat_t=[]
        for id_t in PDBs:
            if id_t[-3:]=='pdb':
                f = open(id_t)
    #            f = open( ''.join([id_t,".pdb"]))
    
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
                    for row in stack:
                        if row[0:25] == 'REMARK 800 EVIDENCE_CODE:':
                            Experimental_type.append(row[26:len(row)])
                            break
            else:
                print(id_t, " skipped")
        #% then count the number of PDB_ids has satisfiable PDB_info
        pickle.dump(pdb_sat_t, open("New_all_SITE_satisfied.p", "wb" ))      
        pickle.dump(Experimental_type, open("New_all_SITE_satisfied_experiment_type.p", "wb" ))