# -*- coding: utf-8 -*-
"""
Created on %(29-Nov-2018) 12.11 A.m

@author: %A.Nishanth C00294860
"""
import os 
import csv
import copy

class site_info_vin_1:
    """
    To produce the results of surface alpha carbon properties from the given pdb_file
    """
    def __init__(self,pdb_source_dir, pdb_name):
#        self.saving_dir = saving_dir
        self.pdb_name  = pdb_name #name of the class
#        self.saving_dir = saving_dir
        # first load the files needed
        os.chdir('/')
        os.chdir(pdb_source_dir)   
#        print("PDB_id_working_on: ",pdb_name)
        pdb_name_open = ''.join([pdb_name,".pdb"])
        f = open(pdb_name_open)
        csv_f = csv.reader(f)
        stack=[]
        for row in csv_f:
            for s in row:
                stack.append(s)
        self.stack = stack     
            
    def retrieve_c_alpha(self):
        """
        to access the coordinate information of c-alpha carbon
        
        returns
        coordinate: normalised coordinates of C_Alpha
        amino_acid: amino acids for the C_Alpha back bone
        
        """
        stack = self.stack
        info_all_cordinates = []
        coordinates = []
        residue_info = []
        resSeq = []      # Residue sequence number
        chain_id_info = []
        
        # site unewanted_ini
        un_wanted_site = []
        atom_all_residues = []
        atom_all_chains = []
        atom_all_res_seq = []
        #% to access the given SITE information from PDB 
        res_site_temp = []
        chain_site_temp = []
        res_seq_site_temp = [] 
        amino_acid = []
        site_occurance_count = []
        SITE_id_info = []
        resolution_PDB  = 1.0
        #        main_site_cond = True
        atom_number = 0
        atom_number_list_c =[]
        for row in stack:
            temp = row.split()
#            if row[0:21] == 'REMARK   2 RESOLUTION':
#                #to access the resolution details
#                resolution_PDB = float(temp[3])
        #                print("Resolution: ", float(temp[3]))
            if row[0:4] == "ATOM":
                atom_number = atom_number + 1
                info_all_cordinates.append(row)
                if row[13:15] == "CA": 
                    atom_number_list_c.append(atom_number)
                    # to make normalised coordinates without affected buy the resolution
                    coordinates.append([float(row[30:38])/resolution_PDB,float(row[38:46])/resolution_PDB,float(row[46:54])/resolution_PDB])
                    #to access the chain ID and residue information to check the SITE info annotation of alpha carbon
                    residue_info.append(row[17:20].split()[0])
#                    amino_acid.append(self.map_PDB_res_simple_amino_acid_format(row[17:20]))
                    chain_id_info.append(row[21])
#                    resSeq.append(row[22:26].split()[0])
                    resSeq.append(row[22:27].split()[0]) # since some of the reduncdant there after
                if len(row)>=27:
                    atom_all_residues.append(row[17:20].split()[0])
                    atom_all_chains.append(row[21])
    #                atom_all_res_seq.append(row[22:26].split()[0])

                    atom_all_res_seq.append(row[22:27].split()[0]) 
            elif row[0:4] == "SITE":    
                # to check the SITE_id
                if temp[2] not in SITE_id_info:
                    SITE_id_info.append(temp[2])
                    site_occurance_count.append(int(temp[3]))
                """ This information is useeful later when you need ed to find the CA with SITE info"""   
                res_site_t, chain_site_t, res_seq_site_t = self.helper_func_retrieve_site_info(row[18:len(row)])
                res_site_temp.append(res_site_t)
                chain_site_temp.append(chain_site_t)
                res_seq_site_temp.append(res_seq_site_t) 
                
            elif row[0:6] == "HETATM":
                un_wanted_site.append(row[17:20].split()[0])    

        res_site = []
        chain_site = []
        res_seq_site = []  
        index_count_res = []
        for i in range(0,len(res_site_temp)):
            index_count_res.append(len(res_site_temp[i]))
            for j in range(0,len(res_site_temp[i])):
                res_site.append(res_site_temp[i][j])
                chain_site.append(chain_site_temp[i][j])
                res_seq_site.append(res_seq_site_temp[i][j])
       
        #% then go through the RESIDUES in CA and check missed residues
        combined_CA_residue_all = []
        for i in range(0,len(residue_info)):
            combined_CA_residue_all.append(''.join([residue_info[i],"_",chain_id_info[i],"_",resSeq[i]]))
        
        combined_residue_all = []
        for i in range(0,len(atom_all_residues)):
            combined_residue_all.append(''.join([atom_all_residues[i],"_",atom_all_chains[i],"_",atom_all_res_seq[i]]))
        
        residue_not_in_c_alpha = [v for v in combined_residue_all if v not in combined_CA_residue_all]         
        # atoms indexes of the C-alpha satisified SITE conditions
        c_alpha_indexes_MOTIF = []
        un_sat_residues = []
        count_un_wanted = 0
        more_SITE_occurance = 0
        for i in range(0,len(res_site)):   
            occurance = 0
            un_satisfied = True     
            for j in range(0,len(coordinates)):
                if (res_site[i] == residue_info[j]) and chain_site[i] == chain_id_info[j] and res_seq_site[i] == resSeq[j]:
        #            if occurance > 0:
        #                print("Occured_twice            : ",j)
        #                print("prevously_occured_alpha_C: ",c_alpha_indexes_MOTIF[-1])
                    c_alpha_indexes_MOTIF.append(j)
                    occurance = occurance +1 
                    un_satisfied = False
            if occurance > 1: 
#                print("Occurance: ",occurance)
                more_SITE_occurance = more_SITE_occurance + occurance-1
            if un_satisfied:
                un_sat_residues.append(i)
                if res_site[i] in un_wanted_site: 
                    count_un_wanted = count_un_wanted + 1
                elif "".join([res_site[i],"_",chain_site[i],"_",res_seq_site[i]]) in residue_not_in_c_alpha:
                    count_un_wanted = count_un_wanted + 1
        #        print("")
        #        print(res_site[i])
        #        print(chain_site[i])
        #        print(res_seq_site[i])
        if len(res_site) == len(c_alpha_indexes_MOTIF) + count_un_wanted  - more_SITE_occurance:
            """In order to group the MOTIFs"""
            c_alpha_indexes_MOTIF_grouped = []
            for i in range(0,len(res_site_temp)):
                c_alpha_indexes_MOTIF_grouped_temp = []
                for j in range(0,len(res_site_temp[i])):
                    for k in range(0,len(coordinates)):
                        if (res_site_temp[i][j] == residue_info[k]) and chain_site_temp[i][j] == chain_id_info[k] and res_seq_site_temp[i][j] == resSeq[k]:
                            c_alpha_indexes_MOTIF_grouped_temp.append(k)
                c_alpha_indexes_MOTIF_grouped.append(c_alpha_indexes_MOTIF_grouped_temp)
            print("perfect :)")    
            return coordinates, amino_acid, c_alpha_indexes_MOTIF,c_alpha_indexes_MOTIF_grouped
        else:
            print("PDB id not satisfied: ",self.pdb_name)
            print(" # res_site             : ", len(res_site))
            print(" count_un_wanted        : ", count_un_wanted)
            print(" # c_alpha_indexes_MOTIF: ", len(c_alpha_indexes_MOTIF) )
            print("more_SITE_occurance     : ",more_SITE_occurance)
            raise NameError('PDB_id not satisfied')
           

    # SITE information functions
    def map_PDB_res_simple_amino_acid_format(self,given):
        """
        residue or aminoacid
        """
        pdb_residue_format = ["ARG","LYS","ASP","GLU","GLN","ASN","HIS","SER","THR","TYR",
                              "CYS","MET","TRP","ALA", "ILE","LEU","PHE","VAL","PRO","GLY"]
        aminoacid_format = ["R","K","D","E","Q","N","H","S","T","Y",
                                "C","M","W","A","I","L","F","V","P","G"]
        for i in range(0,len(pdb_residue_format)):
            if pdb_residue_format[i] == given:
                return aminoacid_format[i]

    def helper_func_retrieve_site_info(self,whole):  
        """
        Where the whole has the information of one row of given PDB's SITEs 
        eg: PDB file has like this
                "SITE     7 AC2 28 HOH A 175  HOH A 186  HOH A 188  HOH A 289           "
            then the whole contains
                "HOH A 175  HOH A 186  HOH A 188  HOH A 289           "
                
        This function uses the space to seperate the information from the data
        """      
        res_site_t = []
        chain_site_t = []
        res_seq_site_t = []
        """ check the whole code again"""
        chk = whole.split()
#        print(chk)
#        print(len(chk))
        i = 0
        while i < len(chk)-1: 
             res_site_t.append(chk[i])
             if len(chk[i+1]) > 1:
                 chain_site_t.append(chk[i+1][0])
                 res_seq_site_t.append(''.join([chk[i+1][1:len(chk[i+1])]]))  
                 i = i + 2
             else:
                 chain_site_t.append(chk[i+1])
                 res_seq_site_t.append(chk[i+2])        
                 i = i + 3
        return res_site_t, chain_site_t, res_seq_site_t