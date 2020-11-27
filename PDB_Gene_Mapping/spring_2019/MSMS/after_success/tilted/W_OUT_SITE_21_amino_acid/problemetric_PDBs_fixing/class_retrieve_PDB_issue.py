# -*- coding: utf-8 -*-
"""
Created on %17-Feb-2019(11.38 A.m)

@author: %A.Nishanth C00294860
"""
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth  import ResidueDepth
#from Bio.PDB import Selection
import pickle
import numpy as np
import csv 
import os

class msms_surface_class:
    """
    USE MSMS tool(1996) to calculte the soluable access area of the surface
        depth of the C_alpha carbons, and 
              of the residue 
                          from the surface
    
    """
    def __init__(self, pdb_source_dir, saving_dir, pdb_name):
        self.saving_dir = saving_dir
        self.pdb_name  = pdb_name #name of the class

        # first load the files needed
        os.chdir('/')
        os.chdir(pdb_source_dir)   
        print("PDB_id_working_on: ",pdb_name)
        pdb_name_open = ''.join([pdb_name,".pdb"])
        f = open(pdb_name_open)
        csv_f = csv.reader(f)
        stack=[]
        for row in csv_f:
            for s in row:
                stack.append(s)
        self.stack = stack     
    
    #% then match the C_aplha from our function coordinates to the depth and residue depth
    def retrieve_info(self):
        """
         then match the depth and residue depth to our C_aplha details
                                (such as: coordinates, amino_acid, MOTIF_indexs_all
        
         Anyway keya are already inorder however make sure every think fits
         some depth are calculated for Heta_atoms we don't consider them thus inorder
         to exclude that and 
             fit with details coordinates, aminoacids and MOTIF idexes 
        """
       
        ca_depth_all, res_depth_all,keys,surface = self.depth_calculator()
        coordinates,amino_acid,chain_id_info,resSeq,info_all_cordinates,atom_number_list_c = self.retrieve_c_alpha()
        
        fin_res_depth_all=[]
        fin_ca_depth_all=[]
        for i in range(0,len(coordinates)):
            for j in range(0,len(keys)):
                if resSeq[i].isdigit() and keys[j][0][0]==chain_id_info[i] and keys[j][1][1]==int(resSeq[i]) and keys[j][1][2]==' ':
                        fin_res_depth_all.append(res_depth_all[j])
                        fin_ca_depth_all.append(ca_depth_all[j])
                elif len(resSeq[i])==3 and resSeq[i][0:2].isdigit():
                    if keys[j][0][0]==chain_id_info[i] and keys[j][1][1]==int(resSeq[i][0:2]) and keys[j][1][2]== resSeq[i][2]:
                        fin_res_depth_all.append(res_depth_all[j])
                        fin_ca_depth_all.append(ca_depth_all[j])
                elif len(resSeq[i])==2 and resSeq[i][0:1].isdigit():
                    if keys[j][0][0]==chain_id_info[i] and keys[j][1][1]==int(resSeq[i][0:1]) and keys[j][1][2]== resSeq[i][1]:
                        fin_res_depth_all.append(res_depth_all[j])
                        fin_ca_depth_all.append(ca_depth_all[j])
                elif len(resSeq[i])>3 and keys[j][0][0]==chain_id_info[i] and keys[j][1][1]==int(resSeq[i][0:3]):
                        if len(resSeq[i])==4 and keys[j][1][2]== resSeq[i][3]:
                            fin_res_depth_all.append(res_depth_all[j])
                            fin_ca_depth_all.append(ca_depth_all[j])
                elif resSeq[i][1:len(resSeq[i])].isdigit() and keys[j][0][0]==chain_id_info[i] and keys[j][1][1]==int(resSeq[i]) and keys[j][1][2]==' ':
                    fin_res_depth_all.append(res_depth_all[j])
                    fin_ca_depth_all.append(ca_depth_all[j])
                            
        os.chdir('/')
        os.chdir(self.saving_dir)
        pickle.dump(coordinates, open( ''.join(["coordinates_",self.pdb_name,".p"]), "wb" ) ) 
        pickle.dump(amino_acid, open( ''.join(["amino_acid_",self.pdb_name,".p"]), "wb" ) ) 
        pickle.dump(fin_res_depth_all, open( ''.join(["fin_res_depth_all_",self.pdb_name,".p"]), "wb" ) ) 
        pickle.dump(fin_ca_depth_all, open( ''.join(["fin_ca_depth_all_",self.pdb_name,".p"]), "wb" ) ) 
        np.save(''.join(['surface_',self.pdb_name,'.npy']), surface) 
        return coordinates,amino_acid,fin_res_depth_all,fin_ca_depth_all,info_all_cordinates,atom_number_list_c

        
    def depth_calculator(self):
        """
        This function uses Biopython library to calculte the depth of the C_A, 
        and the residue depth from the surface calculted by the helper of MSMS tool
        """
        pdb_file = ''.join([self.pdb_name,".pdb"])
        parser = PDBParser()
        structure = parser.get_structure('i',pdb_file)
        model = structure[0]
        #residue_list = Selection.unfold_entities(model, 'R')#%%
        rd = ResidueDepth(model,pdb_file)
        
        keys=rd.keys()# first get the keys
        res_depth_all=[]
        ca_depth_all=[]
        for i in range(0,len(keys)):
            residue_depth, ca_depth=rd[keys[i]]#calculate the depth of the Caplha atom from the surface
            res_depth_all.append(residue_depth)
            ca_depth_all.append(ca_depth)
        
        surface=rd.surface
        return ca_depth_all, res_depth_all, keys,surface


    def retrieve_c_alpha(self):
        
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
        amino_acid = []
        
        atom_number = 0
        atom_number_list_c =[]
        for row in stack:
            temp = row.split()
            if row[0:21] == 'REMARK   2 RESOLUTION':
                #to access the resolution details
                resolution_PDB = float(temp[3])
            if row[0:4] == "ATOM":
                atom_number = atom_number + 1
                info_all_cordinates.append(row)
                if row[13:15] == "CA": 
                    atom_number_list_c.append(atom_number)
                    # to make normalised coordinates without affected buy the resolution
                    coordinates.append([float(row[30:38])/resolution_PDB,float(row[38:46])/resolution_PDB,float(row[46:54])/resolution_PDB])
                    #to access the chain ID and residue information to check the SITE info annotation of alpha carbon
                    residue_info.append(row[17:20].split()[0])
                    amino_acid.append(self.map_PDB_res_simple_amino_acid_format(row[17:20]))
                    chain_id_info.append(row[21])
                    resSeq.append(row[22:27].split()[0]) # since some of the reduncdant there after
                if len(row)>=27:
                    atom_all_residues.append(row[17:20].split()[0])
                    atom_all_chains.append(row[21])
                    atom_all_res_seq.append(row[22:27].split()[0]) 
           
            elif row[0:6] == "HETATM":
                un_wanted_site.append(row[17:20].split()[0])    
        
        return coordinates,amino_acid,chain_id_info,resSeq,info_all_cordinates,atom_number_list_c

        
    """ Helper functions"""    
    def map_PDB_res_simple_amino_acid_format(self,given):
        """
        residue or aminoacid
        """
        pdb_residue_format = ["ARG","LYS","ASP","GLU","GLN","ASN","HIS","SER","THR","TYR",
                              "CYS","MET","TRP","ALA", "ILE","LEU","PHE","VAL","PRO","GLY","SEC"]
        aminoacid_format = ["R","K","D","E","Q","N","H","S","T","Y",
                                "C","M","W","A","I","L","F","V","P","G","U"]
        for i in range(0,len(pdb_residue_format)):
            if pdb_residue_format[i] == given:
                return aminoacid_format[i]     
            
        
