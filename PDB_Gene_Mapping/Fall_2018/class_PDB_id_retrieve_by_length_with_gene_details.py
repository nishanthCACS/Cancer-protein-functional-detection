# -*- coding: utf-8 -*-
"""
Created on %(26-Aug-2018) 2.06Pm

@author: %A.Nishanth C00294860
"""
import os
import pickle

class pdb_id_retrive_length_and_gene_details:
    """
    This class is basically written for create the pikle files for 
    length and Gene details of the geiven group of class
    """
    
    def __init__(self,working_dir,saving_dir,name):
        self.working_dir=working_dir
        self.saving_dir = saving_dir
        self.name = name #name of the class


    def load_main_file(self):
        """
        first read the lines from the PDBid results
        get the basic details
        """
        os.chdir('/')
        os.chdir(self.working_dir)
        with open(''.join([self.name,"_PDB_ids_with_length.txt"]), "r") as ins:
            array = []
            for line in ins:
                array.append(line)
               
        #% Then splkit the PDBids with info
        chk = 0
        whole=[]
        temp=[]
        Uni_gene_t = []
        i=0
        for line in array:
            if ''.join(line[5:9])=='PDB;':
        #        print("Done")
        #        chk =1
                temp.append(line)
                if ''.join(array[i-1][5:13])=='UniGene;' and chk == 0:
                    Uni_gene_t.append(array[i-1])
                    chk =1
            else:
                if chk==1:
                    whole.append(temp)
                chk = 0
                temp=[]
            i=i+1
        #% seperate unigene name
        uni_gene =[]
        for n in Uni_gene_t:
            for i in range(14,len(n)):
                if n[i]==";":
        #           print(n[14:i])  
                   uni_gene.append(n[14:i])
                   
        self.whole = whole
        
        # save the gene details in a pikle file
        os.chdir('/')
        os.chdir(self.saving_dir)
        pickle.dump(uni_gene, open(''.join([self.name,"_uni_gene.p"]), "wb" ) ) 
        
    def X_ray_PDBs_with_length(self):
        """
        then grab the information of PDBs like sequential lenth and X-Ray or NMR and the resolution
        Here I extract the X-ray information only
        
        first check if the X-ray information only
        order is 
        PDB_ID Resolution sequecne_start and end points
        """
        Ids_info_all=[]
        for temp in self.whole:
            Ids_info=[]
            for i in range(0,len(temp)):
                ids_temp_info = []
                if temp[i][16:21]=='X-ray':#check the entry is X-Ray defragmented
        #                print("chk---------------1")
                        ids_temp_info.append(temp[i][10:14])
                        ids_temp_info.append(temp[i][23:28])       
                        for j in range(0,len(temp[i])):
                           if temp[i][j] == "=":
                               ids_temp_info.append(temp[i][j+1:len(temp[i])-2])
                if len(ids_temp_info)>0:
                    Ids_info.append(ids_temp_info)  
            Ids_info_all.append(Ids_info)
           
        #% take the PDBIds and make the distribution
        # find the length of the each PDB sequence
        """
        Here some PDB ids are omitted because of the two sequece are there
        """
        length =[]
        resolution = []
        omitted = []
        start_end_position =[]
        for ids in Ids_info_all:
            o_t =[]
            start_end_t = []
            for id_t in ids:
                if len(id_t) == 3:
                    resolution.append(float(id_t[1]))
                    for j in range(0,len(id_t[2])):
                            if id_t[2][j]=="-":
                                length.append(int(id_t[2][j+1:len(id_t[2])])-int(id_t[2][0:j]))
                                start_end_t.append([int(id_t[2][j+1:len(id_t[2])]),int(id_t[2][0:j])])
                else:
                     o_t.append(id_t)
            omitted.append(o_t)
            start_end_position.append(start_end_t)
        
        #% save the results as pikle
        os.chdir('/')
        os.chdir(self.saving_dir)
        pickle.dump(omitted, open( ''.join([self.name,"_omitted.p"]), "wb" ) )           
        pickle.dump(resolution, open( ''.join([self.name,"_resolution.p"]), "wb" ) )           
        pickle.dump(length, open( ''.join([self.name,"_length.p"]), "wb" ) )     
        pickle.dump(Ids_info_all, open( ''.join([self.name,"_Ids_info_all.p"]), "wb" ) )        
        pickle.dump(start_end_position, open( ''.join([self.name,"_start_end_position.p"]), "wb" ) )   

class selecting_PDBs:    
    """
   This class uses to create the pdb_ids from the pikle files created earlier,
    It gives the freedom to select the threshold length
    """
    def __init__(self,saving_dir,name,threshold_length = 81):
        self.saving_dir = saving_dir
        self.name = name #name of the class
        self.threshold_length = threshold_length

    def selecting_PDBs(self):
        """
        This fumction take the 
        name: ONGO or TSG or Fusion
        threshold_length: PDBid has atleast this length in gene to be selected 
        
        Ids_info_all: THIS ONLY CONTAIIN X-RAY DATA
        """
        os.chdir('/')
        os.chdir(self.saving_dir)
    #    omitted = pickle.load(open( ''.join([name,"_omitted.p"]), "rb" ) )
        Ids_info_all = pickle.load(open( ''.join([self.name,"_Ids_info_all.p"]), "rb" ) )
        PDBs_sel=[]
        for ids in Ids_info_all:
            for id_t in ids:
                if len(id_t) == 3:
        #            resolution.append(float(id_t[1]))
                    for j in range(0,len(id_t[2])):
                            if id_t[2][j]=="-":
                               if (int(id_t[2][j+1:len(id_t[2])])-int(id_t[2][0:j]))+1 > self.threshold_length:
                                   PDBs_sel.append(id_t[0])
    
        pickle.dump(PDBs_sel,open(''.join([self.name,"_PDBs.p"]), "wb" ) )          
        #% write the PDBs in CSV file
        print(os.getcwd())
        f= open(''.join([self.name,"_selected_PDBs.csv"]),"w+")
        #f=open("start3.txt", "a+")
        for i in PDBs_sel:
              f.write(''.join([i,',','\n']))
        #      print(i)
        f.close() 
        
        f= open(''.join([self.name,"_selected_PDBs_for_R.csv"]),"w+")
        for i in PDBs_sel:
              f.write(''.join([i,'.pdb,','\n']))
        f.close() 