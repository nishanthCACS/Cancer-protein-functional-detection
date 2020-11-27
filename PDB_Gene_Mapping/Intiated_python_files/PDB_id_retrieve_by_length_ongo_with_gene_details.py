# -*- coding: utf-8 -*-

"""
Created on %(25-Aug-2018) 11.33Am

@author: %(A.Nishanth)s
"""

import os
from os import listdir
import pickle
#%%
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping')
#%% first read the lines from the PDBid results
with open("ONGO_PDB_ids_with_length.txt", "r") as ins:
    array = []
    for line in ins:
        array.append(line)
#%% Then splkit the PDBids with info

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

#%% 
"""
then grab the information of PDBs like sequential lenth and X-Ray or NMR and the resolution
Here I extract the X-ray information only

first check if the X-ray information only
order is 
PDB_ID Resolution sequecne_start and end points
"""
Ids_info_all=[]
for temp in whole:
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
#%% take the PDBIds and make the distribution
# find the length of the each PDB sequence
"""
Here some PDB ids are omitted because of the two sequece are there
"""
length =[]
resolution = []
omitted = []
start_end_position =[]
a=0
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
#%% seperate unigene name
uni_gene =[]
for n in Uni_gene_t:
    for i in range(14,len(n)):
        if n[i]==";":
#           print(n[14:i])  
           uni_gene.append(n[14:i])
#%% save the results as pikle
pickle.dump(omitted, open( "ONGO_omitted.p", "wb" ) )           
pickle.dump(resolution, open( "ONGO_resolution.p", "wb" ) )           
pickle.dump(length, open( "ONGO_length.p", "wb" ) )     
pickle.dump(Ids_info_all, open( "ONGO_Ids_info_all.p", "wb" ) )        
pickle.dump(start_end_position, open( "ONGO_start_end_position.p", "wb" ) )   
pickle.dump(uni_gene, open( "ONGO_uni_gene.p", "wb" ) ) 