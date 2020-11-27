# -*- coding: utf-8 -*-
"""
Created on  %(26-Aug-2018) 20.30Pm

@author: %A.Nishanth C00294860
"""
import os
import pickle
import copy
#%%
saving_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"

#def selecting_PDBs(saving_dir,threshold_length):
threshold_length = 81
name = "ONGO_TSG"
"""
This fumction take the 
name: ONGO or TSG or Fusion
threshold_length: PDBid has atleast this length in gene to be selected 

Ids_info_all: THIS ONLY CONTAIIN X-RAY DATA
"""
os.chdir('/')
os.chdir(saving_dir)
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
                       if (int(id_t[2][j+1:len(id_t[2])])-int(id_t[2][0:j]))+1 > threshold_length:
                           PDB_t.append(id_t[0])
                           start_end_position_t.append([int(id_t[2][0:j]),int(id_t[2][j+1:len(id_t[2])])])
    PDBs_sel.append(PDB_t)
    start_end_position.append(start_end_position_t)
    
#%%  after threshold satisfication tested the select gene and PDB details
fin_PDB_ids = [] # finalised_PDB_ids 
fin_PDB_ids_start_end_position = [] # finalised_PDB_ids_start_end_position
fin_uni_gene = [] # finalised_uni_gene 
for i in range(0,len(PDBs_sel)): 
    if len(PDBs_sel[i]) > 0:   
        fin_PDB_ids.append(PDBs_sel[i])
        fin_PDB_ids_start_end_position.append(start_end_position[i])
        fin_uni_gene.append(uni_gene[i])  
        
def unique_element(mylist):
    """
    This function gave the unique element as in the order given in the mylist
    """
    used = set()
    unique = [x for x in mylist if x not in used and (used.add(x) or True)]
    return copy.deepcopy(unique)

def start_sorted_pdb_idex_group(start_end_position_t):
    """
    This function only useful for Method_2 and Method_3
    
    This function find the unique start end positions and 
    then sort the starting position in ascending order
    
    pdb_ids_matched_with_unique: list of unique_group_idexes
    avail_start                : available start positions from the unique sequence covered
    sort_start_positions_index : sorted start position indexes of avail_start or  pdb_ids_matched_with_unique
    avail_end                  : available end positions from the unique sequence covered                            
                                
    """
    start_only = []
    end_only = []
    for s_e in start_end_position_t:
        start_only.append(s_e[0])
        end_only.append(s_e[1])
    # then find the unique elements in the given start and end list to find the 
    # same kind of PDB_ids covering the sequence
    unique_start = unique_element(start_only)
    unique_end = unique_element(end_only)
    # by using the unique lists find the similar PDB_ids with their idexes
    pdb_ids_matched_with_unique = []
    
    for i in range(0,len(unique_start)):
        for k in range(0,len(unique_end)):
            chk = []
            for j in range(0,len(start_only)):
                if unique_start[i] ==  start_only[j]:
                    if unique_end[k] == end_only[j]:
                        chk.append(j)
            if len(chk)>0:
                pdb_ids_matched_with_unique.append(chk)
    # checking purpose
    #c=0
    #for i in pdb_ids_matched_with_unique:
    #    c=c+len(i)
    
    avail_start = []
    avail_end = []
    for n in pdb_ids_matched_with_unique:
        avail_start.append(start_end_position_t[n[0]][0]) 
        avail_end.append(start_end_position_t[n[0]][1]) 
    
    # then sort the available start positions
    """
    sort_start_positions_index 
                                specify  group_of_PDB has same sequence overlap
                                (the indexes of "pdb_ids_matched_with_unique")
                                
    "pdb_ids_matched_with_unique" itself contain the indexes of fin_PDB_ids selected Gene's index's PDB_ids    
                                                                                                or 
    start_end_position_t
    """   
    sort_start_positions_index =  [m[0] for m in sorted(enumerate(avail_start), key=lambda x:x[1])] 

    return pdb_ids_matched_with_unique, avail_start, sort_start_positions_index, avail_end  

"""
    Method_3
    
    Choose the maximum sequence cover as possible with overlap 
    
    cov(n+1) = max { cov(n-1) + length(next_PDB) - overlapped_with_previous_one
                     cov(n)}
    
    Here it only consider partial overlaps not full over lap 
    
    Exception: If two or more PDB_ids has same start end position both
                it considered as one for the calculation
    
"""

def covering_sequece_method_3(avail_start, sort_start_positions_index, avail_end ):
    """
    This fuction do the dynamic programming to find the optimal but 
    this o/p's the indexes choosen from the unique groups of PDB_ids
    
     cov(n+1) = max { cov(n-1) + length(next_PDB) - overlapped_with_previous_one
                     cov(n)}
    
    Here it only consider partial overlaps not full over lap 
    
    Exception: If two or more PDB_ids has same start end position both
                it considered as one for the calculation
    
    """
     # first choose the first covering PDB
    cov = [avail_end[sort_start_positions_index[0]]-avail_start[sort_start_positions_index[0]]]
    indexes_choosen_cov = [[sort_start_positions_index[0]]]# this list save the indexes of pdb_ids_matched_with_unique choosen for cov
    #% then check the end position of the covered PDB_id to check overlab with the next id
    for j in range(0,len(sort_start_positions_index)):
    #        print("J chk: ",j)
        # check the unoverlap condition
        # list_name[-1]: means last element of the list
        if avail_end[indexes_choosen_cov[-1][-1]] < avail_start[sort_start_positions_index[j]]:
    #        print("case_1: ",j)
            cov.append(cov[len(cov)-1] + avail_end[sort_start_positions_index[j]] - avail_start[sort_start_positions_index[j]])
            new = copy.deepcopy(indexes_choosen_cov[-1])
            new.append(sort_start_positions_index[j])
            indexes_choosen_cov.append(copy.deepcopy(new))
        else:
            maximum_cov = cov[-1] 
            if maximum_cov < cov[-1] - avail_end[indexes_choosen_cov[-1][-1]]  + avail_end[sort_start_positions_index[j]]:
                cov.append( cov[-1] - avail_end[indexes_choosen_cov[-1][-1]]  + avail_end[sort_start_positions_index[j]])
                new = copy.deepcopy(indexes_choosen_cov[-1])
                new.append((sort_start_positions_index[j]))
                indexes_choosen_cov.append(copy.deepcopy(new))
    #            print("i for case_2: ",i)
    #            print("case_2: ",j)
            else:
                cov.append(cov[-1])
                indexes_choosen_cov.append(copy.deepcopy(indexes_choosen_cov[-1]))
    return indexes_choosen_cov    

def redundant_overlap_removal(indexes_choosen_cov):
    """ 
    Method_3 to avoid this situation checked under here
                 
        [------PDB_1--------]
                     [------PDB_2--------]                
                        [--------------PDB_3----------]  
                        
        In this case PDB_2 is not needed
        
        condition chk
        Start_PDB_3 <= End_PDB_1
        
    """
    start_fraction = []
    end_fraction = [] 
    for index_groups in indexes_choosen_cov[-1]:
        start_fraction.append(avail_start[index_groups])
        end_fraction.append(avail_end[index_groups])
    
    avoided_indexes_choosen_cov_t = copy.deepcopy(indexes_choosen_cov[-1])    
    avoided_indexes_choosen_cov = [avoided_indexes_choosen_cov_t[0]]
    for k in range(0,len(start_fraction)-2):
        if start_fraction[k+2] > end_fraction[k]:
            avoided_indexes_choosen_cov.append(avoided_indexes_choosen_cov_t[k+1])
    if len(avoided_indexes_choosen_cov_t)>1:
        avoided_indexes_choosen_cov.append(avoided_indexes_choosen_cov_t[-1])
    return avoided_indexes_choosen_cov


#%%

#% update thefin_fractions for overlapped PDB_sequences

""" since the under coding is not working I created a function for takle the issue 
lengths_considered = []
for k in range(0,len(start_fraction)-1):
    if start_fraction[k+1] <= end_fraction[k]:
        over_lapped_temp.append(avoided_indexes_choosen_cov[k+1])
    else:
        over_lapped_groups.append(list(copy.deepcopy(over_lapped_temp)))
        over_lapped_temp = []
        over_lapped_temp.append(avoided_indexes_choosen_cov[k+1])
        
"""
def up_to_overlap_grouping(m, avoided_indexes_choosen_cov, start_fraction, end_fraction):
    """
    This function start from the m position and group them until un_overlap(overlap stops) occures
    
    In addition to that, it also calculates fractions for the unique overlapped PDB_s
    """
    # calculate the corresponding fractions
    fraction_t = []
    l_temp = [] # hold length of each PDBs
    over_lapped_temp = [avoided_indexes_choosen_cov[m]]
    for k in range(m,len(start_fraction)-1):
        if start_fraction[k+1] <= end_fraction[k]:
            over_lapped_temp.append(avoided_indexes_choosen_cov[k+1])
            fraction_t.append((1 + end_fraction[k] - start_fraction[k+1])/2)
            l_temp.append(end_fraction[k] - start_fraction[k] + 1)
            if k == len(start_fraction)-2:
                l_temp.append(end_fraction[k+1] - start_fraction[k+1] + 1)
        else:
            l_temp.append(end_fraction[k] - start_fraction[k] + 1)
            break
    if  len(l_temp) > 1:    
        if k == len(start_fraction)- 2:
            length = end_fraction[-1] - start_fraction[m] + 1
        else:
            length = end_fraction[k] - start_fraction[m] + 1
         
        """
        Fraction calculation for this method is totally different
        
        Here overlapped PDB_s together until find a gap them treated as same length
        
        Eg:
            O_l_1: means overlap 1
            
                     <-O_l_1->           <-O_l_2->
         [------PDB_1--------]
                     [------PDB_2--------]                 
                             [----PDB_3----------] 
         <-PDB_1-PDB_2-PDB_3-same_length _1 ----->  
         
         l_1 = PDB_1's overlap length
         
         (l_1 - O_l_1/2) /same_length _1 + (l_2 - O_l_1/2 - O_l_2/2)/same_length _1 
        """   
        fraction_temp = [(l_temp[0] - fraction_t[0])/length]
        if len(fraction_t)>1:
            for i in range(1,len(fraction_t)):
                fraction_temp.append((l_temp[i] - fraction_t[i] - fraction_t[i-1])/length)
            fraction_temp.append((l_temp[-1] - fraction_t[-1])/length)
        elif len(fraction_t) == 1:
            c  = (l_temp[-1] - fraction_t[0])/length
            fraction_temp.append(c)
        else:
            raise ValueError("Fraction WENT WRONG: CONTACT PROGRAMMER")
    
    else:
        # if the given PDB not overlapped with any other PDBs
        fraction_temp = [1.0]
        length = end_fraction[m] - start_fraction[m] + 1
    return k+1, over_lapped_temp, length, fraction_temp  

def fraction_length_Method_3(pdb_ids_matched_with_unique,start_end_position_t,avoided_indexes_choosen_cov, start_fraction, end_fraction):
    """
    This function calculates lengths_considered, fin_fractions for Method_3
    
    This func finalizes the fraction calculated by "up_to_overlap_grouping"
    Because "up_to_overlap_grouping" function only calculated for unique start_end_position
    
    This function consider the groups which has the same start_end positions 
    Then recalculate the fraction by devide the fraction by number of PDB's has same start_and_end position used 
    to cover
    """
    
    m = 0
    over_lapped_group = []
    lengths_considered = []
    fractions_for_all = []
    fin_fractions = []
    if len(start_fraction) == 1:
        if len(pdb_ids_matched_with_unique) == 1:
            lengths_considered.append(end_fraction[0]-start_fraction[0]+1)
            fraction_t = []
            for j in range(0,len(pdb_ids_matched_with_unique[0])):
                fraction_t.append(1/len(pdb_ids_matched_with_unique[0]))
            fin_fractions.append(fraction_t)
        else:
            fraction_t = []
            for j in range(0,len(pdb_ids_matched_with_unique)):
                if j ==  avoided_indexes_choosen_cov[0]:
                    for k in range(0,len(pdb_ids_matched_with_unique[j])):
                        fraction_t.append(1/len(pdb_ids_matched_with_unique[j]))
                else:
                    for k in range(0,len(pdb_ids_matched_with_unique[j])):
                        fraction_t.append(0)
            fin_fractions.append(fraction_t)
            
        
    else:
        while m < len(start_fraction)-1:
            m, over_lapped_temp, length, fraction  = up_to_overlap_grouping(m, avoided_indexes_choosen_cov, start_fraction, end_fraction)
            over_lapped_group.append(over_lapped_temp)
            lengths_considered.append(length)
            fractions_for_all.append(fraction)
     
    
        for j in range(0,len(lengths_considered)):
            fin_fractions_t = []
            for k in range(0,len(start_end_position_t)):
                cond = True
                for o in range(0,len(over_lapped_group[j])):
                    for id_s in pdb_ids_matched_with_unique[over_lapped_group[j][o]]:
                        if id_s == k:
                            cond = False
                            fin_fractions_t.append(fractions_for_all[j][o]/len(pdb_ids_matched_with_unique[over_lapped_group[j][o]]))
        #                    print("j: ",j, "k: ", k,"o: ",o, "id_s: ",id_s )
                if cond:
                     fin_fractions_t.append(0)       
            fin_fractions.append(fin_fractions_t)
    return lengths_considered, fin_fractions
#%%

def method_3_rest(pdb_ids_matched_with_unique, avail_start, sort_start_positions_index, avail_end, pdb_ids_for_gene_t, fin_uni_gene, i):
    """
        This method gives the rest of Method_3 or combine the whole method of Method_3
    """
    #    pdb_ids_matched_with_unique, avail_start, sort_start_positions_index, avail_end = start_sorted_pdb_idex_group(start_end_position_t)
    indexes_choosen_cov = covering_sequece_method_3(avail_start, sort_start_positions_index, avail_end )
    # To fix the overlap redundatly used
    avoided_indexes_choosen_cov = redundant_overlap_removal(indexes_choosen_cov)   
    # update the fin_fractions for overlapped PDB_sequences 
    start_fraction = []
    end_fraction = [] 
    for index_groups in avoided_indexes_choosen_cov:
        start_fraction.append(avail_start[index_groups])
        end_fraction.append(avail_end[index_groups])
    
    #    over_lapped_groups = []
    #    over_lapped_temp = [avoided_indexes_choosen_cov[0]]
    lengths_considered, fin_fractions= fraction_length_Method_3(pdb_ids_matched_with_unique,start_end_position_t,avoided_indexes_choosen_cov, start_fraction, end_fraction)
    whole = []
    whole.append(fin_uni_gene[i])
    whole.append(lengths_considered)      
    whole.append(fin_fractions)
    whole.append(pdb_ids_for_gene_t)
    return whole

#%%
i = 4
#for i in range(0,len(fin_PDB_ids_start_end_position)):
start_end_position_t = fin_PDB_ids_start_end_position[i]    
pdb_ids_for_gene_t  =  copy.deepcopy(fin_PDB_ids[i])
# if only one PDB id exist only use that for prediction no needed to create the weight matrix
if len(start_end_position_t)>1:     
   
    """
    Method_2 and Method_3 usage given
    """
    #first choose the start positions only when we doing that 
    #if it contain same start and end position place their idexes as groups
    pdb_ids_matched_with_unique, avail_start, sort_start_positions_index, avail_end = start_sorted_pdb_idex_group(start_end_position_t)
    
    """
    Method_3
    """
    whole = method_3_rest(pdb_ids_matched_with_unique, avail_start, sort_start_positions_index, avail_end, pdb_ids_for_gene_t, fin_uni_gene, i)
#    pickle.dump(whole, open( ''.join([name,"_method_3_", fin_uni_gene[i],".p"]), "wb" ) )  

#else:
#    """       incase only one PDB_id exist
#    """
#    whole = []
#    whole.append(fin_uni_gene[i])  
#    whole.append(pdb_ids_for_gene_t)
#    pickle.dump(whole, open( ''.join([name,"_direct_", fin_uni_gene[i],".p"]), "wb" ) ) 
#%% 
indexes_choosen_cov = covering_sequece_method_3(avail_start, sort_start_positions_index, avail_end )
#%%  To fix the overlap redundatly used
avoided_indexes_choosen_cov = redundant_overlap_removal(indexes_choosen_cov)   
# update the fin_fractions for overlapped PDB_sequences 
start_fraction = []
end_fraction = [] 
for index_groups in avoided_indexes_choosen_cov:
    start_fraction.append(avail_start[index_groups])
    end_fraction.append(avail_end[index_groups])

#    over_lapped_groups = []
#    over_lapped_temp = [avoided_indexes_choosen_cov[0]]
lengths_considered, fin_fractions= fraction_length_Method_3(pdb_ids_matched_with_unique,start_end_position_t,avoided_indexes_choosen_cov, start_fraction, end_fraction)
whole = []
whole.append(fin_uni_gene[i])
whole.append(lengths_considered)      
whole.append(fin_fractions)
whole.append(pdb_ids_for_gene_t)