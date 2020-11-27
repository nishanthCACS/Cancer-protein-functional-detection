# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""

# -*- coding: utf-8 -*-
"""
Created on  %(06-Nov-2018) 10.30Am

@author: %A.Nishanth C00294860
"""
import os
import pickle
import copy

working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"

clean_must_sat_dir_part = "G:/ULL_Project/optimally_tilted/quater_models/W_OUT_SITE/"
os.chdir('/')
os.chdir(clean_must_sat_dir_part)

c_alpha_surface_cond= os.listdir()
clean_dir=0
clean_must_sat_dir_part_20=''.join([c_alpha_surface_cond[clean_dir],'/20_amino_acid/'])
sat_PDB_dir=''.join([clean_must_sat_dir_part,clean_must_sat_dir_part_20,'K_fold'])
threshold_dir= ''.join(['W_OUT_SITE/',clean_must_sat_dir_part_20])
name="ONGO"

#name="TSG"
saving_dir = "".join([working_dir, "/Spring_2020/",threshold_dir,name])
#%%
threshold_length = 81

os.chdir('/')
os.chdir(sat_PDB_dir)

site_sat=  pickle.load(open( ''.join(['OG',"_train_PDBs.p"]), "rb"))
#site_sat=  pickle.load(open( ''.join(['TSG',"_train_PDBs.p"]), "rb"))

overlapped_PDBs_ONGO = pickle.load(open('overlapped_PDBs_ONGO.p', "rb"))
overlapped_PDBs_TSG = pickle.load(open('overlapped_PDBs_TSG.p', "rb"))
overlapped_PDBs=overlapped_PDBs_ONGO+overlapped_PDBs_TSG
site_sat_flatten = []
for ids in site_sat:
    if ids not in overlapped_PDBs:
        site_sat_flatten.append(ids[0:4])

       
# first load the files needed
os.chdir('/')
os.chdir(working_dir)

"""
This fumction take the 
name: ONGO or TSG or Fusion
threshold_length: PDBid has atleast this length in gene to be selected 

Ids_info_all: THIS ONLY CONTAIIN X-RAY DATA

 Hs.731652
"""
Ids_info_all = pickle.load(open( ''.join([name,"_Ids_info_all.p"]), "rb" ) )
uni_gene = pickle.load(open( ''.join([name,"_uni_gene.p"]), "rb" ) )

PDBs_sel=[]
start_end_position = []
for ids in Ids_info_all:
    PDB_t =[]
    start_end_position_t =[]
    for id_t in ids:
        if len(id_t) == 3:
            for j in range(0,len(id_t[2])):
                if id_t[2][j]=="-":
                   if (int(id_t[2][j+1:len(id_t[2])])-int(id_t[2][0:j]))+1 > threshold_length:
                       if id_t[0] in site_sat_flatten:
                           PDB_t.append(copy.deepcopy(id_t[0]))
                           start_end_position_t.append([int(id_t[2][0:j]),int(id_t[2][j+1:len(id_t[2])])])
#                               else:
#                                   print("Not_satisfied: ",id_t[0])

    PDBs_sel.append(copy.deepcopy(PDB_t))
    start_end_position.append(copy.deepcopy(start_end_position_t))

#%  after threshold satisfication tested the select gene and PDB details
fin_PDB_ids = [] # finalised_PDB_ids 
fin_PDB_ids_start_end_position = [] # finalised_PDB_ids_start_end_position
fin_uni_gene = [] # finalised_uni_gene 
for i in range(0,len(PDBs_sel)): 
    if len(PDBs_sel[i]) > 0:   
        fin_PDB_ids.append(PDBs_sel[i])
        fin_PDB_ids_start_end_position.append(start_end_position[i])
        fin_uni_gene.append(uni_gene[i])  
                
        fin_PDB_ids = fin_PDB_ids
        fin_PDB_ids_start_end_position =fin_PDB_ids_start_end_position
        fin_uni_gene = fin_uni_gene
        """
        Here onwards Helper functions of all three methods are placed
        then create the probability matrix formation for weightage 
        with that create a list to include the corresponding length details


i=12     """
#%%

def unique_element( mylist):
    """
    This function gave the unique element as in the order given in the mylist
    """
    used = set()
    unique = [x for x in mylist if x not in used and (used.add(x) or True)]
    return copy.deepcopy(unique)

def start_sorted_pdb_idex_group( start_end_position_t):
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
       
"""



def redundant_overlap_removal( indexes_choosen_cov, avail_start, avail_end):
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
        else:
            print("[k]: ",start_fraction[k],'-',end_fraction[k])
            print("[k+1]: ",start_fraction[k+1],'-',end_fraction[k+1])
            print("[k+2]: ",start_fraction[k+2],'-',end_fraction[k+2])
            raise("Method-3 error in line 384")

    # if it has only one elementg to avoid adding that again
    if len(avoided_indexes_choosen_cov_t)>1:
        avoided_indexes_choosen_cov.append(avoided_indexes_choosen_cov_t[-1])
    return avoided_indexes_choosen_cov

#% update the fin_fractions for overlapped PDB_sequences
#    """ since the under coding is not working I created a function for takle the issue 
#    lengths_considered = []
#    for k in range(0,len(start_fraction)-1):
#        if start_fraction[k+1] <= end_fraction[k]:
#            over_lapped_temp.append(avoided_indexes_choosen_cov[k+1])
#        else:
#            over_lapped_groups.append(list(copy.deepcopy(over_lapped_temp)))
#            over_lapped_temp = []
#            over_lapped_temp.append(avoided_indexes_choosen_cov[k+1])
#            
#    """

def up_to_overlap_grouping( m, avoided_indexes_choosen_cov, start_fraction, end_fraction):
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
            # checking the ending condition if it overlap with the last one it has to considered
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
            
                     <-O_l_1->   <-O_l_2->
         [------PDB_1--------]
                     [------PDB_2--------]                  
                                 [----PDB_3----------]  
         <-PDB_1-PDB_2-PDB_3-same_length _1 ---------> 
         
         l_1 = PDB_1's overlap length
         
         (l_1 - O_l_1/2) /same_length _1 + (l_2 - O_l_1/2 - O_l_2/2)/same_length _1 
        """   
        fraction_temp = [(l_temp[0] - fraction_t[0])/length]
        if len(fraction_t)>1:
            for i in range(1,len(fraction_t)):
                fraction_temp.append((l_temp[i] - fraction_t[i] - fraction_t[i-1])/length)
            fraction_temp.append((l_temp[-1] - fraction_t[-1])/length)
#                print("Exit_here_1")
        elif len(fraction_t) == 1:
            c  = (l_temp[-1] - fraction_t[0])/length
            fraction_temp.append(c)
#                print("Exit_here_2")
        else:
            raise ValueError("Fraction WENT WRONG: CONTACT PROGRAMMER")
    
    else:
        # if the given PDB not overlapped with any other PDBs
#            print("Exit_here_3")
        fraction_temp = [1.0]
        length = end_fraction[m] - start_fraction[m] + 1
    return k+1, over_lapped_temp, length, fraction_temp  

def fraction_length_Method_3( pdb_ids_matched_with_unique,start_end_position_t,avoided_indexes_choosen_cov, start_fraction, end_fraction):
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
    #incase it only has only one group, fraction canbe calculated directly
    if len(start_fraction) == 1:
        lengths_considered.append(end_fraction[0]-start_fraction[0]+1)
        # matched case has only one group
        if len(pdb_ids_matched_with_unique) == 1:
            fraction_t = []
            for j in range(0,len(pdb_ids_matched_with_unique[0])):
                fraction_t.append(1/len(pdb_ids_matched_with_unique[0]))
            fin_fractions.append(fraction_t)
        else:
            # matched case has some small groups
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
                if cond:
                     fin_fractions_t.append(0)       
            fin_fractions.append(fin_fractions_t)
    return lengths_considered, fin_fractions


def method_3_rest( pdb_ids_matched_with_unique, avail_start, sort_start_positions_index, avail_end, pdb_ids_for_gene_t, start_end_position_t, fin_uni_gene, i):
    """
    This method gives the rest of Method_3 or combine the whole method of Method_3
    """
    indexes_choosen_cov = covering_sequece_method_3(avail_start, sort_start_positions_index, avail_end )
    # To fix the overlap redundatly used
    avoided_indexes_choosen_cov = redundant_overlap_removal(indexes_choosen_cov, avail_start, avail_end) 
    # update the fin_fractions for overlapped PDB_sequences 
    start_fraction = []
    end_fraction = [] 
    for index_groups in avoided_indexes_choosen_cov:
        start_fraction.append(avail_start[index_groups])
        end_fraction.append(avail_end[index_groups])
    
    lengths_considered, fin_fractions= fraction_length_Method_3(pdb_ids_matched_with_unique,start_end_position_t,avoided_indexes_choosen_cov, start_fraction, end_fraction)
    whole = []
    whole.append(fin_uni_gene[i])
    whole.append(lengths_considered)      
    whole.append(fin_fractions)
    whole.append(pdb_ids_for_gene_t)
    return whole
#%%
    
def size_index_chhosen_checker(list_1):
    size_change_places=[]
    for i in range(1,len(list_1)):
        if len(list_1[i])!=len(list_1[i-1]):
            size_change_places.append(i-1)
    size_change_places.append(len(list_1)-1)
    return size_change_places

def covering_sequece_method_3( avail_start, sort_start_positions_index, avail_end ):
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
        # check the unoverlap condition
        # list_name[-1]: means last element of the list
        if avail_end[indexes_choosen_cov[-1][-1]] < avail_start[sort_start_positions_index[j]]:
            cov.append(cov[len(cov)-1] + avail_end[sort_start_positions_index[j]] - avail_start[sort_start_positions_index[j]])
            new = copy.deepcopy(indexes_choosen_cov[-1])
            new.append(sort_start_positions_index[j])
            indexes_choosen_cov.append(copy.deepcopy(new))
        else:
            maximum_cov = cov[-1] 
            if maximum_cov < cov[-1] - avail_end[indexes_choosen_cov[-1][-1]]  + avail_end[sort_start_positions_index[j]]:
                cov.append(cov[-1] - avail_end[indexes_choosen_cov[-1][-1]]  + avail_end[sort_start_positions_index[j]])
                if len(indexes_choosen_cov[-1])==1 and avail_start[sort_start_positions_index[j]]==avail_start[indexes_choosen_cov[-1][-1]]:#to fix the same satrting postion but different ending position
                    indexes_choosen_cov.append([sort_start_positions_index[j]]) 
                else:
                    print("Herer ----1",sort_start_positions_index[j])
                    if avail_start[sort_start_positions_index[j]]==avail_start[indexes_choosen_cov[-1][-1]]:#to fix the same satrting postion but different ending position
                        p=-1
                        while len(indexes_choosen_cov)>-p and avail_start[indexes_choosen_cov[p][-1]]== avail_start[sort_start_positions_index[j]]:
                            p=p-1
                        if len(indexes_choosen_cov[p])>0:
                            new = copy.deepcopy(indexes_choosen_cov[p])     
                            new.append((sort_start_positions_index[j]))
                            indexes_choosen_cov.append(copy.deepcopy(new))
                        else:
                            indexes_choosen_cov.append([sort_start_positions_index[j]])                        
                    elif  len(indexes_choosen_cov[-1])>1:
                        indexes_choosen_cov_change = size_index_chhosen_checker(indexes_choosen_cov)
                        if avail_start[sort_start_positions_index[j]]<= avail_end[indexes_choosen_cov[indexes_choosen_cov_change[-2]][-1]]:
    
                            print("Herer ----2",sort_start_positions_index[j])
                            print("indexes_choosen_cov_change: ",indexes_choosen_cov_change)
                            '''
                            To avoid
                            [------PDB_0---]
                                [------PDB_1------------]
                                            [------PDB_2--------]   
                                               [------PDB_3------------] 
                                                   ::
                                                  [--------------PDB_n----------] 
                            In this kind of situation only PDB_0, PDB_1 and PDB_n is enough to cover the whole primary structure
                            '''
                            p=indexes_choosen_cov_change.pop(-1)
                            p=indexes_choosen_cov_change.pop(-1)
                            p_need=p 

                            print("Herer ----p",p)
    
                            while len(indexes_choosen_cov_change)>0 and avail_start[indexes_choosen_cov[p][-1]]<= avail_start[sort_start_positions_index[j]]<= avail_end[indexes_choosen_cov[p][-1]]:
                                p_need=p 
                                p=indexes_choosen_cov_change.pop(-1)
                                print("Herer ----p",p)
                                print("Herer ----p_need",p_need)
    
                            if len(indexes_choosen_cov[p_need])>0:
                                new = copy.deepcopy(indexes_choosen_cov[p_need])     
                                new.append((sort_start_positions_index[j]))
                                indexes_choosen_cov.append(copy.deepcopy(new))
                            else:
                                indexes_choosen_cov.append([sort_start_positions_index[j]])    
                        else:
                            new = copy.deepcopy(indexes_choosen_cov[-1])
                            new.append((sort_start_positions_index[j]))
                            indexes_choosen_cov.append(copy.deepcopy(new))
                    else:
                        new = copy.deepcopy(indexes_choosen_cov[-1])
                        new.append((sort_start_positions_index[j]))
                        indexes_choosen_cov.append(copy.deepcopy(new))
            else:
                cov.append(cov[-1])
                indexes_choosen_cov.append(copy.deepcopy(indexes_choosen_cov[-1]))
    return indexes_choosen_cov  


"""
This function finalizes(call the three methods functions)
methods and save the results as pikle file
"""
name = name
fin_PDB_ids = fin_PDB_ids
fin_PDB_ids_start_end_position = fin_PDB_ids_start_end_position
fin_uni_gene = fin_uni_gene

# change the directory to save the files
#os.chdir('/')
#os.chdir(saving_dir)
i=1

#i=34
#
'''
i_gene: Hs.505297  name:  TSG
#i=19
fin_uni_gene: Hs.465870  name:  TSG
i=34

'''
start_end_position_t = fin_PDB_ids_start_end_position[i]    
pdb_ids_for_gene_t  =  copy.deepcopy(fin_PDB_ids[i])
# if only one PDB id exist only use that for prediction no needed to create the weight matrix
#
"""
fin_uni_gene: Hs.505297  name:  TSG
fin_uni_gene: Hs.505297  name:  TSG
Method_2 and Method_3 usage given
"""
#first choose the start positions only when we doing that 
#if it contain same start and end position place their idexes as groups
pdb_ids_matched_with_unique, avail_start, sort_start_positions_index, avail_end = start_sorted_pdb_idex_group(start_end_position_t)
indexes_choosen_cov = covering_sequece_method_3(avail_start, sort_start_positions_index, avail_end )
# To fix the overlap redundatly used
avoided_indexes_choosen_cov = redundant_overlap_removal(indexes_choosen_cov, avail_start, avail_end) 
# update the fin_fractions for overlapped PDB_sequences 
start_fraction = []
end_fraction = [] 
for index_groups in avoided_indexes_choosen_cov:
    start_fraction.append(avail_start[index_groups])
    end_fraction.append(avail_end[index_groups])
#%%

"""
Method_3
"""
whole = method_3_rest(pdb_ids_matched_with_unique, avail_start, sort_start_positions_index, avail_end, pdb_ids_for_gene_t, start_end_position_t, fin_uni_gene, i)
#    pickle.dump(whole, open( ''.join([name,"_method_3_", fin_uni_gene[i],".p"]), "wb" ) )  
#%%
#def method_3_rest( pdb_ids_matched_with_unique, avail_start, sort_start_positions_index, avail_end, pdb_ids_for_gene_t, start_end_position_t, fin_uni_gene, i):
"""
This method gives the rest of Method_3 or combine the whole method of Method_3
"""

#%%

#%%
lengths_considered, fin_fractions= fraction_length_Method_3(pdb_ids_matched_with_unique,start_end_position_t,avoided_indexes_choosen_cov, start_fraction, end_fraction)

#%%
#def fraction_length_Method_3( pdb_ids_matched_with_unique,start_end_position_t,avoided_indexes_choosen_cov, start_fraction, end_fraction):
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
#incase it only has only one group, fraction canbe calculated directly
if len(start_fraction) == 1:
    lengths_considered.append(end_fraction[0]-start_fraction[0]+1)
    # matched case has only one group
    if len(pdb_ids_matched_with_unique) == 1:
        fraction_t = []
        for j in range(0,len(pdb_ids_matched_with_unique[0])):
            fraction_t.append(1/len(pdb_ids_matched_with_unique[0]))
        fin_fractions.append(fraction_t)
    else:
        # matched case has some small groups
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
            if cond:
                 fin_fractions_t.append(0)       
        fin_fractions.append(fin_fractions_t)
#    return lengths_considered, fin_fractions

#%%
whole = []
whole.append(fin_uni_gene[i])
whole.append(lengths_considered)      
whole.append(fin_fractions)
whole.append(pdb_ids_for_gene_t)
"""
   Detailed explonation of all three methods    
           
    length of the sequences is placed in front
    
    [long----------------------- overlapped_ part -------]
    
    Method_1
    
    It find the longest overlapped PDB and then use that to give the fractions for the rest of the PDBs 
    Gene    [150---------------------------------------------------------------------------]
    PDB_1                    [60-------------------------------]
    PDB_2       [45---------------20------] 
    PDB_3                       [30--------30----------]
    PDB_4                                            [55----15--------------]   
    PDB_5                                                                     [----12-----]
    
    This algorithm first chooses the PDB_1 because that is the longest among all
    Then calculate the fraction of the rest of the PDBs overlapped with that PDB_1
       PDB_1: 1
       PDB_2: 20/60 = 0.333
       PDB_3: 30/60 = 0.5
       PDB_4: 15/60 = 0.25
       PDB_5: 0
           
       with these fractions the length considered also saved to calculate the overall probability 
       here the length is PDB_1's that is 60
       
       
       Then calculate the remaining length after removing the overlapped part
       
    PDB_1                    
    PDB_2       [25----------]
    PDB_3
    PDB_4                                                   [40--------------]   
    PDB_5                                                                     [----12-----]          
        
        This will lead to go to PDB_4 because that has the highest length[40]
        Then follow the same procedure to calculate the fractions
         
   PDB_1: 0
   PDB_2: 0
   PDB_3: 0
   PDB_4: 1
   PDB_5: 0
       
       with these fractions the length considered also saved to calculate the overall probability 
       here the length is PDB_1's that is PDB_4's: 40
       
       after done every calculation:
                60 (P[PDB_1] + 0.33 P[PDB_2] + 0.5 P[PDB_3] + 0.25 P[PDB_4]) +     40 (P[PDB_4])
         ----------------                                                       ---------------
         [40 + 60 + ... ]                                                       [40 + 60 + ... ]   
       
       
   ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
       
       Method_1 is the only algorithm considered  all PDBs
 -------------------------------------------------------------------------------------------------------
   Method_2 and Method_3  is mainly focused on covering the maximum gene as possible
   with the PDBids of that gene but how we choose those PDBs 
          Method_2 uses non-overlapping  PDB's to cover the gene
          Method_3 uses overlapping PDB's to cover the gene
      For ex:
          
     Gene    [150---------------------------------------------------------------------------]
    PDB_1                    [60-------------------------------]
    PDB_2       [45---------------20------] 
    PDB_3       [45---------------20------] 
    PDB_4                       [30--------30----------]
    PDB_5                       [10----10----]
    PDB_6                                            [55----15--------------]   
    PDB_7                                                                     [----12-----]
 
    Both algorithms uses the unique starting point for caculation
         
    Method_2 uses unoverlapping  PDB's to cover the gene
    
    cov(0) = 0
    cov(1) = 45 PDB_2 and PDB_3 (finally we use this information to give the vote for that part by the probabilities)    
    cov(2) = 60 PDB_1 
    cov(3) = 60 PDB_1 when checking PDB_4
    cov(4) = 60 PDB_1 when checking PDB_5
    cov(5) = 55+45 = 100 ((PDB_2 and PDB_3) and PDB_6)
    cov(6) = 112 ((PDB_2 and PDB_3),PDB_6 and PDB_7)
        
                  45 (0.5 P[PDB_2] + 0.5 P[PDB_3]) +   55 P[PDB_6] +    12 P[PDB_7]
   Method_2_fin = -----------------------------------   -----------   --------------                                                   ---------------
                        112                                112            112
                        
                        
 Method_3 uses overlapping PDB's to cover the gene        
               
    cov(0) = 0
    cov(1) = 45 PDB_2 and PDB_3 (finally we use this information to give the vote for that part by the probabilities)    
    cov(2) = 85 (PDB_2 and PDB_3) and PDB_1 
    cov(3) = 85 (PDB_2 and PDB_3) and PDB_1 when checking PDB_4
    cov(4) = 85 (PDB_2 and PDB_3) and PDB_1  when checking PDB_5
    cov(5) = 125 ((PDB_2 and PDB_3), PDB_1 and PDB_6)
    cov(6) = 137 ((PDB_2 and PDB_3), PDB_1, PDB_6 and PDB_7)
                      

                    (45-20/2)(0.5 P[PDB_2] + 0.5 P[PDB_3]) + (60-(20+15)/2)P[PDB_1] + (40-15/2)P[PDB_6] + 12 P[PDB_7]
   Method_3_fin = ---------------------------------------------------------------------------------------------------                                                  ---------------
                                                                    137                               
     small explonation:
                   
        overlap part of PDB_1 is (20 + 15) so it has half of owenership of that 
       
         since PDB_2 and PDB_3 together share overlap with PDB_1 so this give the overall fraction as
         (45-20/2)/137 and then devided by howmany of them sahre here 2 thats why 0.5 for each
         (0.5 P[PDB_2] + 0.5 P[PDB_3])
         
         that's how  (45-20/2)(0.5 P[PDB_2] + 0.5 P[PDB_3]) formed
   ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
       
       Method_2
       
    use dynamic programming to cover maximum length of the gene sequence with non_over lapping PDBs

    Exception: If two or more PDB_ids has same start end position both it considered as one for the calculation
    
    1> starting positions of PDB_ids are sorted in ascending order
    2> cov (n) = maximum covered_sequence_from_gene by n PDB_ids
    3> 
    cov(0) = 0
    If the next PDB_id is not overlapped with the previous one
    
    cov(n+1) = cov(n) + length(next_PDB_id) 
    
    if it overlapped go back until it non overlapped(eg if it take j steps to go back)
    
    cov (n+1) = max { cov(n),
                      cov(n-j) + len(next_PDB)}
    
    Exception:
        
    After selected PDB_ids two or more PDB_ids has same start end position
    it's noted seperately to give find the average probability for add
       
   :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    Method_3
    
        Choose the maximum sequence cover as possible with overlap 
    
    cov(n+1) = max { cov(n-1) + length(next_PDB) - overlapped_with_previous_one
                     cov(n)}
    
    Here it only consider partial overlaps not full over lap 
    
    Exception: If two or more PDB_ids has same start end position both
                it considered as one for the calculation
    
    ---------------------------------------------------------------------------------------------------------
            first findout the overlapped PDB ids
    This function give the fraction of PDB ids overlapped with the selected PDB id
    and update the un_overlap part removing the overlap fraction
                
    """