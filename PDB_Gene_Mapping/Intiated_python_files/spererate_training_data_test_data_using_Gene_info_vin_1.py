# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""

ONGO_site_sat_occurance_counter = [0,1,5,2,3,4,6,7]
training_data =[]
training_data_rest =[]
testing_data =[]
UniGene_assending_order =  [i[0] for i in sorted(enumerate(ONGO_site_sat_occurance_counter), key=lambda x:x[1])]
cumulative_count = sum(ONGO_site_sat_occurance_counter)

m = 0 # index of UniGene

carry = 0 # to hold the number of elements take for the next one   intial condition
while cumulative_count > 0:
    i = UniGene_assending_order[m]
    if carry < 0:
        carry = 0 
#    if ONGO_site_sat_occurance_counter[i]-3 -carry > 0:
    for j in range(carry,ONGO_site_sat_occurance_counter[i]-3,4):
        testing_data.append(i) 
        print("Here---33")
        for k in range(0,3):
            training_data.append(i)
    
    # to staisfy the ending condition
    cond_2 = True
    if cumulative_count < carry: 
        cond_2 = False
        for j in range(0,cumulative_count):
            print("Here---1")
            print("i        : ",i)
            print("carry    : ",carry) 
            training_data_rest.append(i)
        break
            
    if cond_2:
        new_count = ONGO_site_sat_occurance_counter[i]-carry    
        if new_count >= 4:# since 4 is the window size
            # inorder to add the remaining data for training
            # to add the front part left
            print("Here----2")
            print("i: ",i ," new_count: ", new_count) 
#            print("i        : ",i)
#            print("carry    : ",carry) 
            for j in range(0,carry):
                training_data_rest.append(i)
    
            # to add the end part left
            print("Here----3")
#            print("i        : ",i)
#            print("carry    : ",carry)  
            if cumulative_count < 4: 
                if (new_count % 4 != 0):
                    c = list(range(ONGO_site_sat_occurance_counter[i] - (new_count % 4) ,ONGO_site_sat_occurance_counter[i]))
                    for k in c:
                        training_data_rest.append(i)
            else:
                print("Here----4")
#                print("i        : ",i)
#                print("carry    : ",carry)  
                if (new_count % 4 != 0):
                    c = list(range(ONGO_site_sat_occurance_counter[i] -(new_count % 4) ,ONGO_site_sat_occurance_counter[i]))
                    cond = True # to place the first element of the stack to the test data
                    carry = 4
                    for k in c:
                        if cond:
                            testing_data.append(i)
                            cond= False
                        else:
                            training_data_rest.append(i)
                        carry = carry-1
        else: 
            # in this case consider about overlap front and end
            # to add the front part left
            if carry > 0 and ONGO_site_sat_occurance_counter[i]>0:
                if ONGO_site_sat_occurance_counter[i]>carry:
                    print("Here----5")
#                    print("i        : ",i)
#                    print("carry    : ",carry) 
                    for j in range(0,carry):
                        training_data_rest.append(i)
                    testing_data.append(i)
                    COUNT = 0
                    for j in range(0, ONGO_site_sat_occurance_counter[i] -1 - carry):
                        training_data_rest.append(i)
                        COUNT = COUNT +1
                        
                    carry = 3 - COUNT 
                    print("carry--5: ",carry)
                else:
                    print("Here----6")
#                    print("i        : ",i)
#                    print("carry    : ",carry) 
                    for j in range(0,ONGO_site_sat_occurance_counter[i]):
                        training_data_rest.append(i)
                    carry = carry - ONGO_site_sat_occurance_counter[i]
            elif  ONGO_site_sat_occurance_counter[i]>0:
                print("Here----7")
#                print("i        : ",i)
#                print("carry    : ",carry) 
                carry = 4
                testing_data.append(i)
                for j in range(1,ONGO_site_sat_occurance_counter[i]):
                    training_data_rest.append(i)
                    carry = carry-1
                carry = carry - ONGO_site_sat_occurance_counter[i]
        print("i: ", i," carry    : ",carry)   
    cumulative_count = cumulative_count - ONGO_site_sat_occurance_counter[i]
    m =m + 1
    
print("Actual: ", sum(ONGO_site_sat_occurance_counter))
print("GOT   : ", len(training_data)+len(training_data_rest)+len(testing_data))

#%% to calculte the average
# to count hoe many PDB_ids has site information amongh the genes satisfied threshold condition
# from that we can caluclate the average number of PDBs for Gene
#(this average calculation consider the Genes which they don't has PDBs SITE info)
Total_number_site_satisfied = sum(ONGO_site_sat_occurance_counter) + sum(Fusion_site_sat_occurance_counter) + sum(TSG_site_sat_occurance_counter)
average = Total_number_site_satisfied/(len(ONGO_gene_sat)+len(TSG_gene_sat)+len(Fusion_gene_sat))
#%% create fibonachi bin size for histiogram in excel
maximum = max([max(ONGO_site_sat_occurance_counter),max(TSG_site_sat_occurance_counter),max(Fusion_site_sat_occurance_counter)])

def fib(n):
    """
    When you give the list "n" with fibonachi number
    this function addup the list with next fiibonachi number
    """
    m = copy.deepcopy(n)
    m.append(n[-1] + n[-2])
    return m
n = [0,1,2]
while n[-1] < maximum:
    n = fib(n)
#%%
#for i in UniGene_assending_order:
#    for j in range(0,ONGO_site_sat_occurance_counter[i],4):
#        testing_data.append(i)
#    if ONGO_site_sat_occurance_counter[i] > 4:
#        if (ONGO_site_sat_occurance_counter[i] % 4 != 0):
#            c = list(range(ONGO_site_sat_occurance_counter[i] - 1 -(ONGO_site_sat_occurance_counter[i] % 4) ,ONGO_site_sat_occurance_counter[i]))
#            for k in c:
#                training_data_rest.append(i)
#    elif 0 < ONGO_site_sat_occurance_counter[i] < 4:
#        for k in range(0,ONGO_site_sat_occurance_counter[i] % 4):
#            training_data_rest.append(i)       