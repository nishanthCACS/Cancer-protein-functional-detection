# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""

    
site_sat_occurance_counter =   ONGO_site_sat_occurance_counter  

Unique_site_sat_occurance_counter =  list(set(site_sat_occurance_counter))
# then organize them in the assending order
Unique_site_sat_occurance_counter_assending_key =   [i[0] for i in sorted(enumerate(Unique_site_sat_occurance_counter), key=lambda x:x[1])]

# then the count of each element of UniGene_unique_assending  
# with the unique_elements_in Unique_site_sat_occurance_counter check how frequent they occurs
UniGene_unique_freq = []
UniGene_index = []
for i in range(0, len(Unique_site_sat_occurance_counter)):
    count = 0
    temp = []
    for j in range(0,len(site_sat_occurance_counter)):
        if Unique_site_sat_occurance_counter[i] == site_sat_occurance_counter[j]:
              count = count + 1
              temp.append(j)
    UniGene_unique_freq.append(count)
    UniGene_index.append(temp)
## using these details to map back
#UniGene_index_assending = []
#for i in Unique_site_sat_occurance_counter_assending_key:
#    temp = []
#    for j in range(0,len(site_sat_occurance_counter)):
#        if Unique_site_sat_occurance_counter[i] == site_sat_occurance_counter[j]:
#            temp.append(j)
#    UniGene_index_assending.append(temp)

training_data =[]
testing_data =[]
if Unique_site_sat_occurance_counter[Unique_site_sat_occurance_counter_assending_key[0]] == 0:
    m = 1 # index of UniGene to avoid 0 numeber of SITE occurance
    cumulative_count = sum(UniGene_unique_freq) -  UniGene_unique_freq[Unique_site_sat_occurance_counter_assending_key[0]]
else:
    m=0
    cumulative_count = sum(UniGene_unique_freq) 
carry = 0 # to hold the number of elements take for the next one   intial condition
while cumulative_count > 0:
    print("Cumulative_count: ",cumulative_count)
    print("m               : ", m)
    print("key             : ", Unique_site_sat_occurance_counter_assending_key[m])
    print("carry           : ", carry)
    i = Unique_site_sat_occurance_counter_assending_key[m]
    if carry < 0:
        carry = 0  
    for j in range(carry,UniGene_unique_freq[i]-3,4):
        testing_data.append(m) 
        for k in range(0,3):
            training_data.append(m)
    
    # to staisfy the ending condition
    cond_2 = True
    if cumulative_count < carry: 
        cond_2 = False
        print("m: ",m)
        if m+1 == len(Unique_site_sat_occurance_counter_assending_key):
            for j in range(0,cumulative_count):
                training_data.append(m)
                print("Here-----------------1")
            break
        else:
            for j in range(0,UniGene_unique_freq[i]):
                training_data.append(m)
                print("Here-----------------1-2")
            carry = carry - UniGene_unique_freq[i]
    if cond_2:
        new_count = UniGene_unique_freq[i]-carry    
        if new_count >= 4:# since 4 is the window size
            # inorder to add the remaining data for training
            # to add the front part left
            for j in range(0,carry):
                training_data.append(m)
                print("Here-----------------2")
            # to add the end part left
            if cumulative_count < 4: 
                if (new_count % 4 != 0):
                    c = list(range(UniGene_unique_freq[i] - (new_count % 4) ,UniGene_unique_freq[i]))
                    for k in c:
                        training_data.append(m)
                        print("Here-----------------3")
            else:
                if (new_count % 4 != 0):
                    c = list(range(UniGene_unique_freq[i] -(new_count % 4) ,UniGene_unique_freq[i]))
                    cond = True # to place the first element of the stack to the test data
                    carry = 4
                    for k in c:
                        if cond and (cumulative_count - UniGene_unique_freq[i])>0:
                            testing_data.append(m)
                            cond= False
                        else:
                            training_data.append(m)
                            print("Here-----------------4")
                        carry = carry-1
        else: 
            # in this case consider about overlap front and end
            # to add the front part left
            if carry > 0 and UniGene_unique_freq[i]>0:
                if UniGene_unique_freq[i]>carry:
                    for j in range(0,carry):
                        training_data.append(m)
                    testing_data.append(m)
                    COUNT = 0
                    for j in range(0, UniGene_unique_freq[i] -1 - carry):
                        training_data.append(m)
                        print("Here-----------------5")
                        COUNT = COUNT +1
                        
                    carry = 4 - COUNT 
                else:
                    print("Here-----------------6")
                    for j in range(0,UniGene_unique_freq[i]):
                        training_data.append(m)
                    carry = carry - UniGene_unique_freq[i]
            elif  UniGene_unique_freq[i]>0:
                carry = 4
                testing_data.append(m)
                for j in range(1,UniGene_unique_freq[i]):
                    print("Here-----------------7")
                    training_data.append(m)
                    carry = carry-1
                carry = carry - UniGene_unique_freq[i]
    cumulative_count = cumulative_count - UniGene_unique_freq[i]
    m =m + 1
print("Actual: ", sum(UniGene_unique_freq) -  UniGene_unique_freq[Unique_site_sat_occurance_counter_assending_key[0]])
print("GOT   : ", len(training_data)+len(testing_data))
#%% then select the training data from the index
training_data_Uni_Gene_indexes =[]
test_data_Uni_Gene_indexes =[]
unique_training_data = list(set(training_data))   
training_data_unique_frequent = []
for U in unique_training_data:
    count = 0
    for ele in training_data:
        if ele == U:
            count = count + 1
    training_data_unique_frequent.append(count)

for i in range(0,len(unique_training_data)):
    # first choose the list we wanted  
#    print(unique_training_data[Unique_site_sat_occurance_counter_assending_key[i]])
    list_index_wanted = UniGene_index[Unique_site_sat_occurance_counter_assending_key[unique_training_data[i]]]
    print("unique_training_data[i]: ",unique_training_data[i])
    print("training_data_unique_frequent[i]: ",training_data_unique_frequent[i])
    for j in  range(0,training_data_unique_frequent[i]):
        training_data_Uni_Gene_indexes.append(list_index_wanted[j])

#%% then select the training data from the index
testing_data_Uni_Gene_indexes =[]
test_data_Uni_Gene_indexes =[]
unique_testing_data = list(set(testing_data))   
testing_data_unique_frequent = []
for U in unique_testing_data:
    count = 0
    for ele in testing_data:
        if ele == U:
            count = count + 1
    testing_data_unique_frequent.append(count)

for i in range(0,len(unique_testing_data)):
    # first choose the list we wanted  
#    print(unique_testing_data[Unique_site_sat_occurance_counter_assending_key[i]])
    list_index_wanted = UniGene_index[Unique_site_sat_occurance_counter_assending_key[unique_testing_data[i]]]
    print("unique_testing_data[i]: ",unique_testing_data[i])
    print("testing_data_unique_frequent[i]: ",testing_data_unique_frequent[i])
    # for tesing choose the Genes from the backside way
    for j in  range(1,testing_data_unique_frequent[i]+1):
        testing_data_Uni_Gene_indexes.append(list_index_wanted[-j])
#%% combine both lists and check where missed
all_chk =[]
for i in testing_data_Uni_Gene_indexes:
    all_chk.append(i)
for i in training_data_Uni_Gene_indexes:
    all_chk.append(i)
all_chk_unique =  list(set(all_chk))

total_there = list(range(0,53))
missing = [item for item in total_there  if item not in all_chk_unique]