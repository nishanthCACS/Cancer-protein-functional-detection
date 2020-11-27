# -*- coding: utf-8 -*-
"""
Created on %14-May-2019(14.10 pm)

@author: %A.Nishanth C00294860
"""
import os
import csv

#%%
def detail_extract_finalised_gene(name):
    f = open(name)
    csv_f = csv.reader(f)
    stack=[]
    gene_name=[]
    Uni_gene=[]
    probabilities=[]
    Tier=[]
    method=[]
    class_census=[]
    chk=0
    for row in csv_f:
        stack.append(row)
    #    gene_name.append(row[1][1:len(row[1])])
    #    Uni_gene.append(row[2][1:len(row[2])])
        if chk==1:
            Tier.append(row[0])
            gene_name.append(row[1])
            Uni_gene.append(row[2])
            method.append(row[3])
            class_census.append(row[-1])
            probabilities.append([float(row[4][1:len(row[4])]),float(row[5][1:len(row[5])]),float(row[6][1:len(row[6])])])
        chk=1
    return  gene_name, Uni_gene, probabilities, Tier, method, class_census,stack
        
#% then go through the files and fix the probabilities and the UniGene issues 
"""
if the probabilities higher than .4 then assign that classes as it is 
elif .3 more assign as the classes
"""
def assign_classes(probability):
    if probability[0]>=0.4 and probability[1]>=0.4:
        predicted_class="ONCO_TSG"
    elif probability[2]>=0.4 and probability[1]>=0.4:
        predicted_class="TSG_Fusion"
    elif probability[0]>=0.4 and probability[2]>=0.4:
        predicted_class="ONCO_Fusion"
    elif probability[0]>=0.4:
         predicted_class="ONCO"
    elif probability[1]>=0.4:
         predicted_class="TSG"    
    elif probability[2]>=0.4:
         predicted_class="Fusion"
    elif probability[0]>=0.3 and probability[1]>=0.3 and probability[2]>=0.3:
        predicted_class="ONCO_TSG_Fusion"
    elif probability[0]>=0.3 and probability[1]>=0.3:
        predicted_class="ONCO_TSG"
    elif probability[2]>=0.3 and probability[1]>=0.3:
        predicted_class="TSG_Fusion"
    elif probability[0]>=0.3 and probability[2]>=0.3:
        predicted_class ="ONCO_Fusion"
    else:
         raise ValueError('rule not satisfied')
    return predicted_class
        
def assign_single_class_as(probability):
    if probability[0] == probability[1] and probability[2]<0.5:
        predicted_class="ONCO_TSG"
    elif probability[2] == probability[1] and probability[0]<0.5:
        predicted_class="TSG_Fusion"
    elif probability[0]== probability[2] and probability[1]<0.5:
        predicted_class="ONCO_Fusion"
    elif probability[0]> probability[1] and probability[0]> probability[2]:
         predicted_class="ONCO"
    elif probability[1]> probability[0] and probability[1]> probability[2]:
         predicted_class="TSG"    
    elif probability[2]> probability[1] and probability[2]> probability[0]:
         predicted_class="Fusion"
    else:
         print(probability)
         raise ValueError('rule not satisfied-2')
    return predicted_class        
    
#%%
chk_dir ="C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/all_to_gether_finalised/with_UniGene"
#chk_dir ="C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/all_to_gether_finalised/with_UniGene/T_1_train_test"
os.chdir('/')
os.chdir(chk_dir)
files=os.listdir()
saving_dir= "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/all_to_gether_finalised/UniGene_fixed"
#%%
for j in [2,3,4,0]: 
#for j in range(0,8): 
    os.chdir('/')
    os.chdir(chk_dir)
    name = files[j]
    gene_name, Uni_gene, probabilities, Tier, method, class_census,stack = detail_extract_finalised_gene(name)
    os.chdir('/')
    os.chdir(saving_dir)
    f= open(name,"w+")
    if j==0:# or j==1:
        f.write( 'Tier,\t'+'Gene_name,\t' +'Direct or Ensamble,\t'+ 'ONCO,'+'\t'+ 'TSG,'+ '\t'+ 'Fusion,\t'+'Class_classified,\t' +' Most_probable_class'+ '\t, Assigned class for gene census\n')      
    else:
        f.write( 'Tier,\t'+'Gene_name,\t' +'method_number,\t'+ 'ONCO,'+'\t'+ 'TSG,'+ '\t'+ 'Fusion,\t'+'Class_classified,\t' +' Most_probable_class' + '\t, Assigned class for gene census\n')
    for i in range(0,len(gene_name)):
       predicted = assign_classes(probabilities[i])
       most_likely =assign_single_class_as(probabilities[i])
       f.write(Tier[i] +','+ gene_name[i] +','+ method[i] + ',\t'+ str(probabilities[i][0])  +',\t' + str(probabilities[i][1])+',\t' + str(probabilities[i][2]) +',\t'+predicted+',\t'+most_likely+','+ class_census[i]+'\n' )
    f.close() 
    print(j)
#%%
os.chdir('/')
os.chdir(saving_dir)
#%%
#chk_dir ="C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/all_to_gether_finalised/with_UniGene/T_1_train_test"
os.chdir('/')
os.chdir(chk_dir)
files=os.listdir()
#%%
"""
then change PDB_id classification
"""

def detail_extract_finalised_pdb(name):
    f = open(name)
    csv_f = csv.reader(f)
    stack=[]
    pdb_ids=[]
    Uni_gene=[]
    probabilities=[]
    Tier=[]
    class_census=[]
    chk=0
    for row in csv_f:
        stack.append(row)
    #    gene_name.append(row[1][1:len(row[1])])
    #    Uni_gene.append(row[2][1:len(row[2])])
        if chk==1:
            Tier.append(row[0])
            Uni_gene.append(row[1])
            pdb_ids.append(row[2])
            probabilities.append([float(row[3][1:len(row[3])]),float(row[4][1:len(row[4])]),float(row[5][1:len(row[5])])])
            class_census.append(row[-1])
        chk=1
    return stack, Tier, Uni_gene, pdb_ids, probabilities, class_census
#%%
#name = files[0]
#name = files[1]
name = files[0]
gene_name, Uni_gene_g, probabilities, Tier, method, class_census,stack = detail_extract_finalised_gene(name)
#name = files[8]
#name = files[10]
name = files[5]

stack, Tier, Uni_gene, pdb_ids, probabilities, class_census = detail_extract_finalised_pdb(name)
selected_gene_name=[]
for g in Uni_gene:
    chk=0
    for i in range(0,len(Uni_gene_g)):
        if Uni_gene_g[i]==g:
            selected_gene_name.append(gene_name[i])
            chk=1
    if chk==0:
        print(g)


#%%
os.chdir('/')
os.chdir(saving_dir)
f= open(name,"w+")
f.write( 'Tier,\t'+'PDB_ID,\t'+'Gene_name,\t' + 'ONCO,'+'\t'+ 'TSG,'+ '\t'+ 'Fusion,\t'+'Class_classified,\t' +' Most_probable_class' + '\t, Assigned class for gene census\n')
for i in range(0,len(pdb_ids)):
   predicted = assign_classes(probabilities[i])
   most_likely =assign_single_class_as(probabilities[i])
   f.write(Tier[i]  +','+ pdb_ids[i]+','+ selected_gene_name[i] + ',\t'+ str(probabilities[i][0])  +',\t' + str(probabilities[i][1])+',\t' + str(probabilities[i][2]) +',\t'+predicted+',\t'+most_likely+','+ class_census[i]+'\n' )
f.close() 
#%%
import pickle
'''
problemtric mapped
Hs.574240 : ETV1    
Hs.509067 : PDGFRB 
Hs.654583 : RARA
Hs.350321 : RET
Hs.258855 : KMT2A

chk_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/pdb_pikles_length_gene/ONGO_Fusion_T_1'
os.chdir('/')
os.chdir(chk_dir)
Hs_654583 = pickle.load(open( ''.join(["ONGO_Fusion_T_1_method_3_Hs.654583.p"]), "rb" ) )

'''
problemetric_uni_genes=["Hs.574240","Hs.509067","Hs.654583","Hs.350321","Hs.258855"]
fixing_genes=["ETV1","PDGFRB","RARA","RET","KMT2A"]
fin_selec=[]
for i in range(0,len(fixing_genes)):
    fin_selec.append([fixing_genes[i],problemetric_uni_genes[i]])
#%
chk_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/clean_Finalised_probabilities_gene'
os.chdir('/')
os.chdir(chk_dir)
ONGO_Fusion_T_1_fin_gene = pickle.load(open( ''.join(["ONGO_Fusion_T_1_fin_gene_prob.p"]), "rb" ) )
ONGO_Fusion_T_1_Ensamble_gene = pickle.load(open( ''.join(["ONGO_Fusion_T_1_Ensamble_gene_prob.p"]), "rb" ) )
ONGO_Fusion_T_1_gene_direct_prob = pickle.load(open('ONGO_Fusion_T_1_gene_direct_prob.p', "rb" ) )

#% then extract the wanted probabilities
wanted=[]
for row in ONGO_Fusion_T_1_fin_gene:
    if row[0] in problemetric_uni_genes:
        wanted.append(row)
        
#for row in ONGO_Fusion_T_1_gene_direct_prob:
#    if row[0] in problemetric_uni_genes:
#        wanted.append(row)
#% then create the results for the problemtric cases

def uni_gene_to_id_gene_name(fin_selec,chk_uni_gene):
    for ids in fin_selec:
        if ids[1]==chk_uni_gene:
            return ids[0]
def decider_prob(ids, minimum_prob_decide,i):
    """
    Helper function for documentation for decide calss
    """ 
#    print(ids[i],'----')
    if ids[i][0] > minimum_prob_decide and ids[i][1] > minimum_prob_decide and ids[i][2] > minimum_prob_decide:
       return "ONGO_TSG_Fusion"
    elif ids[i][0] > minimum_prob_decide and ids[i][1] > minimum_prob_decide and ids[i][2] < minimum_prob_decide:
       return "ONGO_TSG"
    elif ids[i][0] > minimum_prob_decide and ids[i][1] < minimum_prob_decide and ids[i][2] > minimum_prob_decide:
       return  "ONGO_Fusion"
    elif ids[i][0] < minimum_prob_decide and ids[i][1] > minimum_prob_decide and ids[i][2] > minimum_prob_decide:
       return "TSG_Fusion"
    elif ids[i][0] > minimum_prob_decide*2:
       return "ONGO" 
    elif ids[i][1] > minimum_prob_decide*2:
       return "TSG"
    elif ids[i][2] > minimum_prob_decide*2:
       return "Fusion"
    else:
       raise ValueError("You needed to change the minimum_prob_decide because it doesn't satisfy all cases")


def method(i,wanted,fin_selec):
    chk_dir ="C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/all_to_gether_finalised/with_UniGene"

    os.chdir('/')
    os.chdir(chk_dir)
    f= open(''.join(['fixing_method',str(i),'.csv']),"w+")
    f.write( 'Tier,\t'+'Gene_name,\t'+'Uni_gene_ID,\t' + 'Method_number,\t'+ 'OG,'+'\t'+ 'TSG,'+ '\t'+ 'Fusion,\t'+'Class_classified' + '\t, Class_census\n')
    
    name = 'ONGO_Fusion'
    minimum_prob_decide =0.25
    for ids in wanted:
            #insert the group details
        predicted = decider_prob(ids, minimum_prob_decide,i)
        f.write('1,\t'+ uni_gene_to_id_gene_name(fin_selec,ids[0])+',\t' +  ids[0] +',\t'  +str(i)+',\t'+ str(round(ids[1][0],2))  +',\t' + str(round(ids[1][1],2))+',\t' + str(round(ids[1][2],2))+',\t'+predicted+',\t' + name+'\n' )
    f.close() 
method(1,wanted,fin_selec)
method(2,wanted,fin_selec)
method(3,wanted,fin_selec)
#%%
#import pickle
#
#chk_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/clean_Finalised_probabilities_gene'
#os.chdir('/')
#os.chdir(chk_dir)
#TSG_T_2_fin_gene = pickle.load(open( ''.join(["TSG_T_2_fin_gene_prob.p"]), "rb" ) )
#TSG_T_2_Ensamble_gene = pickle.load(open( ''.join(["TSG_T_2_Ensamble_gene_prob.p"]), "rb" ) )
#TSG_T_2_gene_direct_prob = pickle.load(open('TSG_T_2_gene_direct_prob.p', "rb" ) )
#fin_selec = [['POLG','Hs.706868']]
#
#wanted=[]
#for row in TSG_T_2_fin_gene:
#    if row[0] in fin_selec[0]:
#        wanted.append(row)
##TSG_T_2_method_3_Hs.706868
#def method(i,wanted,fin_selec):
#    chk_dir ="C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/all_to_gether_finalised/with_UniGene"
#
#    os.chdir('/')
#    os.chdir(chk_dir)
#    f= open(''.join(['fixing_method',str(i),'.csv']),"w+")
#    f.write( 'Tier,\t'+'Gene_name,\t'+'Uni_gene_ID,\t' + 'Method_number,\t'+ 'OG,'+'\t'+ 'TSG,'+ '\t'+ 'Fusion,\t'+'Class_classified' + '\t, Class_census\n')
#    
#    name = 'TSG'
#    minimum_prob_decide =0.25
#    for ids in wanted:
#            #insert the group details
#        predicted = decider_prob(ids, minimum_prob_decide,i)
#        f.write('2,\t'+ uni_gene_to_id_gene_name(fin_selec,ids[0])+',\t' +  ids[0] +',\t'  +str(i)+',\t'+ str(round(ids[1][0],2))  +',\t' + str(round(ids[1][1],2))+',\t' + str(round(ids[1][2],2))+',\t'+predicted+',\t' + name+'\n' )
#    f.close() 
#method(1,wanted,fin_selec)
#method(2,wanted,fin_selec)
#method(3,wanted,fin_selec)