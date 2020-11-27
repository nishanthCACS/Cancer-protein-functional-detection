# -*- coding: utf-8 -*-
"""
Created on %15-May-2019(10.30Am)

@author: %A.Nishanth C00294860
"""
import os
import csv
import pickle
#%%
"""
For checking the counts
"""
def loading_train_test_information(name):
    site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
    
    os.chdir("/")
    os.chdir(site_dir)
    SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
    Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
    Thres_sat_pdb_list =  pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))
    
    clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
    os.chdir("/")
    os.chdir(clean_pikle_dir)
    if name=="ONGO":
        gene_test_details = ['Hs.731652','Hs.37003','Hs.165950','Hs.338207','Hs.431850','Hs.470316','Hs.486502','Hs.507590','Hs.515247','Hs.525622','Hs.631535','Hs.715871','Hs.365116']
    #    gene_test_details = ['Hs.165950','Hs.338207','Hs.431850','Hs.470316','Hs.486502','Hs.507590','Hs.515247','Hs.525622','Hs.631535','Hs.715871','Hs.365116']
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
    elif name =="TSG":
        gene_test_details = ['Hs.137510','Hs.271353','Hs.526879','Hs.194143','Hs.368367','Hs.430589','Hs.445052','Hs.461086','Hs.470174','Hs.515840','Hs.592082','Hs.654514','Hs.740407','Hs.82028']
        clean_unigene_details = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
    elif name =="Fusion":
        gene_test_details = ['Hs.596314','Hs.599481','Hs.210546','Hs.327736','Hs.487027','Hs.516111','Hs.732542']
        clean_unigene_details = Thres_sat_gene_list
        
    SITE_unigene_details_all=sum(SITE_unigene_details,[])
    Thres_sat_pdb_list_all=sum(Thres_sat_pdb_list,[])
    for pdb_id in SITE_unigene_details_all:
        if pdb_id not in Thres_sat_pdb_list_all:
            print('problemetric: ',pdb_id)
    train_unigene_details=[]       
    test_unigene_details=[]
    train_pdb_ids_details=[]
    test_pdb_ids_details=[]
    for i in range(0,len(Thres_sat_gene_list)):
        if Thres_sat_gene_list[i] in clean_unigene_details:
            if Thres_sat_gene_list[i] not in gene_test_details:
                if len(SITE_unigene_details[i])>0:
                    train_pdb_ids_details.append(SITE_unigene_details[i])
                    train_unigene_details.append(Thres_sat_gene_list[i])
            else:
                if len(SITE_unigene_details[i])>0:
                    test_pdb_ids_details.append(SITE_unigene_details[i])
                    test_unigene_details.append(Thres_sat_gene_list[i])
    train_pdb_ids_details_flaten=sum(train_pdb_ids_details,[])
    test_pdb_ids_details_flaten=sum(test_pdb_ids_details,[])
    return  train_unigene_details,test_unigene_details,train_pdb_ids_details_flaten,train_pdb_ids_details,test_pdb_ids_details,test_pdb_ids_details_flaten
name="ONGO"
#name="TSG"
#name="Fusion"
train_unigene_details,test_unigene_details,train_pdb_ids_details_flaten,train_pdb_ids_details,test_pdb_ids_details,test_pdb_ids_details_flaten= loading_train_test_information(name)
#pdbs=sum(pdbs,[])
for pdbs in train_pdb_ids_details_flaten:
    if '4MDQ'==pdbs:
        print('train 4mdq')
    if '721P'==pdbs:
        print('train 721p')
for pdbs in test_pdb_ids_details_flaten:
    if '721P'==pdbs:
        print('test 721p')
    if '4MDQ'==pdbs:
        print('test 4mdq')
#%%
os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/k_3_86/train_results')

name="ONGO"
#name="TSG"
#name="Fusion"
uni_gene_train = pickle.load(open( ''.join([name,"_clean_O_T_unigene.p"]), "rb" ) )
pdbs_train = pickle.load(open( ''.join([name,"_train_pdb_ids_details.p.p"]),  "rb"))      

#%%from class_probabilty_weigtht_initialize_gene_from_PDB_seq_length_clean import probability_weight_initialize_gene_from_PDB_seq_length_clean

clean_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/"
working_dir = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"
site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/"
SITE_MUST = True
clean=False

#name = "ONGO"
name="TSG"

saving_dir = "".join([working_dir, "/After_sucess/SITE_MUST_",name])
#
#"""
## to create the dataset for the Overlapping ONCO Vs TSG genes
#os.chdir('/')
#os.chdir(site_dir)
#uni_gene = pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ) )
#site_sat = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]),  "rb"))      
##% choose the wanted overlapping UniGenes
#for i in range(0,len(uni_gene)):
##    if uni_gene[i]=='Hs.732394':#ONGO
#    if uni_gene[i]=='Hs.734132':#TSG
#        overlapped_O_T=[site_sat[i]]
#pickle.dump(overlapped_O_T, open( ''.join([name,"_overlapped_O_T.p"]), "wb" ) ) 
#"""
#if not os.path.exists(saving_dir):
#   os.mkdir(saving_dir)
#
#obj = probability_weight_initialize_gene_from_PDB_seq_length_clean(working_dir,saving_dir,name,clean=clean,clean_dir=clean_dir,SITE_MUST=SITE_MUST,site_dir=site_dir)
#obj.method_whole_finalize()
#del obj
#"""
##os.chdir('/')
##os.chdir(saving_dir)
##test_unigene=['Hs.732394']#ONGO
##pickle.dump(test_unigene, open("ONGO_test_uni_gene.p", "wb" ))
##name = "TSG"
##saving_dir = "".join([working_dir, "/After_sucess/SITE_MUST_",name])
##os.chdir('/')
##os.chdir(saving_dir)
##test_unigene= ['Hs.734132']#TSG
##pickle.dump(test_unigene, open("TSG_test_uni_gene.p", "wb" ))
#"""
#%% split the Overalpping PDBs and unique PDBs sepertaely
working_dir_NN_prob = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/a_Results_after_success/clean_NN_results'
site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results/"

def pdb_retrieve(name):
    os.chdir('/')
    os.chdir(site_dir)
    pdb_details = pickle.load(open( ''.join([name,"_overlapped_O_T.p"]), "rb" ))
    pdb_details=pdb_details[0]
    return pdb_details

OG_pdb_details = pdb_retrieve("ONGO")
TSG_pdb_details = pdb_retrieve("TSG")
OG_only=[]
OT=[]
for pdb in OG_pdb_details:
    if pdb in TSG_pdb_details:
        OT.append(pdb)
    else:
        OG_only.append(pdb)
TSG_only=[]        
for pdb in TSG_pdb_details:
    if pdb not in OT:
        TSG_only.append(pdb)
os.chdir('/')
os.chdir(working_dir_NN_prob)
f = open('T_1_OT.csv')
csv_f = csv.reader(f)
stack=[]
pdb_ids=[]
probabilities=[]
chk=0
for row in csv_f:
    if chk==1:
        stack.append(row)
        pdb_ids.append(row[0])   
        probabilities.append([round(float(row[1]),2),round(float(row[2]),2),round(float(row[3]),2)])
    chk=1
ONGO_g='PIK3CA'
TSG_g='PIK3R1'
#%%
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
    
f= open("O_T_anotated.csv","w+")
f.write( 'Tier,\t'+'PDB_ID,\t'+'Gene_name,\t' + 'ONCO,'+'\t'+ 'TSG,'+ '\t'+ 'Fusion,\t'+'Class_classified,\t' +' Most_probable_class' + '\t, Assigned class for gene census\n')

for i in range(0,len(pdb_ids)):
    if pdb_ids[i] in OG_only:
        name_g=ONGO_g
        class_census="ONCO"
    elif pdb_ids[i] in TSG_only:
        name_g=TSG_g
        class_census="TSG"
    elif  pdb_ids[i] in OT:
        name_g='PIK3CA or PIK3R1'
        class_census="ONCO_TSG"
    else:
        raise ValueError("Some thing wrong")
    predicted = assign_classes(probabilities[i])
    most_likely =assign_single_class_as(probabilities[i])
    f.write('1,'+ pdb_ids[i]+','+ name_g + ',\t'+ str(probabilities[i][0])  +',\t' + str(probabilities[i][1])+',\t' + str(probabilities[i][2]) +',\t'+predicted+',\t'+most_likely+','+ class_census+'\n' )
f.close() 

