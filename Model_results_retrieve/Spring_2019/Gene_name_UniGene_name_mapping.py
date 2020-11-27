# -*- coding: utf-8 -*-
"""
Created on %08-Feb-2018(14.55pm)

@author: %A.Nishanth C00294860
"""

import os
import csv

os.chdir("/")
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Paper_writing/Comparison_paper_results/Mapping_uniprot_mine')

f = open('HIconfig_ONGO_uniprot.csv')
#f = open('HIconfig_TSG_uniprot.csv')

csv_f = csv.reader(f)
stack=[]
for row in csv_f:
    for s in row:
        stack.append(s)
#%%
gene_name =[]
uni_gene =[] 
for j in range(0,len(stack)):
    if stack[j][0:2]=="Hs":
        gene_name.append(stack[j-1])
        uni_gene.append(stack[j])
#%% then group the uni_genes with the gene_name
#"ONGO"#
unique_gene_name_given =["ABL1","AKT1","AKT2","ALK","BCL6","BRAF","CARD11","CCNE1","CTNNB1","EGFR","ERBB2","EZH2","FAS","FGFR2","FGFR3","FLT3","GNA11","GNAQ","GNAS","HRAS","IDH1","JUN","KDR","KIT","KRAS","MAP2K1","MAP2K2","MAP2K4","MDM2","MDM4","MET","MITF","MLL","MYC","MYCL1","MYCN","MYD88","NFE2L2","NKX2-1","NRAS","PDGFRA","PIK3CA","REL","RET","RNF43","SMO","SOX2","STAT3","TERT","TRAF7","TSHR"]
#TSG
#unique_gene_name_given =["AMER1","APC","ATM","AXIN1","BAP1","BRCA1","BRCA2","CDH1","CDKN2A","CDKN2C","CEBPA","CREBBP","CYLD","DICER1","EP300","FBXW7","GATA3","HNF1A","KDM6A","MAX","MEN1","MLH1","MSH2","MSH6","NF1","NF2","NOTCH1","NOTCH2","PAX5","PIK3R1","PRKAR1A","PTCH1","PTEN","RB1","SETD2","SMAD4","SMARCA4","SMARCB1","SOCS1","STK11","SUFU","TET2","TNFAIP3","TP53","TSC1","TSC2","VHL","WT1"]
unique_gene_name_t=list(set(gene_name))
unique_gene_name=[]
for n in unique_gene_name_t:
    unique_gene_name.append(n.split())
print("check the error avilable")
unique_gene_name_all=list(set(sum(unique_gene_name, [])))
for n in unique_gene_name_given:
    if n not in unique_gene_name_all:
        print(n)
#%%
sel_unigene = []
for n in unique_gene_name_given:
    matched=[]
    for i in range(0,len(unique_gene_name)):
        if n in unique_gene_name[i]:
            matched.append(unique_gene_name_t[i])
    uni_gene_t=[]
    for i in range(0,len(gene_name)):
        if gene_name[i] in matched:
            uni_gene_t.append(uni_gene[i])
    uni_gene_t=list(set(uni_gene_t))
    sel_unigene.append(uni_gene_t)
#%% cleaning the UniGene details
fin_unigene =[]
for l in sel_unigene:
    uni_gene_t=[]
    for i in range(0,len(l)):
        k= l[i].split(";")
        for m in k:
            if m[0:2]=="Hs":
                uni_gene_t.append(m)
    uni_gene_t=list(set(uni_gene_t))
    fin_unigene.append(uni_gene_t)
#%%
           
f= open('ONGO_Hiconfig_gene_name_to_uni_gene.txt',"w+")
#f= open('TSG_Hiconfig_gene_name_to_uni_gene.txt',"w+")

f.write("name, UniGenes"+'\n')    
for i in range(0,len(unique_gene_name_given)):
    f.write(unique_gene_name_given[i])
    for k in fin_unigene[i]:
        f.write( "," + k)
    f.write('\n')    
f.close()      
#import pickle
#pickle.dump(fin_unigene, open("TSG_mapped_uni_gene.p", "wb" )) 
#pickle.dump(unique_gene_name_given, open("TSG_unique_gene_name_given.p", "wb" )) 
#pickle.dump(fin_unigene, open("ONGO_mapped_uni_gene.p", "wb" )) 
#pickle.dump(unique_gene_name_given, open("ONGO_unique_gene_name_given.p", "wb" )) 
#%% 
"""CHECKING THE TRAINING AND TETSING SET HOWMANY OF THEM ARE IN THE HICONFIG"""
import os
import pickle

map_hiconfig_dir = 'C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Paper_writing/Comparison_paper_results/Mapping_uniprot_mine'
clean_pikle_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Spring_2019/results"
name = "ONGO"
#name = "TSG"

os.chdir("/")
os.chdir(map_hiconfig_dir)
fin_unigene = pickle.load(open( ''.join([name,"_mapped_uni_gene.p"]), "rb" ))
unique_gene_name_given = pickle.load(open( ''.join([name,"_unique_gene_name_given.p"]), "rb" ))

os.chdir("/")
os.chdir(clean_pikle_dir)
#    SITE_unigene_details = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ) )
Thres_sat_gene_list =  pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))
#%%
def comparing_the_given_and_ours(unique_gene_name_given,fin_unigene,our_uni_gene_list):
    """
    unique_gene_name_given  : Thier HiConfig gene names
    fin_unigene             : correspoinding UniGene id mapped for that given
    
    outputs
    missed_uni_genes: our genes which are missed in their given HiConfig 
    sat_gene_list   : The genes overlapped between their and ours
    sat_uni_genes   : Corresponding UniGene id in ours
    
    """    
    sat_gene_list=[]
    sat_uni_genes=[]
    missed_uni_genes=[]
    for n in our_uni_gene_list:
        c=[]
        for i in range(0,len(unique_gene_name_given)):
            if n in fin_unigene[i]:
                c.append(unique_gene_name_given[i])
        if len(c)>0:
            sat_gene_list.append(c)
            sat_uni_genes.append(n)
        else:
            missed_uni_genes.append(n)
            
    return sat_gene_list,sat_uni_genes,missed_uni_genes
            
cor_thresh_sat_gene_list,cor_thresh_sat_uni_genes, missed_thresh_uni_genes= comparing_the_given_and_ours(unique_gene_name_given,fin_unigene,Thres_sat_gene_list)
#%%
def find_missed_genes_from_the_paper(cor_sat_gene_list,unique_gene_name_given):
    """
    cor_sat_gene_list       : the Gene names overlapped
    unique_gene_name_given  : HiConfig genes got from the paper
    
    our_missed_names_from_thier_classes: Genes which missed from their paper
    """
    cor_thresh_sat_gene_list_all=list(set(sum(cor_sat_gene_list, [])))
    our_missed_names_from_thier_classes=[]
    for n in unique_gene_name_given:
        if n not in cor_thresh_sat_gene_list_all:
            our_missed_names_from_thier_classes.append(n)
    return our_missed_names_from_thier_classes

our_missed_names_thresh_sat_from_thier_classes = find_missed_genes_from_the_paper(cor_thresh_sat_gene_list,unique_gene_name_given)    
#%% check howmany of them are there before threshold satisfied condition
"""
retrive all of our ids considerded
"""

loading_dir ="C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Uniprot_Mapping/Tier_1_pdb_pikles_length_gene"
os.chdir('/')
os.chdir(loading_dir)
#ids_all = pickle.load(open( ''.join([name,'_Ids_info_all.p']), "rb" ))
uni_gene_all = pickle.load(open( ''.join([name,'_uni_gene.p']), "rb" ))
cor_all_gene_list,cor_all_uni_genes, missed_all_uni_genes= comparing_the_given_and_ours(unique_gene_name_given,fin_unigene,uni_gene_all)
our_missed_names_all_from_thier_classes = find_missed_genes_from_the_paper(cor_all_gene_list,unique_gene_name_given)     
"""
the genes avoided by the fuiltering threshold and X-ray fragment conditions
"""
unsat_genes_missed=[]
for n in our_missed_names_thresh_sat_from_thier_classes:
   if n not in our_missed_names_all_from_thier_classes:
       unsat_genes_missed.append(n)
#%% 
"""Extract the censuz information"""
genes=["ACSL3","AFF1","ASPSCR1","ATIC","BCL7A","BCR","C2orf44","CANT1","CHCHD7","CLIP1","CNTRL","COL1A1","COL2A1","CRTC3","DCTN1","DNAJB1","EIF4A2","EML4","ERC1","EZR","FGFR1OP","FIP1L1","GAS7","GOLGA5","GOPC","GPHN","HERPUD1","HIST1H4I","HLA-A","HOOK3","HSP90AA1","HSP90AB1","IGH","IGK","IGL","IL2","IL21R","ITK","KDSR","KIF5B","KLK2","KTN1","LASP1","LIFR","LMNA","MLLT1","MLLT11","MLLT3","MLLT6","MSN","MUC1","MYH11","MYO5A","NCOA1","NFIB","NIN","NONO","NSD1","NUMA1","NUP214","NUTM2A","NUTM2B","PAFAH1B2","PAX7","PAX8","PCM1","PDE4DIP","PICALM","PPFIBP1","PRCC","PRRX1","RABEP1","RBM15","RNF213","RPN1","SDC4","SLC45A3","SS18","SS18L1","STRN","TCEA1","TCF12","TFG","TMPRSS2","TOP1","TPM4","TPR","TRA","TRB","TRD","TRIP11","ZNF198","ZNF384","ACVR1","AKT1","AKT2","AR","CACNA1D","CALR","CARD11","CCNE1","CD79A","CD79B","CDK4","CHD4","CSF3R","CXCR4","DDR2","EGFR","ERBB3","FGFR4","FLT3","FLT4","FOXA1","FUBP1","GATA2","GNA11","GNAQ","GNAS","H3F3A","H3F3B","HIF1A","HIST1H3B","HRAS","IDH1","IDH2","IKBKB","IL6ST","IL7R","JAK3","JUN","KCNJ5","KDR","KIT","KRAS","MAP2K1","MAP2K2","MAPK1","MDM2","MDM4","MET","MITF","MPL","MTOR","MYCL","MYCN","MYD88","MYOD1","NRAS","NT5C2","PIK3CA","PIK3CB","PPM1D","PREX2","PRKACA","PTPN11","RAC1","REL","SALL4","SF3B1","SIX1","SMO","SOX2","SRC","SRSF2","STAT3","TRRAP","TSHR","U2AF1","USP8","WAS","XPO1","ABL1","ABL2","ACKR3","AFF3","AFF4","ALK","ATF1","BCL11A","BCL2","BCL3","BCL6","BCL9","BRAF","BRD3","BRD4","CCND1","CCND2","CCND3","CD74","CDK6","CREB1","CREB3L2","CRLF2","CRTC1","CTNNB1","DDIT3","DDX5","DDX6","DEK","ELK4","ERBB2","ERG","ETV1","ETV4","ETV5","EWSR1","FCGR2B","FCRL4","FEV","FGFR1","FGFR2","FGFR3","FLI1","FOXP1","FSTL3","HEY1","HIP1","HLF","HMGA1","HMGA2","HNRNPA2B1","HOXA13","HOXC11","HOXC13","HOXD11","HOXD13","JAK2","KAT6A","KDM5A","KMT2A","LCK","LMO1","LMO2","LPP","LYL1","MAF","MAFB","MALT1","MAML2","MECOM","MLLT10","MLLT4","MN1","MSI2","MTCP1","MYB","MYC","NCOA2","NFATC2","NPM1","NR4A3","NTRK3","NUP98","NUTM1","OLIG2","P2RY8","PAX3","PBX1","PDCD1LG2","PDGFB","PDGFRA","PDGFRB","PIM1","PLAG1","PLCG1","POU2AF1","POU5F1","PRDM16","PSIP1","RAF1","RAP1GDS1","RARA","RET","ROS1","RSPO3","SET","SETBP1","SH3GL1","SND1","SRSF3","SSX1","SSX2","SSX4","STAT6","STIL","SYK","TAF15","TAL1","TAL2","TCF7L2","TCL1A","TFE3","TFEB","TLX1","TLX3","TNFRSF17","TRIM27","USP6","WHSC1","WHSC1L1","WWTR1","ZNF521","APOBEC3B","ATP1A1","BCL9L","BCORL1","BMPR1A","BTK","CBLC","CUX1","DAXX","DDB2","EPAS1","ERBB4","EZH2","FES","FOXL2","GATA1","GATA3","GPC3","IRS4","JAK1","KDM6A","KLF4","KMT2D","LEF1","MAP2K4","MAP3K1","MAP3K13","NFE2L2","NKX2-1","NOTCH2","POLQ","PTK6","QKI","RAD21","RECQL4","RHOA","TBX3","TERT","TP63","ARNT","BCL11B","BIRC3","CBL","CIC","CREBBP","ELF4","ESR1","FOXO1","FOXO3","FOXO4","HOXA11","HOXA9","IRF4","MKL1","NFKB2","NOTCH1","NTRK1","PAX5","PRKAR1A","RUNX1","RUNX1T1","STAT5B","SUZ12","TBL1XR1","TCF3","TET1","TP53","TRIM24","WT1","ACVR2A","AMER1","APC","ARID1B","ARID2","ASXL1","ATM","ATP2B3","ATR","ATRX","AXIN1","AXIN2","B2M","BAP1","BARD1","BAX","BLM","BRCA1","BRCA2","BRIP1","BUB1B","CASP8","CBLB","CDC73","CDH1","CDK12","CDKN1B","CDKN2A","CDKN2C","CEBPA","CHEK2","CNOT3","CTCF","CYLD","DDX3X","DICER1","DNM2","DNMT3A","DROSHA","ERCC2","ERCC3","ERCC4","ERCC5","ETNK1","EXT2","FAM46C","FANCA","FANCC","FANCD2","FANCE","FANCF","FANCG","FAS","FAT1","FAT4","FBXO11","FBXW7","FH","FLCN","GRIN2A","HNF1A","KDM5C","KEAP1","KLF6","KMT2C","LRP1B","LZTR1","MAX","MED12","MEN1","MLH1","MSH2","MSH6","MUTYH","NBN","NCOR1","NCOR2","NF2","NFKBIE","PALB2","PBRM1","PHF6","PHOX2B","PIK3R1","PMS2","POLD1","POLE","POT1","PPP2R1A","PPP6C","PRDM1","PRF1","PTCH1","PTEN","PTPN13","PTPRB","PTPRC","PTPRT","RB1","RBM10","RNF43","RPL10","RPL5","SBDS","SDHA","SDHAF2","SDHB","SDHC","SDHD","SETD2","SFRP4","SH2B3","SMAD2","SMAD3","SMAD4","SMARCA4","SMARCB1","SMARCD1","SMARCE1","SOCS1","SPEN","SPOP","STAG2","STK11","SUFU","TET2","TGFBR2","TMEM127","TNFAIP3","TNFRSF14","TRAF7","TSC1","TSC2","UBR5","VHL","WRN","XPA","XPC","ZFHX3","ZRSR2","ABI1","ARHGAP26","ARHGEF12","ARID1A","BCL10","BCOR","BTG1","CAMTA1","CARS","CASC5","CBFA2T3","CBFB","CCDC6","CCNB1IP1","CD274","CDH11","CDX2","CIITA","CLTC","CLTCL1","CNBP","CREB3L1","DDX10","EBF1","EIF3E","ELL","EP300","EPS15","ETV6","EXT1","FHIT","FUS","IKZF1","KAT6B","LRIG3","MLF1","MYH9","NAB2","NCOA4","NDRG1","NF1","NRG1","PER1","PML","PPARG","PTPRK","RAD51B","RANBP2","RHOH","RMI2","RPL22","RSPO2","SFPQ","SLC34A2","TPM3","TRIM33","WIF1","YWHAE","ZBTB16","ZNF278","ZNF331"]
funct=['fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','fusion','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','oncogene','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' fusion"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','"oncogene',' TSG',' fusion"','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','TSG','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"','"TSG',' fusion"']
fin_func=[]
i=0
while len(fin_func)!=len(genes):
    c=[]
    if '"' in funct[i]:
        c.append(funct[i][1:len(funct[i])])
        i=i+1
    if len(c)>0:
        while '"' not in funct[i]:
            c.append(funct[i])
            i=i+1

        c.append(funct[i][0:-1])
        fin_func.append(''.join(c))
    else:
        fin_func.append(funct[i])
    i=i+1
#%%
missed_our_class=[]
our_missed_names_all_from_thier_classes_fin=[]
for n in our_missed_names_all_from_thier_classes:
    if n not in genes:
        print("Missing in census: ",n)
    else:
        for i in range(0,len(genes)):
            if genes[i]==n:
                our_missed_names_all_from_thier_classes_fin.append(n)
                missed_our_class.append(fin_func[i])

        
        