# -*- coding: utf-8 -*-
"""
Created on %23-Feb-2019(12.29 P.m)
@author: %A.Nishanth C00294860
"""
import os
import pickle
class surface_msms_depth_class:
    """
    USE MSMS tool(1996) to calculte the soluable access area of the surface
        depth of the C_alpha carbons, and 
              of the residue 
                          from the surface
    
    """
    def __init__(self, loading_pikle_dir, saving_dir, pdb_name,thresh_hold_ca=7.2,thresh_hold_res=6.7):
        self.saving_dir = saving_dir
        self.pdb_name  = pdb_name #name of the class

        os.chdir('/')
        os.chdir(loading_pikle_dir)
        coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
        aminoacids = pickle.load( open( ''.join(["amino_acid_",pdb_name,".p"]), "rb" ))
        fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
        fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
        coordinates_selected=[]
        corresponding_aminoacids=[]
        for i in range(0,len(fin_res_depth_all)):
            if fin_ca_depth_all[i] <= thresh_hold_ca:
                if fin_res_depth_all[i] <= thresh_hold_res:
                    coordinates_selected.append(coordinates[i])
                    corresponding_aminoacids.append(aminoacids[i])
                 
        res_factor = 2.25 # see the documentation twhy 2.25 is chosen
        #% to find out the surface atoms residues 
        sur_res = []
        sur_res_cor = []
        MOTIF_prop =[]
        for i in selected_calphas:
            sur_res.append(amino_acid[i])
             # multiply each coordinate by 2 (just for increasing the resolution) and then round them to decimal numbers.
            sur_res_cor.append([round(res_factor*s_coordinates[i][0]),round(res_factor*s_coordinates[i][1]),round(res_factor*s_coordinates[i][2])])
            if i in c_alpha_indexes_MOTIF:
                MOTIF_prop.append(1)
            else:
                MOTIF_prop.append(0)
        #% then assign the property to the corresponding aminoacids  
        property_surface = []
        for i in range(0,len(sur_res)):
            if sur_res[i]=="M":
                property_surface.append([0,1,0,0,1,0,1,0,0,0,0,0,1,0,0.207070707,0.079276773])
            elif sur_res[i]=="R":
                property_surface.append([1,0,0,0,0,1,0,0,0,0,1,0,0,1,0.146464646,0.065368567])
            elif sur_res[i]=="K":
                property_surface.append([1,0,0,0,0,1,0,0,0,0,1,0,0,1,0.747474747,1])
            elif sur_res[i]=="D":
                property_surface.append([1,0,0,0,0,1,0,0,0,1,0,1,0,0,0.404040404,0.02364395])
            elif sur_res[i]=="E":
                property_surface.append([1,0,0,0,0,1,0,0,0,1,0,1,0,0,0.439393939,0.066759388])
            elif sur_res[i]=="Q":
                property_surface.append([0,1,0,0,0,1,1,0,0,0,0,0,1,0,0.166666667,0.063977747])
            elif sur_res[i]=="N":
                property_surface.append([0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0.043115438])
            elif sur_res[i]=="H":
                property_surface.append([0,1,0,0,1,0,0,0,0,0,1,0,0,1,0.085858586,0.009735744])
            elif sur_res[i]=="S":
                property_surface.append([0,1,0,0,0,1,1,0,0,0,0,0,1,0,0.176767677,0.069541029])
            elif sur_res[i]=="T":
                property_surface.append([0,1,0,0,0,1,1,0,0,0,0,0,1,0,0.161616162,0.061196106])
            elif sur_res[i]=="Y":
                property_surface.append([0,1,0,1,0,0,0,1,0,0,0,0,1,0,0.156565657,0.068150209])
            elif sur_res[i]=="C":
                property_surface.append([0,1,0,0,1,0,1,0,0,0,0,0,1,0,1,0])
            elif sur_res[i]=="W":
                property_surface.append([0,1,0,1,0,0,0,1,0,0,0,0,1,0,0.297979798,0.093184979])
            elif sur_res[i]=="A":
                property_surface.append([0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.54040404,0.089012517])
            elif sur_res[i]=="I":
                property_surface.append([0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.484848485,0.084840056])
            elif sur_res[i]=="L":
                property_surface.append([0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.404040404,0.090403338])
            elif sur_res[i]=="F":
                property_surface.append([0,0,1,1,0,0,0,1,0,0,0,0,1,0,0.222222222,0.121001391])
            elif sur_res[i]=="V":
                property_surface.append([0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.464646465,0.080667594])
            elif sur_res[i]=="P":
                property_surface.append([0,0,1,1,0,0,0,0,0,0,0,0,1,0,0.909090909,0.038942976])
            elif sur_res[i]=="G":
                property_surface.append([0,0,1,1,0,0,0,0,0,0,0,0,1,0,0.404040404,0.087621697])
        
        os.chdir("/")
        os.chdir(self.saving_dir)
        pdb_name = self.pdb_name
        
        f= open(''.join([pdb_name,'_xy.txt']),"w+")
        f.write("charged(side chain can make salt bridges) , Polar(usually participate in hydrogen bonds as proton donnars & acceptors)")
        f.write(" , Hydrophobic(normally burried inside the protein core) , Hydrophobic , Moderate , Hydrophillic , polar")
        f.write(" Aromatic , Aliphatic , Acid , Basic , negative charge , Neutral , positive charge , Pka_NH2 , P_ka_COOH , MOTIF_sat, x , y"+'\n')
        for i in range(0,len(property_surface)):
            for k in property_surface[i]:
                f.write(str(k) + ",")
            f.write(str(MOTIF_prop[i]) + ",")
            f.write(str(sur_res_cor[i][0]) + "," + str(sur_res_cor[i][1]) +'\n')    
        f.close()      
           
        f= open(''.join([pdb_name,'_yz.txt']),"w+")
        f.write("charged(side chain can make salt bridges) , Polar(usually participate in hydrogen bonds as proton donnars & acceptors)")
        f.write(" , Hydrophobic(normally burried inside the protein core) , Hydrophobic , Moderate , Hydrophillic , polar")
        f.write(" Aromatic , Aliphatic , Acid , Basic , negative charge , Neutral , positive charge , Pka_NH2 , P_ka_COOH, MOTIF_sat,  y, z"+'\n')
        for i in range(0,len(property_surface)):
            for k in property_surface[i]:
                f.write(str(k) + ",")
            f.write(str(MOTIF_prop[i]) + ",")
            f.write(str(sur_res_cor[i][1]) + "," + str(sur_res_cor[i][2]) +'\n')    
        f.close()      
        
        f= open(''.join([pdb_name,'_xz.txt']),"w+")
        f.write("charged(side chain can make salt bridges) , Polar(usually participate in hydrogen bonds as proton donnars & acceptors)")
        f.write(" , Hydrophobic(normally burried inside the protein core) , Hydrophobic , Moderate , Hydrophillic , polar")
        f.write(" Aromatic , Aliphatic , Acid , Basic , negative charge , Neutral , positive charge , Pka_NH2 , P_ka_COOH, MOTIF_sat,  x , z"+'\n')
        for i in range(0,len(property_surface)):
            for k in property_surface[i]:
                f.write(str(k) + ",")
            f.write(str(MOTIF_prop[i]) + ",")
            f.write(str(sur_res_cor[i][0]) + "," + str(sur_res_cor[i][2]) +'\n')    
        f.close()           