# -*- coding: utf-8 -*-
"""
Created on %(19-Jan-2019) 14.25Pm

@author: %A.Nishanth C00294860

Trying to tilt the surface atom to the main eigne value dirrection
"""

import os 
import csv
import math  
import copy
import shutil
import pickle

#Tier_1 used for craeting the training data
# to load the sitre information
site_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"

#%% load the genedetails to select the PDB data 
# count the number of PDBs for each gene
name = "ONGO"

os.chdir('/')
os.chdir(site_dir)

def intialised_part(name):
    """
    This function load the pikle information about SITE satisfied ids 
    and the threshold staisfied information
    """
    fin_PDB_ids = pickle.load(open( ''.join([name,"_gene_list_thres_sat_PDB_ids.p"]), "rb" ))    
    fin_uni_gene = pickle.load(open( ''.join([name,"_thres_sat_gene_list.p"]), "rb" ))  
    gene_sat = pickle.load(open( ''.join([name,"_SITE_satisfied.p"]),  "rb"))                               
                
    # then count the occurance of number of PDBs in this time
    site_sat_occurance_counter = []
    for k in gene_sat:
        site_sat_occurance_counter.append(len(k))
    thrs_sat_occurance_counter = []
    for k in fin_PDB_ids:
        thrs_sat_occurance_counter.append(len(k))
        
    taken_unigenes = []
    taken_PDB_ids = []
    for i in range(0,len(site_sat_occurance_counter)):
        if site_sat_occurance_counter[i]>0:
            taken_unigenes.append(fin_uni_gene[i])
            taken_PDB_ids.append(fin_PDB_ids[i])
    return taken_unigenes, taken_PDB_ids     
#    return  fin_PDB_ids, fin_uni_gene, gene_sat, site_sat_occurance_counter, thrs_sat_occurance_counter 
            
#%% load the information about ONGO, TSG and Fusion data
taken_unigenes, taken_PDB_ids = intialised_part("ONGO") 

#ONGO_fin_PDB_ids, ONGO_fin_uni_gene,  ONGO_gene_sat, ONGO_site_sat_occurance_counter,  ONGO_thrs_sat_occurance_counter = intialised_part("ONGO") 
#TSG_fin_PDB_ids, TSG_fin_uni_gene,  TSG_gene_sat, TSG_site_sat_occurance_counter,  TSG_thrs_sat_occurance_counter = intialised_part("TSG") 
#Fusion_fin_PDB_ids, Fusion_fin_uni_gene,  Fusion_gene_sat, Fusion_site_sat_occurance_counter,  Fusion_thrs_sat_occurance_counter = intialised_part("Fusion")      
#%%
"""
Then go through the PDB_ids and check the SITE_infomation location

"""
from class_site_info_vin_1  import site_info_vin_1

name = "ONGO"
pdb_source_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/",name, "_R"])
#    saving_dir
pdb_name = "3H9R"
chk = site_info_vin_1(pdb_source_dir, pdb_name)
coordinates, amino_acid, c_alpha_indexes_MOTIF,c_alpha_indexes_MOTIF_grouped = chk.retrieve_c_alpha()
#%%%
#class alpha_C_surface_results:
"""
To produce the results of surface alpha carbon properties from the given pdb_file
"""
#    def __init__(self,pdb_source_dir,saving_dir,pdb_name):
name = "ONGO"
pdb_source_dir = ''.join(["C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/",name, "_R"])
#    saving_dir
pdb_name = "121p"
#        self.saving_dir = saving_dir
#        self.pdb_name  = pdb_name #name of the class

# first load the files needed
os.chdir('/')
os.chdir(pdb_source_dir)   
print("PDB_id_working_on: ",pdb_name)
pdb_name_open = ''.join([pdb_name,".pdb"])
f = open(pdb_name_open)
csv_f = csv.reader(f)
stack=[]
for row in csv_f:
    for s in row:
        stack.append(s)
#self.stack = stack     
#%%       
#def retrieve_c_alpha(self,stack):
def retrieve_c_alpha(stack):
    """
    to access the coordinate information of c-alpha carbon
    
    returns
    coordinate: normalised coordinates of C_Alpha
    amino_acid: amino acids for the C_Alpha back bone
    
    """
    info_all_cordinates = []
    coordinates = []
    residue_info = []
    resSeq = []      # Residue sequence number
    chain_id_info = []
    
    #% to access the given SITE information from PDB 
    res_site_temp = []
    chain_site_temp = []
    res_seq_site_temp = [] 
    amino_acid = []
#    resolution_PDB  = 1.0
    site_cond = False
    for row in stack:
        temp = row.split()
        if row[0:21] == 'REMARK   2 RESOLUTION':
            # to access the resolution details
            resolution_PDB = float(temp[3])
#                print("Resolution: ", float(temp[3]))
        elif row[0:4] == "ATOM":
            info_all_cordinates.append(row)
            if row[13:15] == "CA": 
                # to make normalised coordinates without affected buy the resolution
                coordinates.append([float(row[30:38])/resolution_PDB,float(row[38:46])/resolution_PDB,float(row[46:54])/resolution_PDB])
                #to access the chain ID and residue information to check the SITE info annotation of alpha carbon
                residue_info.append(row[17:20])
#                amino_acid.append(self.map_PDB_res_simple_amino_acid_format(row[17:20]))
                amino_acid.append(map_PDB_res_simple_amino_acid_format(row[17:20]))

                chain_id_info.append(row[21])
                resSeq.append(int(row[22:26]))
        elif row[0:4] == "SITE":        
            """ This information is useeful later when you need ed to find the CA with SITE info"""   
#                res_site_t, chain_site_t, res_seq_site_t = self.helper_func_retrieve_site_info(row[18:len(row)])
#                res_site_temp.append(res_site_t)
#                chain_site_temp.append(chain_site_t)
#                res_seq_site_temp.append(res_seq_site_t) 
#                site_cond = self.helper_func_retrieve_site_info(row[18:len(row)])
#        if site_cond:
#            print("Satisfied_site_ID")
#        res_site = []
#        chain_site = []
#        res_seq_site = []  
#        for i in range(0,len(res_site_temp)):
#            for j in range(0,len(res_site_temp[i])):
#                res_site.append(res_site_temp[i][j])
#                chain_site.append(chain_site_temp[i][j])
#                res_seq_site.append(res_seq_site_temp[i][j])
    return coordinates,amino_acid
#%%    
#def create_results(self):
"""
This function is mainly used for produce the results files
"""
#    stack = self.stack
#    coordinates,amino_acid = self.retrieve_c_alpha(stack)
coordinates,amino_acid = retrieve_c_alpha(stack)

#finalised_polar, cen_coordinates = self.change_C_alpha_cor_polar_cor(coordinates)
finalised_polar, cen_coordinates = change_C_alpha_cor_polar_cor(coordinates)
"""
inorder to tilt in the highest eigen value direction first centralize the coordinates
then find the eigen value and eigen vector for the changed coordinates
then find the direction of highest eigen value and then place that axis as x-y axis
(add or negate the angles polar coordiantes to change the coordinates)
"""
import numpy as np
from numpy import linalg as LA

coordinates_np = np.zeros((len(cen_coordinates),3))
for i in range(0,len(cen_coordinates)):
    for j in range(0,3):
        coordinates_np[i][j] = cen_coordinates[i][j]
# calculate the covariance matrix to find the eigen vector and eigen value
coor_cov = np.cov(coordinates_np.T)
w, v = LA.eig(coor_cov)
#% find the highest eigen value first
maxi_direction = v[:,w.argmax()]           
eig_xy_r, eig_xy_angle = cart2pol(maxi_direction[0], maxi_direction[1])   
eig_xyz_r, eig_xy_z_angle = cart2pol(eig_xy_r, maxi_direction[2]) 
#%%

#%%    

#resolution=1# angle degree of resolution
#    selected_calphas = self.surface_select_atom(finalised_polar,resolution)# thsi contain the surface selected atoms    
#    s_coordinates = self.shift_coordinate(coordinates)
#    res_factor = 2.25 # see the documentation twhy 2.25 is chosen
#    #% to find out the surface atoms residues 
#    sur_res = []
#    sur_res_cor = []
#    for i in selected_calphas:
#        sur_res.append(amino_acid[i])
#         # multiply each coordinate by 2 (just for increasing the resolution) and then round them to decimal numbers.
#        sur_res_cor.append([round(res_factor*s_coordinates[i][0]),round(res_factor*s_coordinates[i][1]),round(res_factor*s_coordinates[i][2])])
#       
#    #% then assign the property to the corresponding aminoacids  
#    property_surface = []
#    for i in range(0,len(sur_res)):
#        if sur_res[i]=="M":
#            property_surface.append([0,1,0,0,1,0,1,0,0,0,0,0,1,0,0.207070707,0.079276773])
#        elif sur_res[i]=="R":
#            property_surface.append([1,0,0,0,0,1,0,0,0,0,1,0,0,1,0.146464646,0.065368567])
#        elif sur_res[i]=="K":
#            property_surface.append([1,0,0,0,0,1,0,0,0,0,1,0,0,1,0.747474747,1])
#        elif sur_res[i]=="D":
#            property_surface.append([1,0,0,0,0,1,0,0,0,1,0,1,0,0,0.404040404,0.02364395])
#        elif sur_res[i]=="E":
#            property_surface.append([1,0,0,0,0,1,0,0,0,1,0,1,0,0,0.439393939,0.066759388])
#        elif sur_res[i]=="Q":
#            property_surface.append([0,1,0,0,0,1,1,0,0,0,0,0,1,0,0.166666667,0.063977747])
#        elif sur_res[i]=="N":
#            property_surface.append([0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,0.043115438])
#        elif sur_res[i]=="H":
#            property_surface.append([0,1,0,0,1,0,0,0,0,0,1,0,0,1,0.085858586,0.009735744])
#        elif sur_res[i]=="S":
#            property_surface.append([0,1,0,0,0,1,1,0,0,0,0,0,1,0,0.176767677,0.069541029])
#        elif sur_res[i]=="T":
#            property_surface.append([0,1,0,0,0,1,1,0,0,0,0,0,1,0,0.161616162,0.061196106])
#        elif sur_res[i]=="Y":
#            property_surface.append([0,1,0,1,0,0,0,1,0,0,0,0,1,0,0.156565657,0.068150209])
#        elif sur_res[i]=="C":
#            property_surface.append([0,1,0,0,1,0,1,0,0,0,0,0,1,0,1,0])
#        elif sur_res[i]=="W":
#            property_surface.append([0,1,0,1,0,0,0,1,0,0,0,0,1,0,0.297979798,0.093184979])
#        elif sur_res[i]=="A":
#            property_surface.append([0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.54040404,0.089012517])
#        elif sur_res[i]=="I":
#            property_surface.append([0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.484848485,0.084840056])
#        elif sur_res[i]=="L":
#            property_surface.append([0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.404040404,0.090403338])
#        elif sur_res[i]=="F":
#            property_surface.append([0,0,1,1,0,0,0,1,0,0,0,0,1,0,0.222222222,0.121001391])
#        elif sur_res[i]=="V":
#            property_surface.append([0,0,1,1,0,0,0,0,1,0,0,0,1,0,0.464646465,0.080667594])
#        elif sur_res[i]=="P":
#            property_surface.append([0,0,1,1,0,0,0,0,0,0,0,0,1,0,0.909090909,0.038942976])
#        elif sur_res[i]=="G":
#            property_surface.append([0,0,1,1,0,0,0,0,0,0,0,0,1,0,0.404040404,0.087621697])
#    
#    os.chdir("/")
#    os.chdir(self.saving_dir)
#    pbd_name = self.pdb_name
#    
#    f= open(''.join([pbd_name,'_xy.txt']),"w+")
#    f.write("charged(side chain can make salt bridges) , Polar(usually participate in hydrogen bonds as proton donnars & acceptors)")
#    f.write(" , Hydrophobic(normally burried inside the protein core) , Hydrophobic , Moderate , Hydrophillic , polar")
#    f.write(" Aromatic , Aliphatic , Acid , Basic , negative charge , Neutral , positive charge , Pka_NH2 , P_ka_COOH , x , y"+'\n')
#    for i in range(0,len(property_surface)):
#        for k in property_surface[i]:
#            f.write(str(k) + ",")
#        f.write(str(sur_res_cor[i][0]) + "," + str(sur_res_cor[i][1]) +'\n')    
#    f.close()      
#   
#    f= open(''.join([pbd_name,'_yz.txt']),"w+")
#    f.write("charged(side chain can make salt bridges) , Polar(usually participate in hydrogen bonds as proton donnars & acceptors)")
#    f.write(" , Hydrophobic(normally burried inside the protein core) , Hydrophobic , Moderate , Hydrophillic , polar")
#    f.write(" Aromatic , Aliphatic , Acid , Basic , negative charge , Neutral , positive charge , Pka_NH2 , P_ka_COOH , y , z"+'\n')
#    for i in range(0,len(property_surface)):
#        for k in property_surface[i]:
#            f.write(str(k) + ",")
#        f.write(str(sur_res_cor[i][1]) + "," + str(sur_res_cor[i][2]) +'\n')    
#    f.close()      
#    
#    f= open(''.join([pbd_name,'_xz.txt']),"w+")
#    f.write("charged(side chain can make salt bridges) , Polar(usually participate in hydrogen bonds as proton donnars & acceptors)")
#    f.write(" , Hydrophobic(normally burried inside the protein core) , Hydrophobic , Moderate , Hydrophillic , polar")
#    f.write(" Aromatic , Aliphatic , Acid , Basic , negative charge , Neutral , positive charge , Pka_NH2 , P_ka_COOH , x , z"+'\n')
#    for i in range(0,len(property_surface)):
#        for k in property_surface[i]:
#            f.write(str(k) + ",")
#        f.write(str(sur_res_cor[i][0]) + "," + str(sur_res_cor[i][2]) +'\n')    
#    f.close()
#%%
    # coordinate related functions
def change_C_alpha_cor_polar_cor(coordinates):
    """
    then change the C-alpha coordinates to polar coordinates
    
    returns 
    coordinates    : cetralised coordinates
    finalised_polar: polar_coordinates from the cetralised coordinates
    
    """
    cen_coordinates = copy.deepcopy(coordinates)
    
    g_x = 0
    g_y = 0
    g_z = 0
    
    x =[]
    y=[]
    z=[]
    for i in range(0,len(coordinates)):
        #find the center of gravity
         g_x = g_x + coordinates[i][0]       
         g_y = g_y + coordinates[i][1]
         g_z = g_z + coordinates[i][2]    
         
         x.append(coordinates[i][0])
         y.append(coordinates[i][1])
         z.append(coordinates[i][2])
    
    #% then centralize the coordinates
    for i in range(0,len(coordinates)):
        #find the center of gravity
         cen_coordinates[i][0]  = coordinates[i][0] - g_x/len(coordinates)    
         cen_coordinates[i][1]  = coordinates[i][1] - g_y/len(coordinates)    
         cen_coordinates[i][2]  = coordinates[i][2] - g_z/len(coordinates)       
    
    #% first create an empty matrices to hold the data
    xy_polar = []#here first coloumn contain the radius and the second coloumn contain angle
    for i in range(0,len(coordinates)):
        # find the radius of x-y coordinate
        xy_polar.append(cart2pol(cen_coordinates[i][0] ,cen_coordinates[i][1]))
    
    #then xy to z coordinate angle
    xy_z_polar = []#here first coloumn contain the radius and the second coloumn contain angle
    for i in range(0,len(coordinates)):
        # find the radius of xy-to z coordinate
        #take the xy radius
        xy_z_polar.append(cart2pol(xy_polar[i][0],cen_coordinates[i][2]))
    
    finalised_polar = []
    for i in range(0,len(coordinates)):
        finalised_polar.append([xy_z_polar[i][0], xy_polar[i][1], xy_z_polar[i][1]])

    return finalised_polar, cen_coordinates

def shift_coordinate(coordinates):
    """ 
    find the minimum position to move this shift the coordinates to make NN-train
    """
    s_coordinates = copy.deepcopy(coordinates)
    m_x = coordinates[0][0]
    m_y = coordinates[0][1] 
    m_z = coordinates[0][2]     
    for i in range(0,len(coordinates)):
        if coordinates[i][0] < m_x:
            m_x = coordinates[i][0]
        if coordinates[i][1] < m_y:
            m_y = coordinates[i][1]
        if coordinates[i][2] < m_z:
            m_z = coordinates[i][2]
    # this will linearly change the position
    for i in range(0,len(coordinates)):
         s_coordinates[i][0]  = coordinates[i][0] - m_x 
         s_coordinates[i][1]  = coordinates[i][1] - m_y
         s_coordinates[i][2]  = coordinates[i][2] - m_z
    return s_coordinates


def cart2pol(x, y):
    """
    This fucntion calculates the radius from given 2 coordinates in 
    """
    r = math.sqrt(x**2 + y**2)
    if (x>0):
      t = math.atan(y/x)# check it is in the all region quadrant
    elif (x<0 and y>0):
      t = math.pi + math.atan(y/x)# check it is in the second
    else:
      t = -math.pi+ math.atan(y/x)# check it is in the second
    return r,t     


#%%
    def copy_pdb_results_to_destination(self, dest):
        """
        This function copy the results files of pdbs to the given location
        dest: Destination wanted to copy the file
        """
        pdb_name = self.pdb_name
#        file_xy = ''.join([pdb_name,'.pdb_xy.txt'])
#        file_yz = ''.join([pdb_name,'.pdb_yz.txt'])
#        file_xz = ''.join([pdb_name,'.pdb_xz.txt'])
        file_xy = ''.join([pdb_name,'_xy.txt'])
        file_yz = ''.join([pdb_name,'_yz.txt'])
        file_xz = ''.join([pdb_name,'_xz.txt'])
        shutil.copy(file_xy, dest)          
        shutil.copy(file_yz, dest)    
        shutil.copy(file_xz, dest)    
#        print("copied_success_fully :)")


    def atom_angle_range(self, angle_chk,resolution):
        """
        This function will give the range where the atom belongs
        """
        range_xy_angle = []
        angle_list_rad = [x*math.pi/180 for x in list(range(-180, 181, 2*resolution))]
        for i in range(0,len(angle_chk)):
            for j in range(0,len(angle_list_rad)-1):
                if(angle_list_rad[j]<= angle_chk[i] and angle_chk[i]<angle_list_rad[j+1]):
                    range_xy_angle.append(j)
        return range_xy_angle
        
    
    def surface_select_atom(self, finalised_polar, resolution):
          # this function find out the surface atoms
        angle_xy = [finalised_polar[x][1] for x in range(0,len(finalised_polar))]
        angle_xy_z= [finalised_polar[x][2] for x in range(0,len(finalised_polar))]
        """
        fixing the ranging issue by eigen vector
        
        """
        angle_xy = [finalised_polar[x][1]-eig_xy_angle for x in range(0,len(finalised_polar))]
        for i in range(0,len(finalised_polar)):
            #inorder to fix the anlgles fell between -180 to 180 degrees
            if angle_xy[i] < 0 and angle_xy[i] < -math.pi:
                angle_xy[i] = angle_xy[i] + 2*math.pi 
            if angle_xy[i] > 0 and angle_xy[i] > math.pi:
                angle_xy[i] = angle_xy[i] - 2*math.pi 
        angle_xy_z= [finalised_polar[x][2]-eig_xy_z_angle for x in range(0,len(finalised_polar))]  
        for i in range(0,len(finalised_polar)):
            #inorder to fix the anlgles fell between -180 to 180 degrees
            if angle_xy_z[i] < 0 and angle_xy_z[i] < -math.pi:
                angle_xy_z[i] = angle_xy_z[i] + 2*math.pi 
            if angle_xy_z[i] > 0 and angle_xy_z[i] > math.pi:
                angle_xy_z[i] = angle_xy_z[i] - 2*math.pi 
                

        range_of_xy=self.atom_angle_range(angle_xy,2*resolution)
        range_of_xy_z=self.atom_angle_range(angle_xy_z,2*resolution)
        radius=[finalised_polar[x][0] for x in range(0,len(finalised_polar))]
        way_1_sel_atoms = sel_atom_helper_way(radius,range_of_xy,range_of_xy_z)
        
        # shift the angle by resolution to find the left surface atoms
        angle_xy = [finalised_polar[x][1]-eig_xy_angle-resolution for x in range(0,len(finalised_polar))]
        for i in range(0,len(finalised_polar)):
            #inorder to fix the anlgles fell between -180 to 180 degrees
            if angle_xy[i] < 0 and angle_xy[i] < -math.pi:
                angle_xy[i] = angle_xy[i] + 2*math.pi 
            if angle_xy[i] > 0 and angle_xy[i] > math.pi:
                angle_xy[i] = angle_xy[i] - 2*math.pi 
        
        angle_xy_z = [finalised_polar[x][1]-eig_xy_angle-resolution for x in range(0,len(finalised_polar))]
        for i in range(0,len(finalised_polar)):
            #inorder to fix the anlgles fell between -180 to 180 degrees
            if angle_xy_z[i] < 0 and angle_xy_z[i] < -math.pi:
                angle_xy_z[i] = angle_xy_z[i] + 2*math.pi 
            if angle_xy_z[i] > 0 and angle_xy_z[i] > math.pi:
                angle_xy_z[i] = angle_xy_z[i] - 2*math.pi      
                
        way_2_sel_atoms = sel_atom_helper_way(radius,range_of_xy,range_of_xy_z)
        # then combine bothways surface atoms and return
        sel_atoms= list(set(way_1_sel_atoms + way_2_sel_atoms))   
        return sel_atoms

    
    def sel_atom_helper_way(radius,range_of_xy,range_of_xy_z):
        # then just check the
        checked_range_element = []
        selected_atom = []
        for i in range(0,len(range_of_xy)):
            r=radius[i]
            s=i#to hold the selected item in surface
            if i not in checked_range_element:
              checked_range_element.append(i)
              for j in range(0,len(range_of_xy)):
                if(range_of_xy[i]==range_of_xy[j]):
                  if(range_of_xy_z[i]==range_of_xy_z[j]):
                    checked_range_element.append(j)
                    if(r<radius[j]):
                      r=radius[j]
                      s=j
              selected_atom.append(s)
        return selected_atom

    
    # SITE information functions
    def map_PDB_res_simple_amino_acid_format(self,given):
        """
        residue or aminoacid
        """
        pdb_residue_format = ["ARG","LYS","ASP","GLU","GLN","ASN","HIS","SER","THR","TYR",
                              "CYS","MET","TRP","ALA", "ILE","LEU","PHE","VAL","PRO","GLY"]
        aminoacid_format = ["R","K","D","E","Q","N","H","S","T","Y",
                                "C","M","W","A","I","L","F","V","P","G"]
        for i in range(0,len(pdb_residue_format)):
            if pdb_residue_format[i] == given:
                return aminoacid_format[i]


    def helper_func_retrieve_site_info(self,whole):  
        """
        Where the whole has the information of one row of given PDB's SITEs 
        eg: PDB file has like this
                "SITE     7 AC2 28 HOH A 175  HOH A 186  HOH A 188  HOH A 289           "
            then the whole conatians
                "HOH A 175  HOH A 186  HOH A 188  HOH A 289           "
                
        This function uses the space to seperate the information from the data
        """      
        res_site_t = []
        chain_site_t = []
        res_seq_site_t = []
        """ check the whole code again"""
        chk = whole.split()
#        print(chk)
#        print(len(chk))
        if len(chk) % 3 == 0:
            for i in range(0,len(chk),3):
                 res_site_t.append(chk[i])
                 chain_site_t.append(chk[i+1])
                 if len(chk[i+1]) > 1:
                     res_seq_site_t.append(int(''.join([chk[i+1][1:len(chk[i+1])]])))
                 else:
                     res_seq_site_t.append(int(chk[i+2]))
            return True
        else:
            return False
#            while i< len(whole)-1:
#                if whole[i] == " " and whole[i+2] == " ":
#                    i= len(whole)
#                else:
#                    res_site_t.append(whole[i:i+3])
#                    chain_site_t.append(whole[i+4])
#                    k = 6
#                    while whole[i+k] != " ":
#                        k = k + 1
#                    res_seq_site_t.append(int(whole[i+6:i+k]))
#                    i = i+k+2
#        return res_site_t, chain_site_t, res_seq_site_t     

