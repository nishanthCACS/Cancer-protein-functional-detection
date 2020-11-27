# -*- coding: utf-8 -*-
"""
Created on %26-January-2018(11.11pm)


@author: %A.Nishanth C00294860
loading the experiment type details
"""
import pickle
import os
import csv

def pdb_details_fun(name,saving_dir):
    os.chdir("/")
    os.chdir(saving_dir)
#    experiment_type = pickle.load(open( ''.join([name,"_SITE_satisfied_experiment_type.p"]), "rb" ) )
    pdb_details =pickle.load(open( ''.join([name,"_SITE_satisfied.p"]), "rb" ))
    return pdb_details
#%% For Tier_1
name = "ONGO"
checking_dir_PDB = "C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/ONGO_R"
saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results"

pdb_details = pdb_details_fun(name,saving_dir)
#print("considering: ",pdb_details[9][0])
#pdb_name = pdb_details[9][0]
print("considering: ",pdb_details[0][0])
pdb_name = pdb_details[0][0]
#%% here onwards loading the PDB file and chk
os.chdir("/")
os.chdir(checking_dir_PDB)
pdb_name_open = ''.join([pdb_name,".pdb"])
f = open(pdb_name_open)
csv_f = csv.reader(f)
stack=[]
for row in csv_f:
    for s in row:
        stack.append(s)
#%%     
def map_PDB_res_simple_amino_acid_format(given):
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
        
def helper_func_retrieve_site_info(whole):  
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
        i = 0
        while i < len(chk)-1: 
             res_site_t.append(chk[i])
             if len(chk[i+1]) > 1:
                 chain_site_t.append(chk[i+1][0])
                 res_seq_site_t.append(''.join([chk[i+1][1:len(chk[i+1])]]))  
                 i = i + 2
             else:
                 chain_site_t.append(chk[i+1])
                 res_seq_site_t.append(chk[i+2])        
                 i = i + 3
        return res_site_t, chain_site_t, res_seq_site_t   
        
#%%        
"""Function:     retrieve_c_alpha"""
def retrieve_c_alpha(stack):
    info_all_cordinates = []
    coordinates = []
    residue_info = []
    resSeq = []      # Residue sequence number
    chain_id_info = []
    
    # site unewanted_ini
    un_wanted_site = []
    atom_all_residues = []
    atom_all_chains = []
    atom_all_res_seq = []
    #% to access the given SITE information from PDB 
    res_site_temp = []
    chain_site_temp = []
    res_seq_site_temp = [] 
    amino_acid = []
    #site_occurance_count = []
    SITE_id_info = []
    SITE_id_info_unique = []
    
    
    atom_number = 0
    atom_number_list_c =[]
    for row in stack:
        temp = row.split()
        if row[0:21] == 'REMARK   2 RESOLUTION':
            #to access the resolution details
            resolution_PDB = float(temp[3])
        if row[0:4] == "ATOM":
            atom_number = atom_number + 1
            info_all_cordinates.append(row)
            if row[13:15] == "CA": 
                atom_number_list_c.append(atom_number)
                # to make normalised coordinates without affected buy the resolution
                coordinates.append([float(row[30:38])/resolution_PDB,float(row[38:46])/resolution_PDB,float(row[46:54])/resolution_PDB])
                #to access the chain ID and residue information to check the SITE info annotation of alpha carbon
                residue_info.append(row[17:20].split()[0])
                amino_acid.append(map_PDB_res_simple_amino_acid_format(row[17:20]))
                chain_id_info.append(row[21])
                resSeq.append(row[22:27].split()[0]) # since some of the reduncdant there after
            if len(row)>=27:
                atom_all_residues.append(row[17:20].split()[0])
                atom_all_chains.append(row[21])
                atom_all_res_seq.append(row[22:27].split()[0]) 
        elif row[0:4] == "SITE":    
            # to check the SITE_id
            if temp[2] not in SITE_id_info_unique:
                SITE_id_info_unique.append(temp[2])
    #            site_occurance_count.append(int(temp[3]))
            SITE_id_info.append(temp[2])
            """ This information is useeful later when you need ed to find the CA with SITE info"""   
            
            res_site_t, chain_site_t, res_seq_site_t = helper_func_retrieve_site_info(row[18:len(row)])
            res_site_temp.append(res_site_t)
            chain_site_temp.append(chain_site_t)
            res_seq_site_temp.append(res_seq_site_t) 
            
        elif row[0:6] == "HETATM":
            un_wanted_site.append(row[17:20].split()[0])    
    
    SITE_id_info_all = []
    res_site = []
    chain_site = []
    res_seq_site = []  
    for i in range(0,len(res_site_temp)):
        for j in range(0,len(res_site_temp[i])):
            res_site.append(res_site_temp[i][j])
            chain_site.append(chain_site_temp[i][j])
            res_seq_site.append(res_seq_site_temp[i][j])
            SITE_id_info_all.append(SITE_id_info[i])
    #% then go through the RESIDUES in CA and check missed residues
    combined_CA_residue_all = []
    for i in range(0,len(residue_info)):
        combined_CA_residue_all.append(''.join([residue_info[i],"_",chain_id_info[i],"_",resSeq[i]]))
    
    combined_residue_all = []
    for i in range(0,len(atom_all_residues)):
        combined_residue_all.append(''.join([atom_all_residues[i],"_",atom_all_chains[i],"_",atom_all_res_seq[i]]))
    
    residue_not_in_c_alpha = [v for v in combined_residue_all if v not in combined_CA_residue_all]         
    # atoms indexes of the C-alpha satisified SITE conditions
    c_alpha_indexes_MOTIF = []
    un_sat_residues = []
    count_un_wanted = 0
    more_SITE_occurance = 0
    for i in range(0,len(res_site)):   
        occurance = 0
        un_satisfied = True     
        for j in range(0,len(coordinates)):
            if (res_site[i] == residue_info[j]) and chain_site[i] == chain_id_info[j] and res_seq_site[i] == resSeq[j]:
                c_alpha_indexes_MOTIF.append(j)
                occurance = occurance +1 
                un_satisfied = False
        if occurance > 1: 
            more_SITE_occurance = more_SITE_occurance + occurance-1
        if un_satisfied:
            un_sat_residues.append(i)
            if res_site[i] in un_wanted_site: 
                count_un_wanted = count_un_wanted + 1
            elif "".join([res_site[i],"_",chain_site[i],"_",res_seq_site[i]]) in residue_not_in_c_alpha:
                count_un_wanted = count_un_wanted + 1
        
    if len(res_site) == len(c_alpha_indexes_MOTIF) + count_un_wanted  - more_SITE_occurance:
    #            print("perfect :)")
        return coordinates,amino_acid, c_alpha_indexes_MOTIF
    else:
        print(" # res_site             : ", len(res_site))
        print(" count_un_wanted        : ", count_un_wanted)
        print(" # c_alpha_indexes_MOTIF: ", len(c_alpha_indexes_MOTIF) )
        print("more_SITE_occurance     : ",more_SITE_occurance)
        raise SystemExit("These counts should be same some thing wrong")         
        return False#, coordinates, amino_acid, c_alpha_indexes_MOTIF

#%% 
import copy
import math
import numpy as np
from numpy import linalg as LA

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
#            xy_polar.append(self.cart2pol(cen_coordinates[i][0] ,cen_coordinates[i][1]))
            xy_polar.append(cart2pol(cen_coordinates[i][0] ,cen_coordinates[i][1]))
        #then xy to z coordinate angle
        xy_z_polar = []#here first coloumn contain the radius and the second coloumn contain angle
        for i in range(0,len(coordinates)):
            # find the radius of xy-to z coordinate
            #take the xy radius
#            xy_z_polar.append(self.cart2pol(xy_polar[i][0],cen_coordinates[i][2]))
            xy_z_polar.append(cart2pol(xy_polar[i][0],cen_coordinates[i][2]))
        
        finalised_polar = []
        for i in range(0,len(coordinates)):
            finalised_polar.append([xy_z_polar[i][0], xy_polar[i][1], xy_z_polar[i][1]])
    
        return finalised_polar, cen_coordinates

#def atom_angle_range(self, angle_chk,resolution):
def atom_angle_range(angle_chk,resolution):

    """
    This function will give the range where the atom belongs
    """
    range_xy_angle = []
    angle_list_rad = [x*math.pi/180 for x in list(range(-180, 181, resolution))]
    for i in range(0,len(angle_chk)):
        for j in range(0,len(angle_list_rad)-1):
            if(angle_list_rad[j]<= angle_chk[i] and angle_chk[i]<angle_list_rad[j+1]):
                range_xy_angle.append(j)
    return range_xy_angle

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



def surface_select_atom(coordinates, resolution_xy, resolution_xy_z):
    """
    inorder to tilt in the highest eigen value direction first centralize the coordinates
    then find the eigen value and eigen vector for the changed coordinates
    then find the direction of highest eigen value and then place that axis as x-y axis
    (add or negate the angles polar coordiantes to change the coordinates)
    """
    finalised_polar, cen_coordinates = change_C_alpha_cor_polar_cor(coordinates)
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
        #inorder to fix the anlgles fell between -90 to 90 degrees
        if angle_xy_z[i] < 0 and angle_xy_z[i] < -math.pi/2:
            angle_xy_z[i] = angle_xy_z[i] + math.pi 
        if angle_xy_z[i] > 0 and angle_xy_z[i] > math.pi/2:
            angle_xy_z[i] = angle_xy_z[i] - math.pi 
            

    range_of_xy= atom_angle_range(angle_xy,2*resolution_xy)
    range_of_xy_z= atom_angle_range(angle_xy_z,2*resolution_xy_z)
#    range_of_xy=self.atom_angle_range(angle_xy,2*resolution_xy)
#    range_of_xy_z=self.atom_angle_range(angle_xy_z,2*resolution_xy_z)
    radius=[finalised_polar[x][0] for x in range(0,len(finalised_polar))]
    way_1_sel_atoms = sel_atom_helper_way(radius,range_of_xy,range_of_xy_z)
    
    # shift the angle by resolution to find the left surface atoms
    angle_xy = [finalised_polar[x][1]-eig_xy_angle-resolution_xy for x in range(0,len(finalised_polar))]
    for i in range(0,len(finalised_polar)):
        #inorder to fix the anlgles fell between -180 to 180 degrees
        if angle_xy[i] < 0 and angle_xy[i] < -math.pi:
            angle_xy[i] = angle_xy[i] + 2*math.pi 
        if angle_xy[i] > 0 and angle_xy[i] > math.pi:
            angle_xy[i] = angle_xy[i] - 2*math.pi 
    
    angle_xy_z = [finalised_polar[x][1]-eig_xy_z_angle-resolution_xy_z for x in range(0,len(finalised_polar))]
    for i in range(0,len(finalised_polar)):
        #inorder to fix the anlgles fell between -90 to 90 degrees
        if angle_xy_z[i] < 0 and angle_xy_z[i] < -math.pi/2:
            angle_xy_z[i] = angle_xy_z[i] + math.pi 
        if angle_xy_z[i] > 0 and angle_xy_z[i] > math.pi/2:
            angle_xy_z[i] = angle_xy_z[i] - math.pi      
            
    way_2_sel_atoms = sel_atom_helper_way(radius,range_of_xy,range_of_xy_z)
    # then combine bothways surface atoms and return
    sel_atoms= list(set(way_1_sel_atoms + way_2_sel_atoms))   
    return sel_atoms

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

#%%then go through the polar coordinates and check the angle differences between the MOTIF group indexes it self
resolution_xy = 3
resolution_xy_z = 3

coordinates,amino_acid, c_alpha_indexes_MOTIF = retrieve_c_alpha(stack)
selected_calphas = surface_select_atom(coordinates, resolution_xy, resolution_xy_z)# thsi contain the surface selected atoms    


#selected_calphas = self.surface_select_atom(coordinates)# thsi contain the surface selected atoms    
#s_coordinates = self.shift_coordinate(coordinates)
s_coordinates = shift_coordinate(coordinates)
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


saving_dir = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2018/results/temp"
os.chdir("/")
os.chdir(saving_dir)
#pbd_name = self.pdb_name

f= open(''.join([pdb_name,'_xy.txt']),"w+")
f.write("charged(side chain can make salt bridges) , Polar(usually participate in hydrogen bonds as proton donnars & acceptors)")
f.write(" , Hydrophobic(normally burried inside the protein core) , Hydrophobic , Moderate , Hydrophillic , polar")
f.write(" Aromatic , Aliphatic , Acid , Basic , negative charge , Neutral , positive charge , Pka_NH2 , P_ka_COOH , x , y"+'\n')
for i in range(0,len(property_surface)):
    for k in property_surface[i]:
        f.write(str(k) + ",")
    f.write(str(MOTIF_prop[i]) + ",")
    f.write(str(sur_res_cor[i][0]) + "," + str(sur_res_cor[i][1]) +'\n')    
f.close()      
   
f= open(''.join([pdb_name,'_yz.txt']),"w+")
f.write("charged(side chain can make salt bridges) , Polar(usually participate in hydrogen bonds as proton donnars & acceptors)")
f.write(" , Hydrophobic(normally burried inside the protein core) , Hydrophobic , Moderate , Hydrophillic , polar")
f.write(" Aromatic , Aliphatic , Acid , Basic , negative charge , Neutral , positive charge , Pka_NH2 , P_ka_COOH , y , z"+'\n')
for i in range(0,len(property_surface)):
    for k in property_surface[i]:
        f.write(str(k) + ",")
    f.write(str(MOTIF_prop[i]) + ",")
    f.write(str(sur_res_cor[i][1]) + "," + str(sur_res_cor[i][2]) +'\n')    
f.close()      

f= open(''.join([pdb_name,'_xz.txt']),"w+")
f.write("charged(side chain can make salt bridges) , Polar(usually participate in hydrogen bonds as proton donnars & acceptors)")
f.write(" , Hydrophobic(normally burried inside the protein core) , Hydrophobic , Moderate , Hydrophillic , polar")
f.write(" Aromatic , Aliphatic , Acid , Basic , negative charge , Neutral , positive charge , Pka_NH2 , P_ka_COOH , x , z"+'\n')
for i in range(0,len(property_surface)):
    for k in property_surface[i]:
        f.write(str(k) + ",")
    f.write(str(MOTIF_prop[i]) + ",")
    f.write(str(sur_res_cor[i][0]) + "," + str(sur_res_cor[i][2]) +'\n')    
f.close()