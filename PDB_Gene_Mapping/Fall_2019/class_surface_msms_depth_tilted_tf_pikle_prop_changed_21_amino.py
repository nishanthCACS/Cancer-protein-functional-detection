# -*- coding: utf-8 -*-
"""
Created on %10-Nov-2019(05.47 P.m)
@author: %A.Nishanth C00294860
"""
import os
import pickle
import math
import numpy as np
from copy import deepcopy 

class surface_msms_depth_MOTIF_class:
    """
    USE MSMS tool(1996) to calculte the soluable access area of the surface
        depth of the C_alpha carbons, and 
              of the residue 
                          from the surface
    this vertion fix the principle direction
    """
    def __init__(self, loading_pikle_dir, saving_dir, pdb_name,height=200,negation_included_all=False,w_o_mass=False,assign_prop_round_negation_w_avg_mass_chk=False):
        self.saving_dir = saving_dir
        self.pdb_name  = pdb_name #name of the class
        self.w_o_mass = w_o_mass#this neglet the "Mass spectrometry" property
        self.negation_included_all=negation_included_all
        self.assign_prop_round_negation_w_avg_mass_chk=assign_prop_round_negation_w_avg_mass_chk
        #to select the optimal direction it should be true
        self.optimal_tilt=True
        '''loading_the_thresholds
        Earlier thresh_hold_ca=7.2,thresh_hold_res=6.7
        '''
        loading_dir_threshold = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"
         
        os.chdir("/")
        os.chdir(loading_dir_threshold)
        thresh_hold_ca= pickle.load(open("max_depths_ca_MOTIF.p", "rb"))  
        thresh_hold_res=pickle.load(open("max_depths_res_MOTIF.p", "rb"))  
        
        os.chdir('/')
        os.chdir(loading_pikle_dir)
        coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
        aminoacids = pickle.load( open( ''.join(["amino_acid_",pdb_name,".p"]), "rb" ))
        fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
        fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
        MOTIF_indexs_all = pickle.load( open( ''.join(["MOTIF_indexs_all_",pdb_name,".p"]), "rb" ))         
        
        c_alpha_indexes_MOTIF= sum(MOTIF_indexs_all, [])
        res_factor = 2.25 # see the documentation twhy 2.25 is chosen

        sur_res = []
        sur_res_cor_intial = []
        MOTIF_prop =[]
        
        #% to find out the surface atoms residues 
        for i in range(0,len(fin_res_depth_all)):
            if fin_ca_depth_all[i] <= thresh_hold_ca:
                if fin_res_depth_all[i] <= thresh_hold_res:
                    sur_res.append(aminoacids[i])
                    # multiply each coordinate by 2 (just for increasing the resolution) and then round them to decimal numbers.
                    #sur_res_cor_intial_round.append([round(res_factor*coordinates[i][0]),round(res_factor*coordinates[i][1]),round(res_factor*coordinates[i][2])])
                    sur_res_cor_intial.append([res_factor*coordinates[i][0],res_factor*coordinates[i][1],res_factor*coordinates[i][2]])

                    if i in c_alpha_indexes_MOTIF:
                        MOTIF_prop.append(1)
                    else:
                        MOTIF_prop.append(0)
                        
        if len(sur_res_cor_intial)==0:       
            print(pdb_name, ' not has single atom to satify')
        else:
#            '''first title the coordinates in principle direction; and finally round the o/p coordinates'''
#            sur_res_cor_tilted = self.change_C_alpha_principle_direc(sur_res_cor_intial)
            '''rotate in optimal direction'''
            sur_res_cor_tilted = self.rotate_C_alpha_optimal_direc(sur_res_cor_intial)
            #then shift the coordinates
            sur_res_cor=self.shift_coordinate(sur_res_cor_tilted)
        
            w_o_mass=False
            w_avg_mass=False
            #% then assign the property to the corresponding aminoacids  
            if self.w_o_mass and self.negation_included_all:
                property_surface, amino_acid_ok,w_o_mass= self.assign_prop_round_negation_w_o_mass(sur_res)#to check all amino acids in 20 of them
#            self.property_surface = property_surface
#            self.MOTIF_prop = MOTIF_prop
#            self.sur_res_cor = sur_res_cor
            elif self.assign_prop_round_negation_w_avg_mass_chk and self.negation_included_all:
                property_surface, amino_acid_ok,w_avg_mass= self.assign_prop_round_negation_w_avg_mass(sur_res)#to check all amino acids in 20 of them
            elif self.assign_prop_round_negation_w_avg_mass_chk and not self.negation_included_all:
                property_surface, amino_acid_ok,w_avg_mass= self.assign_prop_round_neg_yes_only_w_avg_mass(sur_res)#to check all amino acids in 20 of them
            else:
                property_surface,amino_acid_ok = self.assign_prop_round(sur_res)
            '''to save in the different from earlier'''
            if not os.path.isdir(self.saving_dir):
                os.makedirs(self.saving_dir)
                print("Making saving direcory: ",self.saving_dir )
            os.chdir("/")
            os.chdir(self.saving_dir)
            pdb_name = self.pdb_name
            if amino_acid_ok:
                MOTIF_prop=np.array(MOTIF_prop,dtype=float)
                property_surface=np.array(property_surface)
                whole_property_surface=np.column_stack((property_surface,MOTIF_prop))
                #assigning the space for the matrix that hold the property
                if w_o_mass and negation_included_all:
                    coordinate_property=np.zeros((3,height,height,19))
                elif w_avg_mass and negation_included_all:
                    coordinate_property=np.zeros((3,height,height,20))
                elif w_avg_mass and not negation_included_all:                    
                    coordinate_property=np.zeros((3,height,height,15))
                else:
                    coordinate_property=np.zeros((3,height,height,21))
                sur_res_cor=np.array(sur_res_cor,dtype=int)
                save_numpy=True

                for i in range(0,len(sur_res_cor)):
                    if sur_res_cor[i,0]>(height-1) or sur_res_cor[i,1]>(height-1) or sur_res_cor[i,2]>(height-1):
                        print(' ')
                        print(pdb_name,' size issue: ')
                        print(' x size: ',np.max(sur_res_cor[:,0]))
                        print(' y size: ',np.max(sur_res_cor[:,1]))
                        print(' z size: ',np.max(sur_res_cor[:,2]))
                        print(' ')
                        save_numpy=False
                        break
                    else:
                        coordinate_property[0,sur_res_cor[i,0],sur_res_cor[i,1],:]=deepcopy(whole_property_surface[i,:])
                        coordinate_property[1,sur_res_cor[i,1],sur_res_cor[i,2],:]=deepcopy(whole_property_surface[i,:])
                        coordinate_property[2,sur_res_cor[i,0],sur_res_cor[i,2],:]=deepcopy(whole_property_surface[i,:])
                if save_numpy and amino_acid_ok: 
                    np.save(pdb_name, coordinate_property)
                else:
                    print('PDB ',pdb_name,' skipped due to size issue')
            else:
                print('PDB ',pdb_name,' skipped due to amino acid issue')


#    def results(self):
#        '''For creation and checking purpose'''
#        property_surface = self.property_surface
#        MOTIF_prop = self.MOTIF_prop
#        sur_res_cor = self.sur_res_cor
#        return property_surface,MOTIF_prop,sur_res_cor
    
    def shift_coordinate(self,coordinates):
        """ 
        find the minimum position to move this shift the coordinates to make NN-train
        """
        s_coordinates = deepcopy(coordinates)
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

    """ using eigen vector to rotate in optimal direction"""
    def rotate_C_alpha_optimal_direc(self,coordinates):
        """
        then change the C-alpha coordinates to the optimal direction
        Inorder to do that calculate the eigen vector and 
        use the eigen vector to rotate the coordinates
        
        returns 
        coordinates             : cetralised coordinates
        finalised_cartesian     : Get the optimal direction rotated coordinates
        """
        cen_coordinates = deepcopy(coordinates)
        
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
        
        cen_coordinates = np.array(cen_coordinates)
        if self.optimal_tilt:
            #calculate the eigen values and vigen vectors
            cen_coordinates_cov=np.cov(cen_coordinates.transpose())
            #eigenvalues = np.linalg.eigvals(cen_coordinates_cov)
            #eigenvecs = np.linalg.eig(cen_coordinates_cov)
            w, v = np.linalg.eig(cen_coordinates_cov)
            finalised_cart =np.matmul(cen_coordinates,v)
        else:
            finalised_cart=deepcopy(cen_coordinates)
            
        rounded_cart=[]
        for i in range(0,len(coordinates)):
            rounded_cart.append([round(finalised_cart[i][0]),round(finalised_cart[i][1]),round(finalised_cart[i][2])])
        return rounded_cart

    """To tilt the coordinate in princilple direction"""
    def cart2pol(self,x, y):
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
    
    def pol2car(self,r,ang):
        """
        This fucntion change the polar coordinates to cartesian
        """
        x=r*math.cos(ang)
        y=r*math.sin(ang)
        return x,y
    
    def change_C_alpha_principle_direc(self, coordinates):
        """
        then change the C-alpha coordinates to the principle direction
        Inorder to do that calculate the eigen vector and calculate the angle of that
        change the cartesian coordinates -> polar coordinates-eign_vector angle
        cange back to cartesian
        
        returns 
        coordinates             : cetralised coordinates
        finalised_polar         : polar_coordinates from the cetralised coordinates
        finalised_cartesian     : Get the priciple direction shifted cartesian coordinate 
        """
        cen_coordinates = deepcopy(coordinates)
        
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
        
        cen_coordinates = np.array(cen_coordinates)
        #calculate the eigen values and vigen vectors
        cen_coordinates_cov=np.cov(cen_coordinates.transpose())
        #eigenvalues = np.linalg.eigvals(cen_coordinates_cov)
        #eigenvecs = np.linalg.eig(cen_coordinates_cov)
        w, v = np.linalg.eig(cen_coordinates_cov)
        maxi_direction = v[:,w.argmax()]
        
        eig_xy_r, eig_xy_angle = self.cart2pol(maxi_direction[0], maxi_direction[1])   
        eig_xyz_r, eig_xy_z_angle = self.cart2pol(eig_xy_r, maxi_direction[2])  
        
        #% first create an empty matrices to hold the data
        xy_polar = []#here first coloumn contain the radius and the second coloumn contain angle
        xy_back=[]
        for i in range(0,len(coordinates)):
            # find the radius of x-y coordinate
            r,ang = self.cart2pol(cen_coordinates[i][0] ,cen_coordinates[i][1])
            xy_polar.append([r,ang-eig_xy_angle])
            xy_back.append(self.pol2car(r,ang-eig_xy_angle))
        #then xy to z coordinate angle
        xy_z_polar = []#here first coloumn contain the radius and the second coloumn contain angle
        z_back=[]
        for i in range(0,len(coordinates)):
            # find the radius of xy-to z coordinate
            #take the xy radius
            r,ang = self.cart2pol(xy_polar[i][0],cen_coordinates[i][2])
            xy_z_polar.append([r,ang -eig_xy_z_angle])
            x_y,z = self.pol2car(r,ang-eig_xy_z_angle)
            z_back.append(z)
            
        finalised_polar = []
        finalised_cart=[]
        rounded_cart=[]
        for i in range(0,len(coordinates)):
            finalised_polar.append([xy_z_polar[i][0], xy_polar[i][1], xy_z_polar[i][1]])
            finalised_cart.append([xy_back[i][0],xy_back[i][1],z_back[i]])
            rounded_cart.append([round(xy_back[i][0]),round(xy_back[i][1]),round(z_back[i])])
        #return np.array(finalised_polar),np.array(finalised_cart)
        return rounded_cart
    
    def assign_prop_round(self,sur_res,rdi=5):
        '''
        This function has binary property o/ps as positive or 0 only and newly indluded properties
        
        This function intake
        sur_res : as surfcae residues as letters
        rdi     : rounding digits of the property
        
        Returns
        property_surface: Assigned surface properties accordingl
        amino_acid_ok   : Return True if all the residues are in the list and satisfied
                          Return False if atleast one residue doen't identified  
        '''
        #% then assign the property to the corresponding aminoacids  
        property_surface = []
        amino_acid_ok=True#to check all amino acids in 20 of them      
        for i in range(0,len(sur_res)):               
            if sur_res[i]=="M":
                property_surface.append([round(0.73059286,rdi),round(0.365360303,rdi),round(0.5,rdi),round(0.282828283,rdi),round(1,rdi),round(0,rdi),round(0,rdi),round(0,rdi),round(1,rdi),round(0,rdi),round(0,rdi),round(0,rdi),round(1,rdi),round(0,rdi),round(1,rdi),round(1,rdi),round(0,rdi),round(0,rdi),round(0.704218486,rdi),round(0.704560284,rdi)])
            elif sur_res[i]=="R":
                property_surface.append([round(0.85297983,rdi),1,round(0.03030303,rdi),round(0.136363636,rdi),0,1,1,1,0,0,1,0,1,0,0,0,0,1,round(0.838895576,rdi),round(0.838761527,rdi)])
            elif sur_res[i]=="K":
                property_surface.append([round(0.715812842,rdi),round(0.85335019,rdi),round(0.545454545,rdi),round(0.171717172,rdi),0,1,round(0.845528455,rdi),1,0,0,round(0.6,rdi),0,1,0,0,1,0,0,round(0.688389053,rdi),round(0.68832162,rdi)])
            elif sur_res[i]=="D":
                property_surface.append([round(0.651739983,rdi),0,round(0.287878788,rdi),round(0.595959596,rdi),0,1,round(0.298373984,rdi),1,0,1,0,1,0,0,0,0,1,0,round(0.618160826,rdi),round(0.618051994,rdi)])
            elif sur_res[i]=="E":
                property_surface.append([round(0.720422386,rdi),round(0.037926675,rdi),round(0.454545455,rdi),round(0.378787879,rdi),0,1,round(0.345528455,rdi),1,0,1,0,0,1,0,0,0,0,1,round(0.693481667,rdi),round(0.693378816,rdi)])
            elif sur_res[i]=="Q":
                property_surface.append([round(0.715600335,rdi),round(0.353982301,rdi),round(0.560606061,rdi),round(0.207070707,rdi),0,1,0,1,0,0,0,0,1,0,0,0,1,0,round(0.688193545,rdi),round(0.688090161,rdi)])
            elif sur_res[i]=="N":
                property_surface.append([round(0.646917932,rdi),round(0.323640961,rdi),round(0.515151515,rdi),0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,round(0.612872705,rdi),round(0.612763339,rdi)])
            elif sur_res[i]=="H":
                property_surface.append([round(0.759719557,rdi),round(0.600505689,rdi),0,round(0.308080808,rdi),0,1,round(0.531707317,rdi),1,0,0,round(0.3,rdi),0,1,1,0,1,0,0,round(0.736561792,rdi),round(0.736476954,rdi)])
            elif sur_res[i]=="S":
                property_surface.append([round(0.514587684,rdi),round(0.357774968,rdi),round(0.590909091,rdi),round(0.247474747,rdi),0,1,0,1,0,0,0,1,0,0,0,0,1,0,round(0.467714707,rdi),round(0.467629809,rdi)])
            elif sur_res[i]=="T":
                property_surface.append([round(0.583270087,rdi),round(0.347661188,rdi),round(0.439393939,rdi),round(0.191919192,rdi),0,1,0,1,0,0,0,1,0,0,0,1,0,0,round(0.543035548,rdi),round(0.542956631,rdi)])
            elif sur_res[i]=="Y":
                property_surface.append([round(0.887198864,rdi),round(0.352718078,rdi),round(0.606060606,rdi),round(0.247474747,rdi),0,1,round(0.8,rdi),1,0,round(0.3,rdi),0,0,1,1,0,0,0,1,round(0.876310913,rdi),round(0.876286921,rdi)])
            elif sur_res[i]=="C":
                property_surface.append([round(0.593228054,rdi),round(0.278128951,rdi),round(0.181818182,rdi),1,1,0,round(0.695121951,rdi),0,1,1,0,1,0,0,0,0,0,1,round(0.553576806,rdi),round(0.55390664,rdi)])
            elif sur_res[i]=="W":
                property_surface.append([1,round(0.384323641,rdi),1,round(0.348484848,rdi),1,0,0,0,1,0,0,0,1,1,0,1,0,0,1,1])
            elif sur_res[i]=="A":
                property_surface.append([round(0.436246979,rdi),round(0.399494311,rdi),round(0.833333333,rdi),round(0.580808081,rdi),1,0,0,0,1,0,0,1,0,0,1,0,1,0,round(0.381757166,rdi),round(0.381708491,rdi)])
            elif sur_res[i]=="I":
                property_surface.append([round(0.642293698,rdi),round(0.404551201,rdi),round(0.787878788,rdi),round(0.525252525,rdi),1,0,0,0,1,0,0,0,1,0,1,1,0,0,round(0.607719687,rdi),round(0.60768842,rdi)])
            elif sur_res[i]=="L":
                property_surface.append([round(0.642293698,rdi),round(0.399494311,rdi),round(0.803030303,rdi),round(0.515151515,rdi),1,0,0,0,1,0,0,0,1,0,1,1,0,0,round(0.607719687,rdi),round(0.60768842,rdi)])
            elif sur_res[i]=="F":
                property_surface.append([round(0.808858159,rdi),round(0.333754741,rdi),round(0.606060606,rdi),round(0.297979798,rdi),1,0,0,0,1,0,0,0,1,1,0,1,0,0,round(0.790353371,rdi),round(0.790365604,rdi)])
            elif sur_res[i]=="V":
                property_surface.append([round(0.573611785,rdi),round(0.398230088,rdi),round(0.893939394,rdi),round(0.515151515,rdi),1,0,0,0,1,0,0,1,0,0,1,1,0,0,round(0.532398846,rdi),round(0.532362135,rdi)])
            elif sur_res[i]=="P":
                property_surface.append([round(0.563740976,rdi),round(0.436156764,rdi),round(0.227272727,rdi),round(0.96969697,rdi),1,0,0,0,1,0,0,1,0,0,0,0,1,0,round(0.521566637,rdi),round(0.52153618,rdi)])
            elif sur_res[i]=="G":
                property_surface.append([round(0.367564576,rdi),round(0.405815424,rdi),round(0.833333333,rdi),round(0.535353535,rdi),1,0,0,0,1,0,0,1,0,0,0,0,0,1,round(0.306436325,rdi),round(0.306381669,rdi)])
            elif sur_res[i]=="U":
                property_surface.append([round(0.822867765,rdi),round(0.331226296,rdi),round(0.166666667,rdi),round(0.646464646,rdi),0,1,round(0.441463415,rdi),0,1,1,0,1,0,0,0,0,1,0,round(0.811232802,rdi),round(0.805805169,rdi)])
            else:
#                    print(i,' sur_res ',sur_res[i])
                amino_acid_ok=False    
                
        return property_surface,amino_acid_ok
    
    def assign_prop_round_negation_w_o_mass(self,sur_res,rdi=5):
        '''
        This function has  property opposite has negation inlcuded properties
            "Normalised_negative_prop_inc" : and not including the property of Mass spectrometry
            
        This function intake
        sur_res : as surface residues as letters
        rdi     : rounding digits of the property
        
        Returns
        property_surface: Assigned surface properties accordingl
        amino_acid_ok   : Return True if all the residues are in the list and satisfied
                          Return False if atleast one residue doen't identified  
        '''
        #% then assign the property to the corresponding aminoacids  
        property_surface = []
        amino_acid_ok=True#to check all amino acids in 20 of them    
        for i in range(0,len(sur_res)):   
            if sur_res[i]=="A":
                property_surface.append([round(0.436246979,rdi), round(0.399494311,rdi), round(0.833333333,rdi), round(0.580808081,rdi), 1, -1, 0, -1, 1, 0, 0, 1, -1, 0, 1, -1, 1, 0])
            elif  sur_res[i]=="C":
                property_surface.append([round(0.593228054,rdi), round(0.278128951,rdi), round(0.181818182,rdi), 1, 1, -1, round(0.695121951,rdi), -1, 1, 1, -1, 1, -1, 0, 0, 0, 0, 1])
            elif  sur_res[i]=="D":
                property_surface.append([round(0.651739983,rdi), 0, round(0.287878788,rdi), round(0.595959596,rdi), -1, 1, round(0.298373984,rdi), 1, -1, 1, -1, 1, -1, 0, 0, -1, 1, 0])
            elif  sur_res[i]=="E":
                property_surface.append([round(0.720422386,rdi), round(0.037926675,rdi), round(0.454545455,rdi), round(0.378787879,rdi), -1, 1, round(0.345528455,rdi), 1, -1, 1, -1, -1, 1, 0, 0, 0, 0, 1])
            elif  sur_res[i]=="F":
                property_surface.append([round(0.808858159,rdi), round(0.333754741,rdi), round(0.606060606,rdi), round(0.297979798,rdi), 1, -1, 0, -1, 1, 0, 0, -1, 1, 1, 0, 1, -1, 0])
            elif  sur_res[i]=="G":
                property_surface.append([round(0.367564576,rdi), round(0.405815424,rdi), round(0.833333333,rdi), round(0.535353535,rdi), 1, -1, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 1])
            elif  sur_res[i]=="H":
                property_surface.append([round(0.759719557,rdi), round(0.600505689,rdi), 0, round(0.308080808,rdi), -1, 1, round(0.531707317,rdi), 1, -1, round(-0.3,rdi), round(0.3,rdi), -1, 1, 1, 0, 1, -1, 0])
            elif  sur_res[i]=="I":
                property_surface.append([round(0.642293698,rdi), round(0.404551201,rdi), round(0.787878788,rdi), round(0.525252525,rdi), 1, -1, 0, -1, 1, 0, 0, -1, 1, 0, 1, 1, -1, 0])
            elif  sur_res[i]=="K":
                property_surface.append([round(0.715812842,rdi), round(0.85335019,rdi), round(0.545454545,rdi), round(0.171717172,rdi), -1, 1, round(0.845528455,rdi), 1, -1, round(-0.6,rdi), round(0.6,rdi), -1, 1, 0, 0, 1, -1, 0])
            elif  sur_res[i]=="L":
                property_surface.append([round(0.642293698,rdi), round(0.399494311,rdi), round(0.803030303,rdi), round(0.515151515,rdi), 1, -1, 0, -1, 1, 0, 0, -1, 1, 0, 1, 1, -1, 0])
            elif  sur_res[i]=="M":
                property_surface.append([round(0.73059286,rdi), round(0.365360303,rdi), round(0.5,rdi), round(0.282828283,rdi), 1, -1, 0, -1, 1, 0, 0, -1, 1, 0, 1, 1, -1, 0])
            elif  sur_res[i]=="N":
                property_surface.append([round(0.646917932,rdi), round(0.323640961,rdi), round(0.515151515,rdi), 0, -1, 1, 0, 1, -1, 0, 0, 1, -1, 0, 0, -1, 1, 0])
            elif  sur_res[i]=="P":
                property_surface.append([round(0.563740976,rdi), round(0.436156764,rdi), round(0.227272727,rdi), round(0.96969697,rdi), 1, -1, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, 1, 0])
            elif  sur_res[i]=="Q":
                property_surface.append([round(0.715600335,rdi), round(0.353982301,rdi), round(0.560606061,rdi), round(0.207070707,rdi), -1, 1, 0, 1, -1, 0, 0, -1, 1, 0, 0, -1, 1, 0])
            elif  sur_res[i]=="R":
                property_surface.append([round(0.85297983,rdi), 1, round(0.03030303,rdi), round(0.136363636,rdi), -1, 1, 1, 1, -1, -1, 1, -1, 1, 0, 0, 0, 0, 1])
            elif  sur_res[i]=="S":
                property_surface.append([round(0.514587684,rdi), round(0.357774968,rdi), round(0.590909091,rdi), round(0.247474747,rdi), -1, 1, 0, 1, -1, 0, 0, 1, -1, 0, 0, -1, 1, 0])
            elif  sur_res[i]=="T":
                property_surface.append([round(0.583270087,rdi), round(0.347661188,rdi), round(0.439393939,rdi), round(0.191919192,rdi), -1, 1, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0])
            elif  sur_res[i]=="U":
                property_surface.append([round(0.822867765,rdi), round(0.331226296,rdi), round(0.166666667,rdi), round(0.646464646,rdi), -1, 1, round(0.441463415,rdi), -1, 1, 1, -1, 1, -1, 0, 0, -1, 1, 0])
            elif  sur_res[i]=="V":
                property_surface.append([round(0.573611785,rdi), round(0.398230088,rdi), round(0.893939394,rdi), round(0.515151515,rdi), 1, -1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 1, -1, 0])
            elif  sur_res[i]=="W":
                property_surface.append([1, round(0.384323641,rdi), 1, round(0.348484848,rdi), 1, -1, 0, -1, 1, 0, 0, -1, 1, 1, 0, 1, -1, 0])
            elif  sur_res[i]=="Y":
                property_surface.append([round(0.887198864,rdi), round(0.352718078,rdi), round(0.606060606,rdi), round(0.247474747,rdi), -1, 1, round(0.8,rdi), 1, -1, round(0.3,rdi), round(-0.3,rdi), -1, 1, 1, 0, 0, 0, 1])
            else:
                amino_acid_ok=False
            
        return property_surface,amino_acid_ok,True 
    
    def assign_prop_round_negation_w_avg_mass(self,sur_res,rdi=5):
        '''
        This function has  property opposite has negation inlcuded properties
            "Normalised_negative_prop_inc" : along with it and only including the property of Mass spectrometry
            and excluding the Mon. mass§ (Da)
        
        This function intake
        sur_res : as surface residues as letters
        rdi     : rounding digits of the property
        
        Returns
        property_surface: Assigned surface properties accordingl
        amino_acid_ok   : Return True if all the residues are in the list and satisfied
                          Return False if atleast one residue doen't identified  
        '''
        #% then assign the property to the corresponding aminoacids  
        property_surface = []
        amino_acid_ok=True#to check all amino acids in 20 of them    
        for i in range(0,len(sur_res)): 
            if sur_res[i]=="A":
                        property_surface.append([round(0.436246979,rdi), round(0.399494311,rdi), round(0.833333333,rdi), round(0.580808081,rdi), 1, -1, 0, -1, 1, 0, 0, 1, -1, 0, 1, -1, 1, 0, round(0.381708491,rdi)])
            elif  sur_res[i]=="C":
                        property_surface.append([round(0.593228054,rdi), round(0.278128951,rdi), round(0.181818182,rdi), 1, 1, -1, round(0.695121951,rdi), -1, 1, 1, -1, 1, -1, 0, 0, 0, 0, 1, round(0.55390664,rdi)])
            elif  sur_res[i]=="D":
                        property_surface.append([round(0.651739983,rdi), 0, round(0.287878788,rdi), round(0.595959596,rdi), -1, 1, round(0.298373984,rdi), 1, -1, 1, -1, 1, -1, 0, 0, -1, 1, 0, round(0.618051994,rdi)])
            elif  sur_res[i]=="E":
                        property_surface.append([round(0.720422386,rdi), round(0.037926675,rdi), round(0.454545455,rdi), round(0.378787879,rdi), -1, 1, round(0.345528455,rdi), 1, -1, 1, -1, -1, 1, 0, 0, 0, 0, 1, round(0.693378816,rdi)])
            elif  sur_res[i]=="F":
                        property_surface.append([round(0.808858159,rdi), round(0.333754741,rdi), round(0.606060606,rdi), round(0.297979798,rdi), 1, -1, 0, -1, 1, 0, 0, -1, 1, 1, 0, 1, -1, 0,round(0.790365604,rdi)])
            elif  sur_res[i]=="G":
                        property_surface.append([round(0.367564576,rdi), round(0.405815424,rdi), round(0.833333333,rdi), round(0.535353535,rdi), 1, -1, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 1,round(0.306381669,rdi)])
            elif  sur_res[i]=="H":
                        property_surface.append([round(0.759719557,rdi), round(0.600505689,rdi), 0, round(0.308080808,rdi), -1, 1, round(0.531707317,rdi), 1, -1, round(-0.3,rdi), round(0.3,rdi), -1, 1, 1, 0, 1, -1, 0,round(0.736476954,rdi)])
            elif  sur_res[i]=="I":
                        property_surface.append([round(0.642293698,rdi), round(0.404551201,rdi), round(0.787878788,rdi), round(0.525252525,rdi), 1, -1, 0, -1, 1, 0, 0, -1, 1, 0, 1, 1, -1, 0,round(0.60768842,rdi)])
            elif  sur_res[i]=="K":
                        property_surface.append([round(0.715812842,rdi), round(0.85335019,rdi), round(0.545454545,rdi), round(0.171717172,rdi), -1, 1, round(0.845528455,rdi), 1, -1, round(-0.6,rdi), round(0.6,rdi), -1, 1, 0, 0, 1, -1, 0,round(0.68832162,rdi)])
            elif  sur_res[i]=="L":
                        property_surface.append([round(0.642293698,rdi), round(0.399494311,rdi), round(0.803030303,rdi), round(0.515151515,rdi), 1, -1, 0, -1, 1, 0, 0, -1, 1, 0, 1, 1, -1, 0,round(0.60768842,rdi)])
            elif  sur_res[i]=="M":
                        property_surface.append([round(0.73059286,rdi), round(0.365360303,rdi), round(0.5,rdi), round(0.282828283,rdi), 1, -1, 0, -1, 1, 0, 0, -1, 1, 0, 1, 1, -1, 0,round(0.704560284,rdi)])
            elif  sur_res[i]=="N":
                        property_surface.append([round(0.646917932,rdi), round(0.323640961,rdi), round(0.515151515,rdi), 0, -1, 1, 0, 1, -1, 0, 0, 1, -1, 0, 0, -1, 1, 0,round(0.612763339,rdi)])
            elif  sur_res[i]=="P":
                        property_surface.append([round(0.563740976,rdi), round(0.436156764,rdi), round(0.227272727,rdi), round(0.96969697,rdi), 1, -1, 0, -1, 1, 0, 0, 1, -1, 0, 0, -1, 1, 0,round(0.52153618,rdi)])
            elif  sur_res[i]=="Q":
                        property_surface.append([round(0.715600335,rdi), round(0.353982301,rdi), round(0.560606061,rdi), round(0.207070707,rdi), -1, 1, 0, 1, -1, 0, 0, -1, 1, 0, 0, -1, 1, 0,round(0.688090161,rdi)])
            elif  sur_res[i]=="R":
                        property_surface.append([round(0.85297983,rdi), 1, round(0.03030303,rdi), round(0.136363636,rdi), -1, 1, 1, 1, -1, -1, 1, -1, 1, 0, 0, 0, 0, 1,round(0.838761527,rdi)])
            elif  sur_res[i]=="S":
                        property_surface.append([round(0.514587684,rdi), round(0.357774968,rdi), round(0.590909091,rdi), round(0.247474747,rdi), -1, 1, 0, 1, -1, 0, 0, 1, -1, 0, 0, -1, 1, 0,round(0.467629809,rdi)])
            elif  sur_res[i]=="T":
                        property_surface.append([round(0.583270087,rdi), round(0.347661188,rdi), round(0.439393939,rdi), round(0.191919192,rdi), -1, 1, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0,round(0.542956631,rdi)])
            elif  sur_res[i]=="U":
                        property_surface.append([round(0.822867765,rdi), round(0.331226296,rdi), round(0.166666667,rdi), round(0.646464646,rdi), -1, 1, round(0.441463415,rdi), -1, 1, 1, -1, 1, -1, 0, 0, -1, 1, 0,round(0.805805169,rdi)])
            elif  sur_res[i]=="V":
                        property_surface.append([round(0.573611785,rdi), round(0.398230088,rdi), round(0.893939394,rdi), round(0.515151515,rdi), 1, -1, 0, -1, 1, 0, 0, 1, -1, 0, 1, 1, -1, 0,round(0.532362135,rdi)])
            elif  sur_res[i]=="W":
                        property_surface.append([1, round(0.384323641,rdi), 1, round(0.348484848,rdi), 1, -1, 0, -1, 1, 0, 0, -1, 1, 1, 0, 1, -1, 0,1])
            elif  sur_res[i]=="Y":
                        property_surface.append([round(0.887198864,rdi), round(0.352718078,rdi), round(0.606060606,rdi), round(0.247474747,rdi), -1, 1, round(0.8,rdi), 1, -1, round(0.3,rdi), round(-0.3,rdi), -1, 1, 1, 0, 0, 0, 1,round(0.876286921,rdi)])
            else:
                  amino_acid_ok=False
            
        return property_surface,amino_acid_ok,True 
    
    
    def assign_prop_round_neg_yes_only_w_avg_mass(self,sur_res,rdi=5):
        '''
        This function has  property opposite has negation inlcuded properties
        Mean only the Yes is factored with +1 and No is changed or factored by (-1)
        
            "Normalised_negative_prop_inc" : along with it and only including the property of Mass spectrometry
            and excluding the Mon. mass§ (Da)
        
        This function intake
        sur_res : as surface residues as letters
        rdi     : rounding digits of the property
        
        Returns
        property_surface: Assigned surface properties accordingl
        amino_acid_ok   : Return True if all the residues are in the list and satisfied
                          Return False if atleast one residue doen't identified  
        '''
        #% then assign the property to the corresponding aminoacids  
        property_surface = []
        amino_acid_ok=True#to check all amino acids in 20 of them    
        for i in range(0,len(sur_res)):     
            if  sur_res[i]=="A":
                property_surface.append([round(0.436246979,rdi), round(0.399494311,rdi), round(0.833333333,rdi), round(0.580808081,rdi), 1, 0, -1, 0, 1, 0, 1, -1, 0, round(0.381489437,rdi)])
            elif  sur_res[i]=="C":
                property_surface.append([round(0.593228054,rdi), round(0.278128951,rdi), round(0.181818182,rdi), 1, 1, round(0.695121951,rdi), -1, 1, 1, 0, 0, 0, 1, round(0.553188579,rdi)])
            elif  sur_res[i]=="D":
                property_surface.append([round(0.651739983,rdi), 0, round(0.287878788,rdi), round(0.595959596,rdi), -1, round(0.298373984,rdi), 1, 1, 1, 0, 0, -1, 0, round(0.617727307,rdi)])
            elif  sur_res[i]=="E":
                property_surface.append([round(0.720422386,rdi), round(0.037926675,rdi), round(0.454545455,rdi), round(0.378787879,rdi), -1, round(0.345528455,rdi), 1, 1, -1, 0, 0, 0, 1, round(0.692995324,rdi)])
            elif  sur_res[i]=="F":
                property_surface.append([round(0.808858159,rdi), round(0.333754741,rdi), round(0.606060606,rdi), round(0.297979798,rdi), 1, 0, -1, 0, -1, 1, 0, 1, 0, round(0.789799092,rdi)])
            elif  sur_res[i]=="G":
                property_surface.append([round(0.367564576,rdi), round(0.405815424,rdi), round(0.833333333,rdi), round(0.535353535,rdi), 1, 0, -1, 0, 1, 0, 0, 0, 1, round(0.30622142,rdi)])
            elif  sur_res[i]=="H":
                property_surface.append([round(0.759719557,rdi), round(0.600505689,rdi), 0, round(0.308080808,rdi), -1, round(0.531707317,rdi), 1, round(-0.3,rdi), -1, 1, 0, 1, 0, round(0.736045237,rdi)])
            elif  sur_res[i]=="I":
                property_surface.append([round(0.642293698,rdi), round(0.404551201,rdi), round(0.787878788,rdi), round(0.525252525,rdi), 1, 0, -1, 0, -1, 0, 1, 1, 0, round(0.60729349,rdi)])
            elif  sur_res[i]=="K":
                property_surface.append([round(0.715812842,rdi), round(0.85335019,rdi), round(0.545454545,rdi), round(0.171717172,rdi), -1, round(0.845528455,rdi), 1, round(-0.6,rdi), -1, 0, 0, 1, 0, round(0.687906282,rdi)])
            elif  sur_res[i]=="L":
                property_surface.append([round(0.642293698,rdi), round(0.399494311,rdi), round(0.803030303,rdi), round(0.515151515,rdi), 1, 0, -1, 0, -1, 0, 1, 1, 0, round(0.60729349,rdi)])
            elif  sur_res[i]=="M":
                property_surface.append([round(0.73059286,rdi), round(0.365360303,rdi), round(0.5,rdi), round(0.282828283,rdi), 1, 0, -1, 0, -1, 0, 1, 1, 0, round(0.703724614,rdi)])
            elif  sur_res[i]=="N":
                property_surface.append([round(0.646917932,rdi), round(0.323640961,rdi), round(0.515151515,rdi), 0, -1, 0, 1, 0, 1, 0, 0, -1, 0, round(0.612442894,rdi)])
            elif  sur_res[i]=="P":
                property_surface.append([round(0.563740976,rdi), round(0.436156764,rdi), round(0.227272727,rdi), round(0.96969697,rdi), 1, 0, -1, 0, 1, 0, 0, -1, 0, round(0.52120086,rdi)])
            elif  sur_res[i]=="Q":
                property_surface.append([round(0.715600335,rdi), round(0.353982301,rdi), round(0.560606061,rdi), round(0.207070707,rdi), -1, 0, 1, 0, -1, 0, 0, -1, 0, round(0.687710911,rdi)])
            elif  sur_res[i]=="R":
                property_surface.append([round(0.85297983,rdi), 1, round(0.03030303,rdi), round(0.136363636,rdi), -1, 1, 1, -1, -1, 0, 0, 0, 1, round(0.838307254,rdi)])
            elif  sur_res[i]=="S":
                property_surface.append([round(0.514587684,rdi), round(0.357774968,rdi), round(0.590909091,rdi), round(0.247474747,rdi), -1, 0, 1, 0, 1, 0, 0, -1, 0, round(0.467386696,rdi)])
            elif  sur_res[i]=="T":
                property_surface.append([round(0.583270087,rdi), round(0.347661188,rdi), round(0.439393939,rdi), round(0.191919192,rdi), -1, 0, 1, 0, 1, 0, 0, 1, 0, round(0.542654714,rdi)])
            elif  sur_res[i]=="U":
                property_surface.append([round(0.822867765,rdi), round(0.331226296,rdi), round(0.166666667,rdi), round(0.646464646,rdi), -1, round(0.441463415,rdi), -1, 1, 1, 0, 0, -1, 0, round(0.81066388,rdi)])
            elif  sur_res[i]=="V":
                property_surface.append([round(0.573611785,rdi), round(0.398230088,rdi), round(0.893939394,rdi), round(0.515151515,rdi), 1, 0, -1, 0, 1, 0, 1, 1, 0, round(0.532025472,rdi)])
            elif  sur_res[i]=="W":
                property_surface.append([1, round(0.384323641,rdi), 1, round(0.348484848,rdi), 1, 0, -1, 0, -1, 1, 0, 1, 0, round(0.999298695,rdi)])
            elif  sur_res[i]=="Y":
                property_surface.append([round(0.887198864,rdi), round(0.352718078,rdi), round(0.606060606,rdi), round(0.247474747,rdi), -1, round(0.8,rdi), 1, round(0.3,rdi), -1, 1, 0, 0, 1, round(0.875696351,rdi)])
            else:
                  amino_acid_ok=False
            
        return property_surface,amino_acid_ok,True 

class surface_msms_depth_class(surface_msms_depth_MOTIF_class):
    """
    USE MSMS tool(1996) to calculte the soluable access area of the surface
        depth of the C_alpha carbons, and 
              of the residue 
                          from the surface
    this vertion fix the principle direction
    """
    def __init__(self, loading_pikle_dir, saving_dir, pdb_name,height=200,negation_included_all=False,w_o_mass=False,assign_prop_round_negation_w_avg_mass_chk=False):
        self.saving_dir = saving_dir
        self.pdb_name  = pdb_name #name of the class
        self.w_o_mass = w_o_mass#this neglet the "Mass spectrometry" property
        self.negation_included_all=negation_included_all
        self.assign_prop_round_negation_w_avg_mass_chk=assign_prop_round_negation_w_avg_mass_chk
        '''loading_the_thresholds
        Earlier thresh_hold_ca=7.2,thresh_hold_res=6.7
        '''
        loading_dir_threshold = "C:/Users/nishy/Documents/Projects_UL/Mot_if_study_python/Fall_2019/SITE_Author"
         
        os.chdir("/")
        os.chdir(loading_dir_threshold)
        thresh_hold_ca= pickle.load(open("max_depths_ca_MOTIF.p", "rb"))  
        thresh_hold_res=pickle.load(open("max_depths_res_MOTIF.p", "rb"))  
        
        os.chdir('/')
        os.chdir(loading_pikle_dir)
        coordinates = pickle.load( open( ''.join(["coordinates_",pdb_name,".p"]), "rb" ))
        aminoacids = pickle.load( open( ''.join(["amino_acid_",pdb_name,".p"]), "rb" ))
        fin_res_depth_all = pickle.load( open( ''.join(["fin_res_depth_all_",pdb_name,".p"]), "rb" ))
        fin_ca_depth_all = pickle.load( open( ''.join(["fin_ca_depth_all_",pdb_name,".p"]), "rb" ))
        
        res_factor = 2.25 # see the documentation twhy 2.25 is chosen

        sur_res = []
        sur_res_cor_intial = []
        
        #% to find out the surface atoms residues 
        for i in range(0,len(fin_res_depth_all)):
            if fin_ca_depth_all[i] <= thresh_hold_ca:
                if fin_res_depth_all[i] <= thresh_hold_res:
                    sur_res.append(aminoacids[i])
                    # multiply each coordinate by 2 (just for increasing the resolution) and then round them to decimal numbers.
                    #sur_res_cor_intial_round.append([round(res_factor*coordinates[i][0]),round(res_factor*coordinates[i][1]),round(res_factor*coordinates[i][2])])
                    sur_res_cor_intial.append([res_factor*coordinates[i][0],res_factor*coordinates[i][1],res_factor*coordinates[i][2]])
                        
        if len(sur_res_cor_intial)==0:       
            print(pdb_name, ' not has single atom to satify')
        else:
#            '''first title the coordinates in principle direction; and finally round the o/p coordinates'''
#            sur_res_cor_tilted = self.change_C_alpha_principle_direc(sur_res_cor_intial)
            '''rotate in optimal direction'''
            sur_res_cor_tilted = self.rotate_C_alpha_optimal_direc(sur_res_cor_intial)
            #then shift the coordinates
            sur_res_cor=self.shift_coordinate(sur_res_cor_tilted)
        
            w_o_mass=False
            w_avg_mass=False
            #% then assign the property to the corresponding aminoacids  
            if self.w_o_mass and self.negation_included_all:
                property_surface, amino_acid_ok,w_o_mass= self.assign_prop_round_negation_w_o_mass(sur_res)#to check all amino acids in 20 of them
#            self.property_surface = property_surface
#            self.MOTIF_prop = MOTIF_prop
#            self.sur_res_cor = sur_res_cor
            elif self.assign_prop_round_negation_w_avg_mass_chk and self.negation_included_all:
                property_surface, amino_acid_ok,w_avg_mass= self.assign_prop_round_negation_w_avg_mass(sur_res)#to check all amino acids in 20 of them
            elif self.assign_prop_round_negation_w_avg_mass_chk and not self.negation_included_all:
                property_surface, amino_acid_ok,w_avg_mass= self.assign_prop_round_neg_yes_only_w_avg_mass(sur_res)#to check all amino acids in 20 of them
            else:
                property_surface,amino_acid_ok = self.assign_prop_round(sur_res)
                
            '''to save in the different from earlier'''
            if not os.path.isdir(self.saving_dir):
                os.makedirs(self.saving_dir)
                print("Making saving direcory: ",self.saving_dir )
            os.chdir("/")
            os.chdir(self.saving_dir)
            pdb_name = self.pdb_name
            if amino_acid_ok:
                whole_property_surface=np.array(property_surface)
               #assigning the space for the matrix that hold the property
                '''Adjusted without SITE'''
                if w_o_mass and negation_included_all:
                    coordinate_property=np.zeros((3,height,height,18))
                elif w_avg_mass and negation_included_all:
                    coordinate_property=np.zeros((3,height,height,19))
                elif w_avg_mass and not negation_included_all:                    
                    coordinate_property=np.zeros((3,height,height,14))
                else:
                    coordinate_property=np.zeros((3,height,height,20))
                sur_res_cor=np.array(sur_res_cor,dtype=int)
                save_numpy=True

                for i in range(0,len(sur_res_cor)):
                    if sur_res_cor[i,0]>(height-1) or sur_res_cor[i,1]>(height-1) or sur_res_cor[i,2]>(height-1):
                        print(' ')
                        print(pdb_name,' size issue: ')
                        print(' x size: ',np.max(sur_res_cor[:,0]))
                        print(' y size: ',np.max(sur_res_cor[:,1]))
                        print(' z size: ',np.max(sur_res_cor[:,2]))
                        print(' ')
                        save_numpy=False
                        break
                    else:
                        coordinate_property[0,sur_res_cor[i,0],sur_res_cor[i,1],:]=deepcopy(whole_property_surface[i,:])
                        coordinate_property[1,sur_res_cor[i,1],sur_res_cor[i,2],:]=deepcopy(whole_property_surface[i,:])
                        coordinate_property[2,sur_res_cor[i,0],sur_res_cor[i,2],:]=deepcopy(whole_property_surface[i,:])
                if save_numpy and amino_acid_ok: 
                    np.save(pdb_name, coordinate_property)
                else:
                    print('PDB ',pdb_name,' skipped due to size issue')
            else:
                print('PDB ',pdb_name,' skipped due to amino acid issue')


