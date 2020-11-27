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
    def __init__(self, loading_pikle_dir, saving_dir, pdb_name,thresh_hold_ca=7.2,thresh_hold_res=6.7):
        self.saving_dir = saving_dir
        self.pdb_name  = pdb_name #name of the class

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
        
            #% then assign the property to the corresponding aminoacids  
            property_surface = []
            amino_acid_ok=True#to check all amino acids in 20 of them
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
                else:
#                    print(i,' sur_res ',sur_res[i])
                    amino_acid_ok=False
#            self.property_surface = property_surface
#            self.MOTIF_prop = MOTIF_prop
#            self.sur_res_cor = sur_res_cor

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
                
                coordinate_property=np.zeros((3,200,200,17))
                sur_res_cor=np.array(sur_res_cor,dtype=int)
                save_numpy=True

                for i in range(0,len(sur_res_cor)):
                    if sur_res_cor[i,0]>199 or sur_res_cor[i,1]>199 or sur_res_cor[i,2]>199:
                        print(' ')
                        print(pdb_name,' size issue: ')
                        print(' x size: ',sur_res_cor[i,0])
                        print(' y size: ',sur_res_cor[i,1])
                        print(' z size: ',sur_res_cor[i,2])
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
        #calculate the eigen values and vigen vectors
        cen_coordinates_cov=np.cov(cen_coordinates.transpose())
        #eigenvalues = np.linalg.eigvals(cen_coordinates_cov)
        #eigenvecs = np.linalg.eig(cen_coordinates_cov)
        w, v = np.linalg.eig(cen_coordinates_cov)
        finalised_cart =np.matmul(cen_coordinates,v)
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