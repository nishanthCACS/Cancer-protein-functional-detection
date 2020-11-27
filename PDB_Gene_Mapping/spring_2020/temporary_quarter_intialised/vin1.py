# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""
import numpy as np
    def quaterize_coordinate(self,coordinates):
        """ 
        Take the coordinates and find which quarter it fell
        
        find the minimum position to move this shift the coordinates to make NN-train
        If the coordinate fell in quarter 1, assign 1 in quarter 1 for that coordinate
        
        zero_cl=give some gap to the quarters overlap to use the continuity between them
        
        prefered -5 to use if the kernal size 7 is used 
        """
        zero_cl=-5
        quarter=np.zeros((8,len(coordinates)))

        s_coordinates = deepcopy(coordinates)
        
        for i in range(0,len(coordinates)):
            if coordinates[i][2] >= zero_cl:
                s_coordinates[i][2] = coordinates[i][2]-zero_cl

                if coordinates[i][0] >= zero_cl and coordinates[i][1] >=zero_cl:
                    quarter[0][i]=1
                    s_coordinates[i][0] = coordinates[i][0]-zero_cl
                    s_coordinates[i][1] = coordinates[i][1]-zero_cl                 
                elif coordinates[i][0] < -zero_cl and coordinates[i][1] >= zero_cl:
                    quarter[1][i]=1
                    s_coordinates[i][0] = (-1)*(coordinates[i][0]+zero_cl)
                    s_coordinates[i][1] = coordinates[i][1]-zero_cl                                        
                elif coordinates[i][0] < -zero_cl and coordinates[i][1] < -zero_cl:
                    quarter[2][i]=1
                    s_coordinates[i][0] = (-1)*(coordinates[i][0]+zero_cl)
                    s_coordinates[i][1] = (-1)*(coordinates[i][1]+zero_cl)
                elif coordinates[i][0] >= zero_cl and coordinates[i][1] < -zero_cl:
                    quarter[3][i]=1
                    s_coordinates[i][0] = coordinates[i][0]-zero_cl
                    s_coordinates[i][1] = (-1)*(coordinates[i][1]+zero_cl)
                
            elif coordinates[i][2] <  -zero_cl:
                s_coordinates[i][2] = (-1)*(coordinates[i][2]+zero_cl)

                elif coordinates[i][0]>=zero_cl and coordinates[i][1] >=zero_cl:
                    quarter[4][i]=1
                    s_coordinates[i][0] = coordinates[i][0]-zero_cl
                    s_coordinates[i][1] = coordinates[i][1]-zero_cl
                elif coordinates[i][0] < -zero_cl and coordinates[i][1] >= zero_cl:
                    quarter[5][i]=1
                    s_coordinates[i][0] = (-1)*(coordinates[i][0]+zero_cl)
                    s_coordinates[i][1] = coordinates[i][1]-zero_cl
                elif coordinates[i][0] < -zero_cl and coordinates[i][1] < -zero_cl:
                    quarter[6][i]=1
                    s_coordinates[i][0] = (-1)*(coordinates[i][0]+zero_cl)
                    s_coordinates[i][1] = (-1)*(coordinates[i][1]+zero_cl)   
                elif coordinates[i][0] >= zero_cl and coordinates[i][1] < -zero_cl:
                    quarter[7][i]=1
                    s_coordinates[i][0] = coordinates[i][0]-zero_cl
                    s_coordinates[i][1] = (-1)*(coordinates[i][1]+zero_cl)
        
        return s_coordinates,quarter
    
    
    def quarter_finalise_properties(quarter,whole_property_surface,sur_res_cor):
        """
        This assign the properties for the given coordinates of quater and save it
        """
        height = 128
        if amino_acid_ok:
            coordinate_property=np.zeros((8,3,height,height,self.channel))
            sur_res_cor=np.array(sur_res_cor,dtype=int)
    
            save_numpy=True
            for quat_num in range(0,8):
                quarter_sat= deepcopy(quarter[quat_num][:])
                for i in range(0,len(quarter_sat)):
                    if quarter_sat[i]==1:
                        if sur_res_cor[i,0]>(height-1) or sur_res_cor[i,1]>(height-1) or sur_res_cor[i,2]>(height-1):
                            print(' ')
                            print(pdb_name,' size issue: ')
                            print(' x size: ',np.max(sur_res_cor[:,0]))
                            print(' y size: ',np.max(sur_res_cor[:,1]))
                            print(' z size: ',np.max(sur_res_cor[:,2]))
                            print(' ')
                            save_numpy=False
                            #if one of the quatery not satisfied then drop the whole PDB set
                            break
                        else:
                            coordinate_property[quat_num,0,sur_res_cor[i,0],sur_res_cor[i,1],:]=deepcopy(whole_property_surface[i,:])
                            coordinate_property[quat_num,1,sur_res_cor[i,1],sur_res_cor[i,2],:]=deepcopy(whole_property_surface[i,:])
                            coordinate_property[quat_num,2,sur_res_cor[i,0],sur_res_cor[i,2],:]=deepcopy(whole_property_surface[i,:])
                if not save_numpy:
                    break
            if save_numpy:
                np.save(pdb_name, coordinate_property)
            else:
                print("Size exceed")
        else:
            print("Aminoacid not satisfied")            
