# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""

   def __data_generation_full_clean(self, list_IDs_temp,list_IDs_class_temp):
        'Generates data containing batch_size samples only this part iws editted' # X : (n_samples, *dim, n_channels)
        # Initialization
        	q0_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q0_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q0_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
        
        q1_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q1_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q1_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
        
        q2_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q2_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q2_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
		
		q3_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q3_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q3_x3= np.empty((self.batch_size,*self.dim, self.n_channels))

		q4_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q4_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q4_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
		
		q5_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q5_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q5_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
		
		q6_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q6_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q6_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
		
		q7_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q7_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q7_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
		

        if self.Full_clean:
            y = np.empty((self.batch_size), dtype=int)
        else:
            raise ("Miss use of Function __data_generation_full_clean")
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data_temp = np.load(ID)
            	
            q0_x1[i,] = deepcopy(data_temp[0,0,:,:,:])
            q0_x2[i,] = deepcopy(data_temp[0,1,:,:,:])
            q0_x3[i,] = deepcopy(data_temp[0,2,:,:,:])
            
			q1_x1[i,] = deepcopy(data_temp[1,0,:,:,:])
            q1_x2[i,] = deepcopy(data_temp[1,1,:,:,:])
            q1_x3[i,] = deepcopy(data_temp[1,2,:,:,:])

			q2_x1[i,] = deepcopy(data_temp[2,0,:,:,:])
            q2_x2[i,] = deepcopy(data_temp[2,1,:,:,:])
            q2_x3[i,] = deepcopy(data_temp[2,2,:,:,:])
			
			q3_x1[i,] = deepcopy(data_temp[3,0,:,:,:])
            q3_x2[i,] = deepcopy(data_temp[3,1,:,:,:])
            q3_x3[i,] = deepcopy(data_temp[3,2,:,:,:])
			
			q4_x1[i,] = deepcopy(data_temp[4,0,:,:,:])
            q4_x2[i,] = deepcopy(data_temp[4,1,:,:,:])
            q4_x3[i,] = deepcopy(data_temp[4,2,:,:,:])
			
			q5_x1[i,] = deepcopy(data_temp[5,0,:,:,:])
            q5_x2[i,] = deepcopy(data_temp[5,1,:,:,:])
            q5_x3[i,] = deepcopy(data_temp[5,2,:,:,:])
			
			q6_x1[i,] = deepcopy(data_temp[6,0,:,:,:])
            q6_x2[i,] = deepcopy(data_temp[6,1,:,:,:])
            q6_x3[i,] = deepcopy(data_temp[6,2,:,:,:])
			
			q7_x1[i,] = deepcopy(data_temp[7,0,:,:,:])
            q7_x2[i,] = deepcopy(data_temp[7,1,:,:,:])
            q7_x3[i,] = deepcopy(data_temp[7,2,:,:,:])
            
            y[i] = self.labels[ID]
            if  self.labels[ID]>2:
                print('Wrongly placed id: ',ID)
        if len(list_IDs_temp)>self.batch_size:
            print('-----exceed batch size')
        X=[q0_x1,q0_x2,q0_x3,q1_x1,q1_x2,q1_x3,q2_x1,q2_x2,q2_x3,q3_x1,q3_x2,q3_x3,q4_x1,q4_x2,q4_x3,q5_x1,q5_x2,q5_x3,q6_x1,q6_x2,q6_x3,q7_x1,q7_x2,q7_x3]
        return X,keras.utils.to_categorical(y, num_classes=self.n_classes)
  
    def __data_generation(self, list_IDs_temp,list_IDs_class_temp):
      'Generates data containing batch_size samples only this part iws editted' # X : (n_samples, *dim, n_channels)
        # Initialization
        	q0_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q0_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q0_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
        
        q1_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q1_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q1_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
        
        q2_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q2_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q2_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
		
		q3_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q3_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q3_x3= np.empty((self.batch_size,*self.dim, self.n_channels))

		q4_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q4_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q4_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
		
		q5_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q5_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q5_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
		
		q6_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q6_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q6_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
		
		q7_x1= np.empty((self.batch_size,*self.dim, self.n_channels))
        q7_x2= np.empty((self.batch_size,*self.dim, self.n_channels))
        q7_x3= np.empty((self.batch_size,*self.dim, self.n_channels))
      if not self.Full_clean:
          y = np.empty((self.batch_size,self.n_classes), dtype=float)
      else:
          raise ("Miss use of Function __data_generation")
      # Generate data
      for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data_temp = np.load(ID)
            q0_x1[i,] = deepcopy(data_temp[0,0,:,:,:])
            q0_x2[i,] = deepcopy(data_temp[0,1,:,:,:])
            q0_x3[i,] = deepcopy(data_temp[0,2,:,:,:])
            
    			q1_x1[i,] = deepcopy(data_temp[1,0,:,:,:])
            q1_x2[i,] = deepcopy(data_temp[1,1,:,:,:])
            q1_x3[i,] = deepcopy(data_temp[1,2,:,:,:])
    
    			q2_x1[i,] = deepcopy(data_temp[2,0,:,:,:])
            q2_x2[i,] = deepcopy(data_temp[2,1,:,:,:])
            q2_x3[i,] = deepcopy(data_temp[2,2,:,:,:])
    			
    			q3_x1[i,] = deepcopy(data_temp[3,0,:,:,:])
            q3_x2[i,] = deepcopy(data_temp[3,1,:,:,:])
            q3_x3[i,] = deepcopy(data_temp[3,2,:,:,:])
    			
    			q4_x1[i,] = deepcopy(data_temp[4,0,:,:,:])
            q4_x2[i,] = deepcopy(data_temp[4,1,:,:,:])
            q4_x3[i,] = deepcopy(data_temp[4,2,:,:,:])
    			
    			q5_x1[i,] = deepcopy(data_temp[5,0,:,:,:])
            q5_x2[i,] = deepcopy(data_temp[5,1,:,:,:])
            q5_x3[i,] = deepcopy(data_temp[5,2,:,:,:])
    			
    			q6_x1[i,] = deepcopy(data_temp[6,0,:,:,:])
            q6_x2[i,] = deepcopy(data_temp[6,1,:,:,:])
            q6_x3[i,] = deepcopy(data_temp[6,2,:,:,:])
    			
    			q7_x1[i,] = deepcopy(data_temp[7,0,:,:,:])
            q7_x2[i,] = deepcopy(data_temp[7,1,:,:,:])
            q7_x3[i,] = deepcopy(data_temp[7,2,:,:,:])
          
            y[i,:] = deepcopy(list_IDs_class_temp[i])
    
      X=[q0_x1,q0_x2,q0_x3,q1_x1,q1_x2,q1_x3,q2_x1,q2_x2,q2_x3,q3_x1,q3_x2,q3_x3,q4_x1,q4_x2,q4_x3,q5_x1,q5_x2,q5_x3,q6_x1,q6_x2,q6_x3,q7_x1,q7_x2,q7_x3]
      return X,y
          
  
