# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
"""


        inp_q0_x1 = keras.Input(shape=(128, 128, self.channels))
        inp_q0_x2 = keras.Input(shape=(128, 128, self.channels))
        inp_q0_x3 = keras.Input(shape=(128, 128, self.channels))
        
        inp_q1_x1 = keras.Input(shape=(128, 128, self.channels))
        inp_q1_x2 = keras.Input(shape=(128, 128, self.channels))
        inp_q1_x3 = keras.Input(shape=(128, 128, self.channels))
       		
		inp_q2_x1 = keras.Input(shape=(128, 128, self.channels))
        inp_q2_x2 = keras.Input(shape=(128, 128, self.channels))
        inp_q2_x3 = keras.Input(shape=(128, 128, self.channels))
		
		inp_q3_x1 = keras.Input(shape=(128, 128, self.channels))
        inp_q3_x2 = keras.Input(shape=(128, 128, self.channels))
        inp_q3_x3 = keras.Input(shape=(128, 128, self.channels))
        		
		inp_q4_x1 = keras.Input(shape=(128, 128, self.channels))
        inp_q4_x2 = keras.Input(shape=(128, 128, self.channels))
        inp_q4_x3 = keras.Input(shape=(128, 128, self.channels))
        
		inp_q5_x1 = keras.Input(shape=(128, 128, self.channels))
        inp_q5_x2 = keras.Input(shape=(128, 128, self.channels))
        inp_q5_x3 = keras.Input(shape=(128, 128, self.channels))
        		
		inp_q6_x1 = keras.Input(shape=(128, 128, self.channels))
        inp_q6_x2 = keras.Input(shape=(128, 128, self.channels))
        inp_q6_x3 = keras.Input(shape=(128, 128, self.channels))
		
		inp_q7_x1 = keras.Input(shape=(128, 128, self.channels))
        inp_q7_x2 = keras.Input(shape=(128, 128, self.channels))
        inp_q7_x3 = keras.Input(shape=(128, 128, self.channels))
        
                
        tower_q0_x1 = self.parallel(inp_q0_x1,f_inc,f_d)
        tower_q0_x2 = self.parallel(inp_q0_x2,f_inc,f_d)
        tower_q0_x3 = self.parallel(inp_q0_x3,f_inc,f_d)
        
		tower_q1_x1 = self.parallel(inp_q1_x1,f_inc,f_d)
        tower_q1_x2 = self.parallel(inp_q1_x2,f_inc,f_d)
        tower_q1_x3 = self.parallel(inp_q1_x3,f_inc,f_d)
		
        tower_q2_x1 = self.parallel(inp_q2_x1,f_inc,f_d)
        tower_q2_x2 = self.parallel(inp_q2_x2,f_inc,f_d)
        tower_q2_x3 = self.parallel(inp_q2_x3,f_inc,f_d)

		tower_q3_x1 = self.parallel(inp_q3_x1,f_inc,f_d)
        tower_q3_x2 = self.parallel(inp_q3_x2,f_inc,f_d)
        tower_q3_x3 = self.parallel(inp_q3_x3,f_inc,f_d)
		
		tower_q4_x1 = self.parallel(inp_q4_x1,f_inc,f_d)
        tower_q4_x2 = self.parallel(inp_q4_x2,f_inc,f_d)
        tower_q4_x3 = self.parallel(inp_q4_x3,f_inc,f_d)
		
        tower_q5_x1 = self.parallel(inp_q5_x1,f_inc,f_d)
        tower_q5_x2 = self.parallel(inp_q5_x2,f_inc,f_d)
        tower_q5_x3 = self.parallel(inp_q5_x3,f_inc,f_d)
		
		tower_q6_x1 = self.parallel(inp_q6_x1,f_inc,f_d)
        tower_q6_x2 = self.parallel(inp_q6_x2,f_inc,f_d)
        tower_q6_x3 = self.parallel(inp_q6_x3,f_inc,f_d)
		
        tower_q7_x1 = self.parallel(inp_q7_x1,f_inc,f_d)
        tower_q7_x2 = self.parallel(inp_q7_x2,f_inc,f_d)
        tower_q7_x3 = self.parallel(inp_q7_x3,f_inc,f_d)
        
        merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3,tower_q1_x1, tower_q1_x2, tower_q1_x3,tower_q2_x1, tower_q2_x2, tower_q2_x3,tower_q3_x1, tower_q3_x2, tower_q3_x3,tower_q4_x1, tower_q4_x2, tower_q4_x3,tower_q5_x1, tower_q5_x2, tower_q5_x3,tower_q6_x1, tower_q6_x2, tower_q6_x3,tower_q7_x1, tower_q7_x2, tower_q7_x3], axis=1)
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation=self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation=self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        
        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        