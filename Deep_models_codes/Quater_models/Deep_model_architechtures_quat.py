#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:45:12 2019

@author: c00294860

This codes have different architechtures
"""
from tensorflow import keras
from tensorflow.keras import layers


'''Defining the swish -activation function'''
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation


class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x, beta = 1):
    '''
    "Swish:

    In 2017, Researchers at Google Brain, published a new paper where they proposed their novel activation function named as Swish.
     Swish is a gated version of Sigmoid Activation Function, where the mathematical formula is: f(x, β) = x * sigmoid(x, β) 
     [Often Swish is referred as SILU or simply Swish-1 when β = 1].

    Their results showcased a significant improvement in Top-1 Accuracy for ImageNet at 0.9% better than ReLU for Mobile 
    NASNet-A and 0.7% for Inception ResNet v2." from https://towardsdatascience.com/mish-8283934a72df on  Fri Dec 27 14:44:30 2019
    
    '''
    return (x * sigmoid(beta * x))

get_custom_objects().clear()
get_custom_objects().update({'swish': Swish(swish)})   
 
class model_accidental_par_try:
    
    def __init__(self,channels=17,activation='relu'):

        self.activation = activation
        self.channels = channels
        self.square_height=128
        self.square_width=128
        
    def model_maker(self,p1=3,d1=32):
        
        k1=p1
        k2=p1
        k3=p1
        k4=p1
        inputs = keras.Input(shape=(self.square_height, self.square_width, self.channels))

        xy = layers.Conv2D(d1, (k1,k1),strides=(1,1),padding='same',activation= self.activation)(inputs)
        block_1_xy_output = layers.MaxPooling2D(pool_size=(4, 4))(xy)
        
        xy = layers.Conv2D(d1, (k2,k2),strides=(1,1),padding='same',activation= self.activation)(block_1_xy_output)
        block_2_xy_output = layers.MaxPooling2D(pool_size=(2, 2))(xy)
    
        xy = layers.Conv2D(2*d1, (k3,k3), padding='same',activation= self.activation)(block_2_xy_output)
        xy = layers.MaxPooling2D(pool_size=(2, 2))(xy)

        xy = layers.Conv2D(2*d1, (k4,k4), padding='same',activation= self.activation)(xy)
        block_3_xy_output = layers.MaxPooling2D(pool_size=(2, 2))(xy)

        self.parallel = keras.models.Model(inputs=inputs,outputs= block_3_xy_output, name='parallel')     

        
        channels =self.channels

        inp_q0_x1 = keras.Input(shape=(self.square_height,self.square_width, channels))
        inp_q0_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q0_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        inp_q1_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
           		
        inp_q2_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q3_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q4_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        inp_q5_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q6_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q7_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        
        tower_q0_x1 = self.parallel(inp_q0_x1)
        tower_q0_x2 = self.parallel(inp_q0_x2)
        tower_q0_x3 = self.parallel(inp_q0_x3)
        
        tower_q1_x1 = self.parallel(inp_q1_x1)
        tower_q1_x2 = self.parallel(inp_q1_x2)
        tower_q1_x3 = self.parallel(inp_q1_x3)
        		
        tower_q2_x1 = self.parallel(inp_q2_x1)
        tower_q2_x2 = self.parallel(inp_q2_x2)
        tower_q2_x3 = self.parallel(inp_q2_x3)
        
        tower_q3_x1 = self.parallel(inp_q3_x1)
        tower_q3_x2 = self.parallel(inp_q3_x2)
        tower_q3_x3 = self.parallel(inp_q3_x3)
        		
        tower_q4_x1 = self.parallel(inp_q4_x1)
        tower_q4_x2 = self.parallel(inp_q4_x2)
        tower_q4_x3 = self.parallel(inp_q4_x3)
        		
        tower_q5_x1 = self.parallel(inp_q5_x1)
        tower_q5_x2 = self.parallel(inp_q5_x2)
        tower_q5_x3 = self.parallel(inp_q5_x3)
        		
        tower_q6_x1 = self.parallel(inp_q6_x1)
        tower_q6_x2 = self.parallel(inp_q6_x2)
        tower_q6_x3 = self.parallel(inp_q6_x3)
        		
        tower_q7_x1 = self.parallel(inp_q7_x1)
        tower_q7_x2 = self.parallel(inp_q7_x2)
        tower_q7_x3 = self.parallel(inp_q7_x3)
        
        # merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3], axis=1)
        merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3,tower_q1_x1,tower_q1_x2,tower_q1_x3,tower_q2_x1, tower_q2_x2, tower_q2_x3,tower_q3_x1, tower_q3_x2, tower_q3_x3,tower_q4_x1, tower_q4_x2, tower_q4_x3,tower_q5_x1, tower_q5_x2, tower_q5_x3,tower_q6_x1, tower_q6_x2, tower_q6_x3,tower_q7_x1, tower_q7_x2, tower_q7_x3], axis=1)
      
        model_name =''.join(['model_accidental_use_same_NN_p1_',str(p1),'_d1_',str(d1),'.h5'])
            
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation= self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation= self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.models.Model(inputs=[inp_q0_x1,inp_q0_x2,inp_q0_x3 ,inp_q1_x1,inp_q1_x2,inp_q1_x3,inp_q2_x1,inp_q2_x2,inp_q2_x3,inp_q3_x1,inp_q3_x2,inp_q3_x3,inp_q4_x1,inp_q4_x2,inp_q4_x3,inp_q5_x1,inp_q5_x2,inp_q5_x3,inp_q6_x1,inp_q6_x2,inp_q6_x3,inp_q7_x1,inp_q7_x2,inp_q7_x3], outputs=outputs, name=model_name)
        
        return model,model_name
      
class model_accidental_try:
    
    def __init__(self,channels=17,activation='relu'):

        self.activation = activation
        self.channels = channels
        self.square_height=128
        self.square_width=128
        
    def parallel(self,inputs,k1=7,k2=7,k3=7,k4=7,d1=32):
    
        xy = layers.Conv2D(d1, (k1,k1),strides=(1,1),padding='same',activation= self.activation)(inputs)
        block_1_xy_output = layers.MaxPooling2D(pool_size=(4, 4))(xy)
        
        xy = layers.Conv2D(d1, (k2,k2),strides=(1,1),padding='same',activation= self.activation)(block_1_xy_output)
        block_2_xy_output = layers.MaxPooling2D(pool_size=(2, 2))(xy)
    
        xy = layers.Conv2D(2*d1, (k3,k3), padding='same',activation= self.activation)(block_2_xy_output)
        xy = layers.MaxPooling2D(pool_size=(2, 2))(xy)

        xy = layers.Conv2D(2*d1, (k4,k4), padding='same',activation= self.activation)(xy)
        block_3_xy_output = layers.MaxPooling2D(pool_size=(2, 2))(xy)
        return block_3_xy_output
 
    def model_maker(self,p1=7,d1=32):
        
        channels =self.channels

        inp_q0_x1 = keras.Input(shape=(self.square_height,self.square_width, channels))
        inp_q0_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q0_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        inp_q1_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
           		
        inp_q2_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q3_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q4_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        inp_q5_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q6_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q7_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        
        tower_q0_x1 = self.parallel(inp_q0_x1)
        tower_q0_x2 = self.parallel(inp_q0_x2)
        tower_q0_x3 = self.parallel(inp_q0_x3)
        
        tower_q1_x1 = self.parallel(inp_q1_x1)
        tower_q1_x2 = self.parallel(inp_q1_x2)
        tower_q1_x3 = self.parallel(inp_q1_x3)
        		
        tower_q2_x1 = self.parallel(inp_q2_x1)
        tower_q2_x2 = self.parallel(inp_q2_x2)
        tower_q2_x3 = self.parallel(inp_q2_x3)
        
        tower_q3_x1 = self.parallel(inp_q3_x1)
        tower_q3_x2 = self.parallel(inp_q3_x2)
        tower_q3_x3 = self.parallel(inp_q3_x3)
        		
        tower_q4_x1 = self.parallel(inp_q4_x1)
        tower_q4_x2 = self.parallel(inp_q4_x2)
        tower_q4_x3 = self.parallel(inp_q4_x3)
        		
        tower_q5_x1 = self.parallel(inp_q5_x1)
        tower_q5_x2 = self.parallel(inp_q5_x2)
        tower_q5_x3 = self.parallel(inp_q5_x3)
        		
        tower_q6_x1 = self.parallel(inp_q6_x1)
        tower_q6_x2 = self.parallel(inp_q6_x2)
        tower_q6_x3 = self.parallel(inp_q6_x3)
        		
        tower_q7_x1 = self.parallel(inp_q7_x1)
        tower_q7_x2 = self.parallel(inp_q7_x2)
        tower_q7_x3 = self.parallel(inp_q7_x3)
        
        # merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3], axis=1)
        merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3,tower_q1_x1,tower_q1_x2,tower_q1_x3,tower_q2_x1, tower_q2_x2, tower_q2_x3,tower_q3_x1, tower_q3_x2, tower_q3_x3,tower_q4_x1, tower_q4_x2, tower_q4_x3,tower_q5_x1, tower_q5_x2, tower_q5_x3,tower_q6_x1, tower_q6_x2, tower_q6_x3,tower_q7_x1, tower_q7_x2, tower_q7_x3], axis=1)
      
        model_name =''.join(['model_accidental_p1_',str(p1),'_d1_',str(d1),'.h5'])
            
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation= self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation= self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.models.Model(inputs=[inp_q0_x1,inp_q0_x2,inp_q0_x3 ,inp_q1_x1,inp_q1_x2,inp_q1_x3,inp_q2_x1,inp_q2_x2,inp_q2_x3,inp_q3_x1,inp_q3_x2,inp_q3_x3,inp_q4_x1,inp_q4_x2,inp_q4_x3,inp_q5_x1,inp_q5_x2,inp_q5_x3,inp_q6_x1,inp_q6_x2,inp_q6_x3,inp_q7_x1,inp_q7_x2,inp_q7_x3], outputs=outputs, name=model_name)
        
        return model,model_name


class model_brain_incept_RESIDUAL_quat_with_k3_to_7_vin_1:

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu',reducer_7x1_selector=1):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.square_height=128
        self.square_width=128
        self.reducer_7x1_selector=reducer_7x1_selector
        self.tower_min_max_only = tower_min_max_only
        
    def layer_1_reducer(self,lay_1_all_output,d_lay_1_to_2,m_p=4):
        
        layer_1_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_1_all_output)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_1_to_7=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        return incept_1_to_3,incept_1_to_5,incept_1_to_7,layer_1_pool

    def layer_2_reducer(self,lay_2_all_output,d_lay_1_to_2,m_p=3):
        
        layer_2_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_2_all_output)       
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_1_to_7=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        return incept_1_to_3,incept_1_to_5,incept_1_to_7,layer_2_pool

    def layer_3_reducer(self,lay_3_all_output,d_lay_3_to_4,m_p=2):
        
        layer_3_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_3_all_output)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_1_to_5=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_1_to_7=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        return incept_1_to_3,incept_1_to_5,incept_1_to_7,layer_3_pool

    def layer_4_final(self,lay_4_all_output,d_lay_3_to_4,m_p=2):
        layer_4_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_4_all_output)
        incept_1_to_final =layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_4_pool)
        return incept_1_to_final
    
    def model_maker(self,f_inc=1,f_d=1):

        d1=128*f_inc
        d3=32*f_inc
        d5=32*f_inc
        d7=32*f_inc

        d_max=32*f_inc  
        d_lay_1_to_2 = 32*f_d
        d_lay_3_to_4 = 64*f_d
        
        m_p_l1=4#maxpool layer 1 size
        m_p_l2=3
        m_p_l3=2
        m_p_l4=2
        
        channels=self.channels
        
        inputs = keras.Input(shape=(self.square_height, self.square_width, self.channels))
        '''Inputs intialisation'''
        inp_q0_x1 = keras.Input(shape=(self.square_height,self.square_width, channels))
        inp_q0_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q0_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))   
        
        inp_q1_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
           		
        inp_q2_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q3_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q4_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        inp_q5_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q6_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q7_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x3 = keras.Input(shape=(self.square_height, self.square_width, channels)) 
        
        lay_1_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(inputs)
        if self.reducer_7x1_selector==0:
            lay_1_incept_7_1 = layers.Conv2D(d7, (1,7),strides=(1,1),padding='same',activation=self.activation)(inputs)
            lay_1_incept_7 = layers.Conv2D(d7, (7,1),strides=(1,1),padding='same',activation=self.activation)(lay_1_incept_7_1)
        elif self.reducer_7x1_selector==1:
            lay_1_incept_7_1 = layers.Conv2D(d7, (1,7),strides=(1,1),padding='same',activation=self.activation)(inputs)
            lay_1_incept_7_2 = layers.Conv2D(d7, (7,1),strides=(1,1),padding='same',activation=self.activation)(lay_1_incept_7_1)
            lay_1_incept_7 = layers.Conv2D(d7, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_1_incept_7_2)
    
        else:
            lay_1_incept_7 = layers.Conv2D(d7, (7,7),strides=(1,1),padding='same',activation=self.activation)(inputs)

        lay_1_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
        lay_1_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_1_incept_max_pool)
        
        lay_1_all_output = layers.concatenate([lay_1_incept_1, lay_1_incept_3,lay_1_incept_5,lay_1_incept_7,lay_1_incept_max_pool_depth], axis=3)
        '''layer 1 general network'''    
        layer_1_INCEPT_Net = keras.models.Model(inputs, lay_1_all_output, name='layer_1_INCEPT')     
        

        '''place the size of NN layer_2 inputs'''
        #since the stride is 1 ans same padding these equation works
        lay_1_height=int(self.square_height/m_p_l1)
        lay_1_width = int(self.square_width/m_p_l1)
        
        
        lay_1_inp_incept_1_to_3 = keras.Input(shape=(lay_1_height,lay_1_width, d_lay_1_to_2))
        lay_1_inp_incept_1_to_5 = keras.Input(shape=(lay_1_height,lay_1_width, d_lay_1_to_2))
        lay_1_inp_incept_1_to_7 = keras.Input(shape=(lay_1_height,lay_1_width, d_lay_1_to_2))
        inp_lay_1_pool = keras.Input(shape=(lay_1_height,lay_1_width,d1+d3+d5+d7+d_max))
        
        lay_2_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_1_pool)
        lay_2_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_1_inp_incept_1_to_3)
        lay_2_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_1_inp_incept_1_to_5)
        if self.reducer_7x1_selector==0:
            lay_2_incept_7_1 = layers.Conv2D(d7, (1,7),strides=(1,1),padding='same',activation=self.activation)(lay_1_inp_incept_1_to_7)
            lay_2_incept_7 = layers.Conv2D(d7, (7,1),strides=(1,1),padding='same',activation=self.activation)(lay_2_incept_7_1)
        elif self.reducer_7x1_selector==1:
           lay_2_incept_7_1 = layers.Conv2D(d7, (1,7),strides=(1,1),padding='same',activation=self.activation)(lay_1_inp_incept_1_to_7)
           lay_2_incept_7_2 = layers.Conv2D(d7, (7,1),strides=(1,1),padding='same',activation=self.activation)(lay_2_incept_7_1)
           lay_2_incept_7 = layers.Conv2D(d7, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_2_incept_7_2)     
        else:
            lay_2_incept_7 = layers.Conv2D(d7, (7,7),strides=(1,1),padding='same',activation=self.activation)(lay_1_inp_incept_1_to_7)
 
        lay_2_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_1_pool)
        lay_2_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_2_incept_max_pool)
        
        lay_2_all_output = layers.concatenate([lay_2_incept_1, lay_2_incept_3,lay_2_incept_5,lay_2_incept_7,lay_2_incept_max_pool_depth], axis=3)
        '''layer 2 general network'''    
        layer_2_INCEPT_Net = keras.models.Model(inputs=[lay_1_inp_incept_1_to_3,lay_1_inp_incept_1_to_5,lay_1_inp_incept_1_to_7,inp_lay_1_pool],outputs= lay_2_all_output, name='layer_2_INCEPT')     

        '''place the size of NN layer_3 inputs'''
        #since the stride is 1 ans same padding these equation works
        lay_2_height=int(lay_1_height/m_p_l2)
        lay_2_width = int(lay_1_height/m_p_l2)
        
        lay_2_inp_incept_1_to_3 = keras.Input(shape=(lay_2_height,lay_1_width, d_lay_1_to_2))
        lay_2_inp_incept_1_to_5 = keras.Input(shape=(lay_2_width,lay_1_width, d_lay_1_to_2))      
        lay_2_inp_incept_1_to_7 = keras.Input(shape=(lay_2_width,lay_1_width, d_lay_1_to_2))      

        inp_lay_2_pool = keras.Input(shape=(lay_2_height,lay_1_width,d1+d3+d5+d7+d_max))

        lay_3_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_2_pool)
        lay_3_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_2_inp_incept_1_to_3)
        lay_3_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_2_inp_incept_1_to_5)
        if self.reducer_7x1_selector==0:
            lay_3_incept_7_1 = layers.Conv2D(d7, (1,7),strides=(1,1),padding='same',activation=self.activation)(lay_2_inp_incept_1_to_7)
            lay_3_incept_7 = layers.Conv2D(d7, (7,1),strides=(1,1),padding='same',activation=self.activation)(lay_3_incept_7_1)
        elif self.reducer_7x1_selector==1:
            lay_3_incept_7_1 = layers.Conv2D(d7, (1,7),strides=(1,1),padding='same',activation=self.activation)(lay_2_inp_incept_1_to_7)
            lay_3_incept_7_2 = layers.Conv2D(d7, (7,1),strides=(1,1),padding='same',activation=self.activation)(lay_3_incept_7_1)
            lay_3_incept_7 = layers.Conv2D(d7, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_3_incept_7_2)       
        else:
            lay_3_incept_7 = layers.Conv2D(d7, (7,7),strides=(1,1),padding='same',activation=self.activation)(lay_2_inp_incept_1_to_7)
        
        lay_3_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_2_pool)
        lay_3_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_3_incept_max_pool)
        
        lay_3_all_output = layers.concatenate([lay_3_incept_1, lay_3_incept_3,lay_3_incept_5,lay_3_incept_7,lay_3_incept_max_pool_depth], axis=3)
        '''layer 3 general network'''    
        layer_3_INCEPT_Net = keras.models.Model(inputs=[lay_2_inp_incept_1_to_3,lay_2_inp_incept_1_to_5,lay_2_inp_incept_1_to_7,inp_lay_2_pool],outputs= lay_3_all_output, name='layer_3_INCEPT')     

        '''place the size of NN layer_4 inputs'''
        #since the stride is 1 ans same padding these equation works
        lay_3_height=int(lay_2_height/m_p_l3)
        lay_3_width = int(lay_2_height/m_p_l3)
        
        lay_3_inp_incept_1_to_3 = keras.Input(shape=(lay_3_height, lay_3_width,d_lay_3_to_4))
        lay_3_inp_incept_1_to_5 = keras.Input(shape=(lay_3_height, lay_3_width,d_lay_3_to_4))
        lay_3_inp_incept_1_to_7 = keras.Input(shape=(lay_3_height, lay_3_width,d_lay_3_to_4))

        inp_lay_3_pool = keras.Input(shape=(lay_3_height, lay_3_width,d1+d3+d5+d7+d_max))       
        
        lay_4_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_3_pool)
        lay_4_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_3_inp_incept_1_to_3)
        lay_4_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_3_inp_incept_1_to_5)
        if self.reducer_7x1_selector==0:
            lay_4_incept_7_1 = layers.Conv2D(d7, (1,7),strides=(1,1),padding='same',activation=self.activation)(lay_3_inp_incept_1_to_7)
            lay_4_incept_7 = layers.Conv2D(d7, (7,1),strides=(1,1),padding='same',activation=self.activation)(lay_4_incept_7_1)
        elif self.reducer_7x1_selector==1:
            lay_4_incept_7_1 = layers.Conv2D(d7, (1,7),strides=(1,1),padding='same',activation=self.activation)(lay_3_inp_incept_1_to_7)
            lay_4_incept_7_2 = layers.Conv2D(d7, (7,1),strides=(1,1),padding='same',activation=self.activation)(lay_4_incept_7_1)
            lay_4_incept_7 = layers.Conv2D(d7, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_4_incept_7_2)       
        else:
            lay_4_incept_7 = layers.Conv2D(d7, (7,7),strides=(1,1),padding='same',activation=self.activation)(lay_3_inp_incept_1_to_7)
 
        lay_4_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_3_pool)
        lay_4_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_4_incept_max_pool)
        
        lay_4_all_output = layers.concatenate([lay_4_incept_1, lay_4_incept_3,lay_4_incept_5,lay_4_incept_7,lay_4_incept_max_pool_depth], axis=3)
    
        '''layer 3 general network'''    
        layer_4_INCEPT_Net = keras.models.Model(inputs=[lay_3_inp_incept_1_to_3,lay_3_inp_incept_1_to_5,lay_3_inp_incept_1_to_7,inp_lay_3_pool], outputs=lay_4_all_output, name='layer_4_INCEPT')     
        
        '''Applying layer_1 in projections'''
        #layer1 output of projection 1
        inp_q0_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q0_x1) 
        inp_q0_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q0_x2) 
        inp_q0_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q0_x3) 
        
        inp_q1_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q1_x1) 
        inp_q1_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q1_x2) 
        inp_q1_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q1_x3) 

        inp_q2_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q2_x1) 
        inp_q2_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q2_x2) 
        inp_q2_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q2_x3) 

        inp_q3_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q3_x1) 
        inp_q3_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q3_x2) 
        inp_q3_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q3_x3)     
        
        inp_q4_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q4_x1) 
        inp_q4_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q4_x2) 
        inp_q4_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q4_x3) 
        
        inp_q5_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q5_x1) 
        inp_q5_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q5_x2) 
        inp_q5_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q5_x3) 

        inp_q6_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q6_x1) 
        inp_q6_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q6_x2) 
        inp_q6_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q6_x3) 

        inp_q7_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q7_x1) 
        inp_q7_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q7_x2) 
        inp_q7_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q7_x3) 
        
        lay_1_inp_q0_x1_incept_1_to_3,lay_1_inp_q0_x1_incept_1_to_5,lay_1_inp_q0_x1_incept_1_to_7,inp_q0_x1_lay_1_pool = self.layer_1_reducer(inp_q0_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q0_x2_incept_1_to_3,lay_1_inp_q0_x2_incept_1_to_5,lay_1_inp_q0_x2_incept_1_to_7,inp_q0_x2_lay_1_pool = self.layer_1_reducer(inp_q0_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q0_x3_incept_1_to_3,lay_1_inp_q0_x3_incept_1_to_5,lay_1_inp_q0_x3_incept_1_to_7,inp_q0_x3_lay_1_pool = self.layer_1_reducer(inp_q0_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
       
        lay_1_inp_q1_x1_incept_1_to_3,lay_1_inp_q1_x1_incept_1_to_5,lay_1_inp_q1_x1_incept_1_to_7,inp_q1_x1_lay_1_pool = self.layer_1_reducer(inp_q1_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q1_x2_incept_1_to_3,lay_1_inp_q1_x2_incept_1_to_5,lay_1_inp_q1_x2_incept_1_to_7,inp_q1_x2_lay_1_pool = self.layer_1_reducer(inp_q1_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q1_x3_incept_1_to_3,lay_1_inp_q1_x3_incept_1_to_5,lay_1_inp_q1_x3_incept_1_to_7,inp_q1_x3_lay_1_pool = self.layer_1_reducer(inp_q1_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        
        lay_1_inp_q2_x1_incept_1_to_3,lay_1_inp_q2_x1_incept_1_to_5,lay_1_inp_q2_x1_incept_1_to_7,inp_q2_x1_lay_1_pool = self.layer_1_reducer(inp_q2_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q2_x2_incept_1_to_3,lay_1_inp_q2_x2_incept_1_to_5,lay_1_inp_q2_x2_incept_1_to_7,inp_q2_x2_lay_1_pool = self.layer_1_reducer(inp_q2_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q2_x3_incept_1_to_3,lay_1_inp_q2_x3_incept_1_to_5,lay_1_inp_q2_x3_incept_1_to_7,inp_q2_x3_lay_1_pool = self.layer_1_reducer(inp_q2_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
       
        lay_1_inp_q3_x1_incept_1_to_3,lay_1_inp_q3_x1_incept_1_to_5,lay_1_inp_q3_x1_incept_1_to_7,inp_q3_x1_lay_1_pool = self.layer_1_reducer(inp_q3_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q3_x2_incept_1_to_3,lay_1_inp_q3_x2_incept_1_to_5,lay_1_inp_q3_x2_incept_1_to_7,inp_q3_x2_lay_1_pool = self.layer_1_reducer(inp_q3_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q3_x3_incept_1_to_3,lay_1_inp_q3_x3_incept_1_to_5,lay_1_inp_q3_x3_incept_1_to_7,inp_q3_x3_lay_1_pool = self.layer_1_reducer(inp_q3_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        
        lay_1_inp_q4_x1_incept_1_to_3,lay_1_inp_q4_x1_incept_1_to_5,lay_1_inp_q4_x1_incept_1_to_7,inp_q4_x1_lay_1_pool = self.layer_1_reducer(inp_q4_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q4_x2_incept_1_to_3,lay_1_inp_q4_x2_incept_1_to_5,lay_1_inp_q4_x2_incept_1_to_7,inp_q4_x2_lay_1_pool = self.layer_1_reducer(inp_q4_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q4_x3_incept_1_to_3,lay_1_inp_q4_x3_incept_1_to_5,lay_1_inp_q4_x3_incept_1_to_7,inp_q4_x3_lay_1_pool = self.layer_1_reducer(inp_q4_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
       
        lay_1_inp_q5_x1_incept_1_to_3,lay_1_inp_q5_x1_incept_1_to_5,lay_1_inp_q5_x1_incept_1_to_7,inp_q5_x1_lay_1_pool = self.layer_1_reducer(inp_q5_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q5_x2_incept_1_to_3,lay_1_inp_q5_x2_incept_1_to_5,lay_1_inp_q5_x2_incept_1_to_7,inp_q5_x2_lay_1_pool = self.layer_1_reducer(inp_q5_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q5_x3_incept_1_to_3,lay_1_inp_q5_x3_incept_1_to_5,lay_1_inp_q5_x3_incept_1_to_7,inp_q5_x3_lay_1_pool = self.layer_1_reducer(inp_q5_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        
        lay_1_inp_q6_x1_incept_1_to_3,lay_1_inp_q6_x1_incept_1_to_5,lay_1_inp_q6_x1_incept_1_to_7,inp_q6_x1_lay_1_pool = self.layer_1_reducer(inp_q6_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q6_x2_incept_1_to_3,lay_1_inp_q6_x2_incept_1_to_5,lay_1_inp_q6_x2_incept_1_to_7,inp_q6_x2_lay_1_pool = self.layer_1_reducer(inp_q6_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q6_x3_incept_1_to_3,lay_1_inp_q6_x3_incept_1_to_5,lay_1_inp_q6_x3_incept_1_to_7,inp_q6_x3_lay_1_pool = self.layer_1_reducer(inp_q6_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
       
        lay_1_inp_q7_x1_incept_1_to_3,lay_1_inp_q7_x1_incept_1_to_5,lay_1_inp_q7_x1_incept_1_to_7,inp_q7_x1_lay_1_pool = self.layer_1_reducer(inp_q7_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q7_x2_incept_1_to_3,lay_1_inp_q7_x2_incept_1_to_5,lay_1_inp_q7_x2_incept_1_to_7,inp_q7_x2_lay_1_pool = self.layer_1_reducer(inp_q7_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q7_x3_incept_1_to_3,lay_1_inp_q7_x3_incept_1_to_5,lay_1_inp_q7_x3_incept_1_to_7,inp_q7_x3_lay_1_pool = self.layer_1_reducer(inp_q7_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)               

        '''Applying layer_2 in projections'''
        #layer1 output of projection 1
        inp_q0_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q0_x1_incept_1_to_3,lay_1_inp_q0_x1_incept_1_to_5,lay_1_inp_q0_x1_incept_1_to_7,inp_q0_x1_lay_1_pool]) 
        inp_q0_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q0_x2_incept_1_to_3,lay_1_inp_q0_x2_incept_1_to_5,lay_1_inp_q0_x2_incept_1_to_7,inp_q0_x2_lay_1_pool]) 
        inp_q0_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q0_x3_incept_1_to_3,lay_1_inp_q0_x3_incept_1_to_5,lay_1_inp_q0_x3_incept_1_to_7,inp_q0_x3_lay_1_pool]) 

        inp_q1_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q1_x1_incept_1_to_3,lay_1_inp_q1_x1_incept_1_to_5,lay_1_inp_q1_x1_incept_1_to_7,inp_q1_x1_lay_1_pool]) 
        inp_q1_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q1_x2_incept_1_to_3,lay_1_inp_q1_x2_incept_1_to_5,lay_1_inp_q1_x2_incept_1_to_7,inp_q1_x2_lay_1_pool]) 
        inp_q1_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q1_x3_incept_1_to_3,lay_1_inp_q1_x3_incept_1_to_5,lay_1_inp_q1_x3_incept_1_to_7,inp_q1_x3_lay_1_pool]) 

        inp_q2_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q2_x1_incept_1_to_3,lay_1_inp_q2_x1_incept_1_to_5,lay_1_inp_q2_x1_incept_1_to_7,inp_q2_x1_lay_1_pool]) 
        inp_q2_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q2_x2_incept_1_to_3,lay_1_inp_q2_x2_incept_1_to_5,lay_1_inp_q2_x2_incept_1_to_7,inp_q2_x2_lay_1_pool]) 
        inp_q2_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q2_x3_incept_1_to_3,lay_1_inp_q2_x3_incept_1_to_5,lay_1_inp_q2_x3_incept_1_to_7,inp_q2_x3_lay_1_pool]) 

        inp_q3_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q3_x1_incept_1_to_3,lay_1_inp_q3_x1_incept_1_to_5,lay_1_inp_q3_x1_incept_1_to_7,inp_q3_x1_lay_1_pool]) 
        inp_q3_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q3_x2_incept_1_to_3,lay_1_inp_q3_x2_incept_1_to_5,lay_1_inp_q3_x2_incept_1_to_7,inp_q3_x2_lay_1_pool]) 
        inp_q3_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q3_x3_incept_1_to_3,lay_1_inp_q3_x3_incept_1_to_5,lay_1_inp_q3_x3_incept_1_to_7,inp_q3_x3_lay_1_pool]) 

        inp_q4_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q4_x1_incept_1_to_3,lay_1_inp_q4_x1_incept_1_to_5,lay_1_inp_q4_x1_incept_1_to_7,inp_q4_x1_lay_1_pool]) 
        inp_q4_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q4_x2_incept_1_to_3,lay_1_inp_q4_x2_incept_1_to_5,lay_1_inp_q4_x2_incept_1_to_7,inp_q4_x2_lay_1_pool]) 
        inp_q4_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q4_x3_incept_1_to_3,lay_1_inp_q4_x3_incept_1_to_5,lay_1_inp_q4_x3_incept_1_to_7,inp_q4_x3_lay_1_pool]) 

        inp_q5_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q5_x1_incept_1_to_3,lay_1_inp_q5_x1_incept_1_to_5,lay_1_inp_q5_x1_incept_1_to_7,inp_q5_x1_lay_1_pool]) 
        inp_q5_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q5_x2_incept_1_to_3,lay_1_inp_q5_x2_incept_1_to_5,lay_1_inp_q5_x2_incept_1_to_7,inp_q5_x2_lay_1_pool]) 
        inp_q5_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q5_x3_incept_1_to_3,lay_1_inp_q5_x3_incept_1_to_5,lay_1_inp_q5_x3_incept_1_to_7,inp_q5_x3_lay_1_pool]) 

        inp_q6_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q6_x1_incept_1_to_3,lay_1_inp_q6_x1_incept_1_to_5,lay_1_inp_q6_x1_incept_1_to_7,inp_q6_x1_lay_1_pool]) 
        inp_q6_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q6_x2_incept_1_to_3,lay_1_inp_q6_x2_incept_1_to_5,lay_1_inp_q6_x2_incept_1_to_7,inp_q6_x2_lay_1_pool]) 
        inp_q6_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q6_x3_incept_1_to_3,lay_1_inp_q6_x3_incept_1_to_5,lay_1_inp_q6_x3_incept_1_to_7,inp_q6_x3_lay_1_pool]) 

        inp_q7_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q7_x1_incept_1_to_3,lay_1_inp_q7_x1_incept_1_to_5,lay_1_inp_q7_x1_incept_1_to_7,inp_q7_x1_lay_1_pool]) 
        inp_q7_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q7_x2_incept_1_to_3,lay_1_inp_q7_x2_incept_1_to_5,lay_1_inp_q7_x2_incept_1_to_7,inp_q7_x2_lay_1_pool]) 
        inp_q7_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q7_x3_incept_1_to_3,lay_1_inp_q7_x3_incept_1_to_5,lay_1_inp_q7_x3_incept_1_to_7,inp_q7_x3_lay_1_pool]) 
  
        #make residUAL NETWORK
        inp_q0_x_1_lay_2_all_output= layers.add([inp_q0_x1_lay_1_pool, inp_q0_x_1_lay_2_incept_all_output])
        inp_q0_x_2_lay_2_all_output= layers.add([inp_q0_x2_lay_1_pool, inp_q0_x_2_lay_2_incept_all_output])
        inp_q0_x_3_lay_2_all_output= layers.add([inp_q0_x3_lay_1_pool, inp_q0_x_3_lay_2_incept_all_output])
        
        inp_q1_x_1_lay_2_all_output= layers.add([inp_q1_x1_lay_1_pool, inp_q1_x_1_lay_2_incept_all_output])
        inp_q1_x_2_lay_2_all_output= layers.add([inp_q1_x2_lay_1_pool, inp_q1_x_2_lay_2_incept_all_output])
        inp_q1_x_3_lay_2_all_output= layers.add([inp_q1_x3_lay_1_pool, inp_q1_x_3_lay_2_incept_all_output])
        
        inp_q2_x_1_lay_2_all_output= layers.add([inp_q2_x1_lay_1_pool, inp_q2_x_1_lay_2_incept_all_output])
        inp_q2_x_2_lay_2_all_output= layers.add([inp_q2_x2_lay_1_pool, inp_q2_x_2_lay_2_incept_all_output])
        inp_q2_x_3_lay_2_all_output= layers.add([inp_q2_x3_lay_1_pool, inp_q2_x_3_lay_2_incept_all_output])
        
        inp_q3_x_1_lay_2_all_output= layers.add([inp_q3_x1_lay_1_pool, inp_q3_x_1_lay_2_incept_all_output])
        inp_q3_x_2_lay_2_all_output= layers.add([inp_q3_x2_lay_1_pool, inp_q3_x_2_lay_2_incept_all_output])
        inp_q3_x_3_lay_2_all_output= layers.add([inp_q3_x3_lay_1_pool, inp_q3_x_3_lay_2_incept_all_output])

        inp_q4_x_1_lay_2_all_output= layers.add([inp_q4_x1_lay_1_pool, inp_q4_x_1_lay_2_incept_all_output])
        inp_q4_x_2_lay_2_all_output= layers.add([inp_q4_x2_lay_1_pool, inp_q4_x_2_lay_2_incept_all_output])
        inp_q4_x_3_lay_2_all_output= layers.add([inp_q4_x3_lay_1_pool, inp_q4_x_3_lay_2_incept_all_output])
        
        inp_q5_x_1_lay_2_all_output= layers.add([inp_q5_x1_lay_1_pool, inp_q5_x_1_lay_2_incept_all_output])
        inp_q5_x_2_lay_2_all_output= layers.add([inp_q5_x2_lay_1_pool, inp_q5_x_2_lay_2_incept_all_output])
        inp_q5_x_3_lay_2_all_output= layers.add([inp_q5_x3_lay_1_pool, inp_q5_x_3_lay_2_incept_all_output])
        
        inp_q6_x_1_lay_2_all_output= layers.add([inp_q6_x1_lay_1_pool, inp_q6_x_1_lay_2_incept_all_output])
        inp_q6_x_2_lay_2_all_output= layers.add([inp_q6_x2_lay_1_pool, inp_q6_x_2_lay_2_incept_all_output])
        inp_q6_x_3_lay_2_all_output= layers.add([inp_q6_x3_lay_1_pool, inp_q6_x_3_lay_2_incept_all_output])
        
        inp_q7_x_1_lay_2_all_output= layers.add([inp_q7_x1_lay_1_pool, inp_q7_x_1_lay_2_incept_all_output])
        inp_q7_x_2_lay_2_all_output= layers.add([inp_q7_x2_lay_1_pool, inp_q7_x_2_lay_2_incept_all_output])
        inp_q7_x_3_lay_2_all_output= layers.add([inp_q7_x3_lay_1_pool, inp_q7_x_3_lay_2_incept_all_output])
       
        #layer2 output of projection 1
        lay_2_inp_q0_x1_incept_1_to_3,lay_2_inp_q0_x1_incept_1_to_5,lay_2_inp_q0_x1_incept_1_to_7,inp_q0_x1_lay_2_pool = self.layer_2_reducer(inp_q0_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q0_x2_incept_1_to_3,lay_2_inp_q0_x2_incept_1_to_5,lay_2_inp_q0_x2_incept_1_to_7,inp_q0_x2_lay_2_pool = self.layer_2_reducer(inp_q0_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q0_x3_incept_1_to_3,lay_2_inp_q0_x3_incept_1_to_5,lay_2_inp_q0_x3_incept_1_to_7,inp_q0_x3_lay_2_pool = self.layer_2_reducer(inp_q0_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q1_x1_incept_1_to_3,lay_2_inp_q1_x1_incept_1_to_5,lay_2_inp_q1_x1_incept_1_to_7,inp_q1_x1_lay_2_pool = self.layer_2_reducer(inp_q1_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q1_x2_incept_1_to_3,lay_2_inp_q1_x2_incept_1_to_5,lay_2_inp_q1_x2_incept_1_to_7,inp_q1_x2_lay_2_pool = self.layer_2_reducer(inp_q1_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q1_x3_incept_1_to_3,lay_2_inp_q1_x3_incept_1_to_5,lay_2_inp_q1_x3_incept_1_to_7,inp_q1_x3_lay_2_pool = self.layer_2_reducer(inp_q1_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

        lay_2_inp_q2_x1_incept_1_to_3,lay_2_inp_q2_x1_incept_1_to_5,lay_2_inp_q2_x1_incept_1_to_7,inp_q2_x1_lay_2_pool = self.layer_2_reducer(inp_q2_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q2_x2_incept_1_to_3,lay_2_inp_q2_x2_incept_1_to_5,lay_2_inp_q2_x2_incept_1_to_7,inp_q2_x2_lay_2_pool = self.layer_2_reducer(inp_q2_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q2_x3_incept_1_to_3,lay_2_inp_q2_x3_incept_1_to_5,lay_2_inp_q2_x3_incept_1_to_7,inp_q2_x3_lay_2_pool = self.layer_2_reducer(inp_q2_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q3_x1_incept_1_to_3,lay_2_inp_q3_x1_incept_1_to_5,lay_2_inp_q3_x1_incept_1_to_7,inp_q3_x1_lay_2_pool = self.layer_2_reducer(inp_q3_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q3_x2_incept_1_to_3,lay_2_inp_q3_x2_incept_1_to_5,lay_2_inp_q3_x2_incept_1_to_7,inp_q3_x2_lay_2_pool = self.layer_2_reducer(inp_q3_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q3_x3_incept_1_to_3,lay_2_inp_q3_x3_incept_1_to_5,lay_2_inp_q3_x3_incept_1_to_7,inp_q3_x3_lay_2_pool = self.layer_2_reducer(inp_q3_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q4_x1_incept_1_to_3,lay_2_inp_q4_x1_incept_1_to_5,lay_2_inp_q4_x1_incept_1_to_7,inp_q4_x1_lay_2_pool = self.layer_2_reducer(inp_q4_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q4_x2_incept_1_to_3,lay_2_inp_q4_x2_incept_1_to_5,lay_2_inp_q4_x2_incept_1_to_7,inp_q4_x2_lay_2_pool = self.layer_2_reducer(inp_q4_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q4_x3_incept_1_to_3,lay_2_inp_q4_x3_incept_1_to_5,lay_2_inp_q4_x3_incept_1_to_7,inp_q4_x3_lay_2_pool = self.layer_2_reducer(inp_q4_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q5_x1_incept_1_to_3,lay_2_inp_q5_x1_incept_1_to_5,lay_2_inp_q5_x1_incept_1_to_7,inp_q5_x1_lay_2_pool = self.layer_2_reducer(inp_q5_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q5_x2_incept_1_to_3,lay_2_inp_q5_x2_incept_1_to_5,lay_2_inp_q5_x2_incept_1_to_7,inp_q5_x2_lay_2_pool = self.layer_2_reducer(inp_q5_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q5_x3_incept_1_to_3,lay_2_inp_q5_x3_incept_1_to_5,lay_2_inp_q5_x3_incept_1_to_7,inp_q5_x3_lay_2_pool = self.layer_2_reducer(inp_q5_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

        lay_2_inp_q6_x1_incept_1_to_3,lay_2_inp_q6_x1_incept_1_to_5,lay_2_inp_q6_x1_incept_1_to_7,inp_q6_x1_lay_2_pool = self.layer_2_reducer(inp_q6_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q6_x2_incept_1_to_3,lay_2_inp_q6_x2_incept_1_to_5,lay_2_inp_q6_x2_incept_1_to_7,inp_q6_x2_lay_2_pool = self.layer_2_reducer(inp_q6_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q6_x3_incept_1_to_3,lay_2_inp_q6_x3_incept_1_to_5,lay_2_inp_q6_x3_incept_1_to_7,inp_q6_x3_lay_2_pool = self.layer_2_reducer(inp_q6_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q7_x1_incept_1_to_3,lay_2_inp_q7_x1_incept_1_to_5,lay_2_inp_q7_x1_incept_1_to_7,inp_q7_x1_lay_2_pool = self.layer_2_reducer(inp_q7_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q7_x2_incept_1_to_3,lay_2_inp_q7_x2_incept_1_to_5,lay_2_inp_q7_x2_incept_1_to_7,inp_q7_x2_lay_2_pool = self.layer_2_reducer(inp_q7_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q7_x3_incept_1_to_3,lay_2_inp_q7_x3_incept_1_to_5,lay_2_inp_q7_x3_incept_1_to_7,inp_q7_x3_lay_2_pool = self.layer_2_reducer(inp_q7_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        '''Applying layer_3 in projections'''
        #layer3 output of projection 1
        inp_q0_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q0_x1_incept_1_to_3,lay_2_inp_q0_x1_incept_1_to_5,lay_2_inp_q0_x1_incept_1_to_7,inp_q0_x1_lay_2_pool]) 
        inp_q0_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q0_x2_incept_1_to_3,lay_2_inp_q0_x2_incept_1_to_5,lay_2_inp_q0_x2_incept_1_to_7,inp_q0_x2_lay_2_pool]) 
        inp_q0_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q0_x3_incept_1_to_3,lay_2_inp_q0_x3_incept_1_to_5,lay_2_inp_q0_x3_incept_1_to_7,inp_q0_x3_lay_2_pool]) 
      
        inp_q1_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q1_x1_incept_1_to_3,lay_2_inp_q1_x1_incept_1_to_5,lay_2_inp_q1_x1_incept_1_to_7,inp_q1_x1_lay_2_pool]) 
        inp_q1_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q1_x2_incept_1_to_3,lay_2_inp_q1_x2_incept_1_to_5,lay_2_inp_q1_x2_incept_1_to_7,inp_q1_x2_lay_2_pool]) 
        inp_q1_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q1_x3_incept_1_to_3,lay_2_inp_q1_x3_incept_1_to_5,lay_2_inp_q1_x3_incept_1_to_7,inp_q1_x3_lay_2_pool]) 
 
        inp_q2_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q2_x1_incept_1_to_3,lay_2_inp_q2_x1_incept_1_to_5,lay_2_inp_q2_x1_incept_1_to_7,inp_q2_x1_lay_2_pool]) 
        inp_q2_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q2_x2_incept_1_to_3,lay_2_inp_q2_x2_incept_1_to_5,lay_2_inp_q2_x2_incept_1_to_7,inp_q2_x2_lay_2_pool]) 
        inp_q2_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q2_x3_incept_1_to_3,lay_2_inp_q2_x3_incept_1_to_5,lay_2_inp_q2_x3_incept_1_to_7,inp_q2_x3_lay_2_pool]) 
      
        inp_q3_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q3_x1_incept_1_to_3,lay_2_inp_q3_x1_incept_1_to_5,lay_2_inp_q3_x1_incept_1_to_7,inp_q3_x1_lay_2_pool]) 
        inp_q3_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q3_x2_incept_1_to_3,lay_2_inp_q3_x2_incept_1_to_5,lay_2_inp_q3_x2_incept_1_to_7,inp_q3_x2_lay_2_pool]) 
        inp_q3_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q3_x3_incept_1_to_3,lay_2_inp_q3_x3_incept_1_to_5,lay_2_inp_q3_x3_incept_1_to_7,inp_q3_x3_lay_2_pool])        

        inp_q4_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q4_x1_incept_1_to_3,lay_2_inp_q4_x1_incept_1_to_5,lay_2_inp_q4_x1_incept_1_to_7,inp_q4_x1_lay_2_pool]) 
        inp_q4_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q4_x2_incept_1_to_3,lay_2_inp_q4_x2_incept_1_to_5,lay_2_inp_q4_x2_incept_1_to_7,inp_q4_x2_lay_2_pool]) 
        inp_q4_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q4_x3_incept_1_to_3,lay_2_inp_q4_x3_incept_1_to_5,lay_2_inp_q4_x3_incept_1_to_7,inp_q4_x3_lay_2_pool]) 
      
        inp_q5_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q5_x1_incept_1_to_3,lay_2_inp_q5_x1_incept_1_to_5,lay_2_inp_q5_x1_incept_1_to_7,inp_q5_x1_lay_2_pool]) 
        inp_q5_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q5_x2_incept_1_to_3,lay_2_inp_q5_x2_incept_1_to_5,lay_2_inp_q5_x2_incept_1_to_7,inp_q5_x2_lay_2_pool]) 
        inp_q5_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q5_x3_incept_1_to_3,lay_2_inp_q5_x3_incept_1_to_5,lay_2_inp_q5_x3_incept_1_to_7,inp_q5_x3_lay_2_pool]) 
 
        inp_q6_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q6_x1_incept_1_to_3,lay_2_inp_q6_x1_incept_1_to_5,lay_2_inp_q6_x1_incept_1_to_7,inp_q6_x1_lay_2_pool]) 
        inp_q6_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q6_x2_incept_1_to_3,lay_2_inp_q6_x2_incept_1_to_5,lay_2_inp_q6_x2_incept_1_to_7,inp_q6_x2_lay_2_pool]) 
        inp_q6_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q6_x3_incept_1_to_3,lay_2_inp_q6_x3_incept_1_to_5,lay_2_inp_q6_x3_incept_1_to_7,inp_q6_x3_lay_2_pool]) 
      
        inp_q7_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q7_x1_incept_1_to_3,lay_2_inp_q7_x1_incept_1_to_5,lay_2_inp_q7_x1_incept_1_to_7,inp_q7_x1_lay_2_pool]) 
        inp_q7_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q7_x2_incept_1_to_3,lay_2_inp_q7_x2_incept_1_to_5,lay_2_inp_q7_x2_incept_1_to_7,inp_q7_x2_lay_2_pool]) 
        inp_q7_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q7_x3_incept_1_to_3,lay_2_inp_q7_x3_incept_1_to_5,lay_2_inp_q7_x3_incept_1_to_7,inp_q7_x3_lay_2_pool])      
        #make residUAL NETWORK
        inp_q0_x_1_lay_3_all_output= layers.add([inp_q0_x1_lay_2_pool, inp_q0_x_1_lay_3_incept_all_output])
        inp_q0_x_2_lay_3_all_output= layers.add([inp_q0_x2_lay_2_pool, inp_q0_x_2_lay_3_incept_all_output])
        inp_q0_x_3_lay_3_all_output= layers.add([inp_q0_x3_lay_2_pool, inp_q0_x_3_lay_3_incept_all_output])
        
        inp_q1_x_1_lay_3_all_output= layers.add([inp_q1_x1_lay_2_pool, inp_q1_x_1_lay_3_incept_all_output])
        inp_q1_x_2_lay_3_all_output= layers.add([inp_q1_x2_lay_2_pool, inp_q1_x_2_lay_3_incept_all_output])
        inp_q1_x_3_lay_3_all_output= layers.add([inp_q1_x3_lay_2_pool, inp_q1_x_3_lay_3_incept_all_output])
        
        inp_q2_x_1_lay_3_all_output= layers.add([inp_q2_x1_lay_2_pool, inp_q2_x_1_lay_3_incept_all_output])
        inp_q2_x_2_lay_3_all_output= layers.add([inp_q2_x2_lay_2_pool, inp_q2_x_2_lay_3_incept_all_output])
        inp_q2_x_3_lay_3_all_output= layers.add([inp_q2_x3_lay_2_pool, inp_q2_x_3_lay_3_incept_all_output])
        
        inp_q3_x_1_lay_3_all_output= layers.add([inp_q3_x1_lay_2_pool, inp_q3_x_1_lay_3_incept_all_output])
        inp_q3_x_2_lay_3_all_output= layers.add([inp_q3_x2_lay_2_pool, inp_q3_x_2_lay_3_incept_all_output])
        inp_q3_x_3_lay_3_all_output= layers.add([inp_q3_x3_lay_2_pool, inp_q3_x_3_lay_3_incept_all_output])

        inp_q4_x_1_lay_3_all_output= layers.add([inp_q4_x1_lay_2_pool, inp_q4_x_1_lay_3_incept_all_output])
        inp_q4_x_2_lay_3_all_output= layers.add([inp_q4_x2_lay_2_pool, inp_q4_x_2_lay_3_incept_all_output])
        inp_q4_x_3_lay_3_all_output= layers.add([inp_q4_x3_lay_2_pool, inp_q4_x_3_lay_3_incept_all_output])
        
        inp_q5_x_1_lay_3_all_output= layers.add([inp_q5_x1_lay_2_pool, inp_q5_x_1_lay_3_incept_all_output])
        inp_q5_x_2_lay_3_all_output= layers.add([inp_q5_x2_lay_2_pool, inp_q5_x_2_lay_3_incept_all_output])
        inp_q5_x_3_lay_3_all_output= layers.add([inp_q5_x3_lay_2_pool, inp_q5_x_3_lay_3_incept_all_output])
        
        inp_q6_x_1_lay_3_all_output= layers.add([inp_q6_x1_lay_2_pool, inp_q6_x_1_lay_3_incept_all_output])
        inp_q6_x_2_lay_3_all_output= layers.add([inp_q6_x2_lay_2_pool, inp_q6_x_2_lay_3_incept_all_output])
        inp_q6_x_3_lay_3_all_output= layers.add([inp_q6_x3_lay_2_pool, inp_q6_x_3_lay_3_incept_all_output])
        
        inp_q7_x_1_lay_3_all_output= layers.add([inp_q7_x1_lay_2_pool, inp_q7_x_1_lay_3_incept_all_output])
        inp_q7_x_2_lay_3_all_output= layers.add([inp_q7_x2_lay_2_pool, inp_q7_x_2_lay_3_incept_all_output])
        inp_q7_x_3_lay_3_all_output= layers.add([inp_q7_x3_lay_2_pool, inp_q7_x_3_lay_3_incept_all_output])        
        #layer3 output of projection 1
        lay_3_inp_q0_x1_incept_1_to_3,lay_3_inp_q0_x1_incept_1_to_5,lay_3_inp_q0_x1_incept_1_to_7,inp_q0_x1_lay_3_pool = self.layer_3_reducer(inp_q0_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q0_x2_incept_1_to_3,lay_3_inp_q0_x2_incept_1_to_5,lay_3_inp_q0_x2_incept_1_to_7,inp_q0_x2_lay_3_pool = self.layer_3_reducer(inp_q0_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q0_x3_incept_1_to_3,lay_3_inp_q0_x3_incept_1_to_5,lay_3_inp_q0_x3_incept_1_to_7,inp_q0_x3_lay_3_pool = self.layer_3_reducer(inp_q0_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q1_x1_incept_1_to_3,lay_3_inp_q1_x1_incept_1_to_5,lay_3_inp_q1_x1_incept_1_to_7,inp_q1_x1_lay_3_pool = self.layer_3_reducer(inp_q1_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q1_x2_incept_1_to_3,lay_3_inp_q1_x2_incept_1_to_5,lay_3_inp_q1_x2_incept_1_to_7,inp_q1_x2_lay_3_pool = self.layer_3_reducer(inp_q1_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q1_x3_incept_1_to_3,lay_3_inp_q1_x3_incept_1_to_5,lay_3_inp_q1_x3_incept_1_to_7,inp_q1_x3_lay_3_pool = self.layer_3_reducer(inp_q1_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
     
        lay_3_inp_q2_x1_incept_1_to_3,lay_3_inp_q2_x1_incept_1_to_5,lay_3_inp_q2_x1_incept_1_to_7,inp_q2_x1_lay_3_pool = self.layer_3_reducer(inp_q2_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q2_x2_incept_1_to_3,lay_3_inp_q2_x2_incept_1_to_5,lay_3_inp_q2_x2_incept_1_to_7,inp_q2_x2_lay_3_pool = self.layer_3_reducer(inp_q2_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q2_x3_incept_1_to_3,lay_3_inp_q2_x3_incept_1_to_5,lay_3_inp_q2_x3_incept_1_to_7,inp_q2_x3_lay_3_pool = self.layer_3_reducer(inp_q2_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q3_x1_incept_1_to_3,lay_3_inp_q3_x1_incept_1_to_5,lay_3_inp_q3_x1_incept_1_to_7,inp_q3_x1_lay_3_pool = self.layer_3_reducer(inp_q3_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q3_x2_incept_1_to_3,lay_3_inp_q3_x2_incept_1_to_5,lay_3_inp_q3_x2_incept_1_to_7,inp_q3_x2_lay_3_pool = self.layer_3_reducer(inp_q3_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q3_x3_incept_1_to_3,lay_3_inp_q3_x3_incept_1_to_5,lay_3_inp_q3_x3_incept_1_to_7,inp_q3_x3_lay_3_pool = self.layer_3_reducer(inp_q3_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q4_x1_incept_1_to_3,lay_3_inp_q4_x1_incept_1_to_5,lay_3_inp_q4_x1_incept_1_to_7,inp_q4_x1_lay_3_pool = self.layer_3_reducer(inp_q4_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q4_x2_incept_1_to_3,lay_3_inp_q4_x2_incept_1_to_5,lay_3_inp_q4_x2_incept_1_to_7,inp_q4_x2_lay_3_pool = self.layer_3_reducer(inp_q4_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q4_x3_incept_1_to_3,lay_3_inp_q4_x3_incept_1_to_5,lay_3_inp_q4_x3_incept_1_to_7,inp_q4_x3_lay_3_pool = self.layer_3_reducer(inp_q4_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q5_x1_incept_1_to_3,lay_3_inp_q5_x1_incept_1_to_5,lay_3_inp_q5_x1_incept_1_to_7,inp_q5_x1_lay_3_pool = self.layer_3_reducer(inp_q5_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q5_x2_incept_1_to_3,lay_3_inp_q5_x2_incept_1_to_5,lay_3_inp_q5_x2_incept_1_to_7,inp_q5_x2_lay_3_pool = self.layer_3_reducer(inp_q5_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q5_x3_incept_1_to_3,lay_3_inp_q5_x3_incept_1_to_5,lay_3_inp_q5_x3_incept_1_to_7,inp_q5_x3_lay_3_pool = self.layer_3_reducer(inp_q5_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
     
        lay_3_inp_q6_x1_incept_1_to_3,lay_3_inp_q6_x1_incept_1_to_5,lay_3_inp_q6_x1_incept_1_to_7,inp_q6_x1_lay_3_pool = self.layer_3_reducer(inp_q6_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q6_x2_incept_1_to_3,lay_3_inp_q6_x2_incept_1_to_5,lay_3_inp_q6_x2_incept_1_to_7,inp_q6_x2_lay_3_pool = self.layer_3_reducer(inp_q6_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q6_x3_incept_1_to_3,lay_3_inp_q6_x3_incept_1_to_5,lay_3_inp_q6_x3_incept_1_to_7,inp_q6_x3_lay_3_pool = self.layer_3_reducer(inp_q6_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q7_x1_incept_1_to_3,lay_3_inp_q7_x1_incept_1_to_5,lay_3_inp_q7_x1_incept_1_to_7,inp_q7_x1_lay_3_pool = self.layer_3_reducer(inp_q7_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q7_x2_incept_1_to_3,lay_3_inp_q7_x2_incept_1_to_5,lay_3_inp_q7_x2_incept_1_to_7,inp_q7_x2_lay_3_pool = self.layer_3_reducer(inp_q7_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q7_x3_incept_1_to_3,lay_3_inp_q7_x3_incept_1_to_5,lay_3_inp_q7_x3_incept_1_to_7,inp_q7_x3_lay_3_pool = self.layer_3_reducer(inp_q7_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)        
        '''Applying layer_4 in projections'''
        inp_q0_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q0_x1_incept_1_to_3,lay_3_inp_q0_x1_incept_1_to_5,lay_3_inp_q0_x1_incept_1_to_7,inp_q0_x1_lay_3_pool]) 
        inp_q0_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q0_x2_incept_1_to_3,lay_3_inp_q0_x2_incept_1_to_5,lay_3_inp_q0_x2_incept_1_to_7,inp_q0_x2_lay_3_pool]) 
        inp_q0_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q0_x3_incept_1_to_3,lay_3_inp_q0_x3_incept_1_to_5,lay_3_inp_q0_x3_incept_1_to_7,inp_q0_x3_lay_3_pool]) 
        
        inp_q1_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q1_x1_incept_1_to_3,lay_3_inp_q1_x1_incept_1_to_5,lay_3_inp_q1_x1_incept_1_to_7,inp_q1_x1_lay_3_pool]) 
        inp_q1_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q1_x2_incept_1_to_3,lay_3_inp_q1_x2_incept_1_to_5,lay_3_inp_q1_x2_incept_1_to_7,inp_q1_x2_lay_3_pool]) 
        inp_q1_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q1_x3_incept_1_to_3,lay_3_inp_q1_x3_incept_1_to_5,lay_3_inp_q1_x3_incept_1_to_7,inp_q1_x3_lay_3_pool]) 
        
        inp_q2_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q2_x1_incept_1_to_3,lay_3_inp_q2_x1_incept_1_to_5,lay_3_inp_q2_x1_incept_1_to_7,inp_q2_x1_lay_3_pool]) 
        inp_q2_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q2_x2_incept_1_to_3,lay_3_inp_q2_x2_incept_1_to_5,lay_3_inp_q2_x2_incept_1_to_7,inp_q2_x2_lay_3_pool]) 
        inp_q2_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q2_x3_incept_1_to_3,lay_3_inp_q2_x3_incept_1_to_5,lay_3_inp_q2_x3_incept_1_to_7,inp_q2_x3_lay_3_pool]) 
        
        inp_q3_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q3_x1_incept_1_to_3,lay_3_inp_q3_x1_incept_1_to_5,lay_3_inp_q3_x1_incept_1_to_7,inp_q3_x1_lay_3_pool]) 
        inp_q3_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q3_x2_incept_1_to_3,lay_3_inp_q3_x2_incept_1_to_5,lay_3_inp_q3_x2_incept_1_to_7,inp_q3_x2_lay_3_pool]) 
        inp_q3_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q3_x3_incept_1_to_3,lay_3_inp_q3_x3_incept_1_to_5,lay_3_inp_q3_x3_incept_1_to_7,inp_q3_x3_lay_3_pool]) 

        inp_q4_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q4_x1_incept_1_to_3,lay_3_inp_q4_x1_incept_1_to_5,lay_3_inp_q4_x1_incept_1_to_7,inp_q4_x1_lay_3_pool]) 
        inp_q4_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q4_x2_incept_1_to_3,lay_3_inp_q4_x2_incept_1_to_5,lay_3_inp_q4_x2_incept_1_to_7,inp_q4_x2_lay_3_pool]) 
        inp_q4_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q4_x3_incept_1_to_3,lay_3_inp_q4_x3_incept_1_to_5,lay_3_inp_q4_x3_incept_1_to_7,inp_q4_x3_lay_3_pool]) 
        
        inp_q5_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q5_x1_incept_1_to_3,lay_3_inp_q5_x1_incept_1_to_5,lay_3_inp_q5_x1_incept_1_to_7,inp_q5_x1_lay_3_pool]) 
        inp_q5_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q5_x2_incept_1_to_3,lay_3_inp_q5_x2_incept_1_to_5,lay_3_inp_q5_x2_incept_1_to_7,inp_q5_x2_lay_3_pool]) 
        inp_q5_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q5_x3_incept_1_to_3,lay_3_inp_q5_x3_incept_1_to_5,lay_3_inp_q5_x3_incept_1_to_7,inp_q5_x3_lay_3_pool]) 
        
        inp_q6_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q6_x1_incept_1_to_3,lay_3_inp_q6_x1_incept_1_to_5,lay_3_inp_q6_x1_incept_1_to_7,inp_q6_x1_lay_3_pool]) 
        inp_q6_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q6_x2_incept_1_to_3,lay_3_inp_q6_x2_incept_1_to_5,lay_3_inp_q6_x2_incept_1_to_7,inp_q6_x2_lay_3_pool]) 
        inp_q6_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q6_x3_incept_1_to_3,lay_3_inp_q6_x3_incept_1_to_5,lay_3_inp_q6_x3_incept_1_to_7,inp_q6_x3_lay_3_pool]) 
        
        inp_q7_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q7_x1_incept_1_to_3,lay_3_inp_q7_x1_incept_1_to_5,lay_3_inp_q7_x1_incept_1_to_7,inp_q7_x1_lay_3_pool]) 
        inp_q7_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q7_x2_incept_1_to_3,lay_3_inp_q7_x2_incept_1_to_5,lay_3_inp_q7_x2_incept_1_to_7,inp_q7_x2_lay_3_pool]) 
        inp_q7_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q7_x3_incept_1_to_3,lay_3_inp_q7_x3_incept_1_to_5,lay_3_inp_q7_x3_incept_1_to_7,inp_q7_x3_lay_3_pool]) 
                                                    
        #make residUAL NETWORK
        inp_q0_x_1_lay_4_all_output= layers.add([inp_q0_x1_lay_3_pool, inp_q0_x_1_lay_4_incept_all_output])
        inp_q0_x_2_lay_4_all_output= layers.add([inp_q0_x2_lay_3_pool, inp_q0_x_2_lay_4_incept_all_output])
        inp_q0_x_3_lay_4_all_output= layers.add([inp_q0_x3_lay_3_pool, inp_q0_x_3_lay_4_incept_all_output])
        
        inp_q1_x_1_lay_4_all_output= layers.add([inp_q1_x1_lay_3_pool, inp_q1_x_1_lay_4_incept_all_output])
        inp_q1_x_2_lay_4_all_output= layers.add([inp_q1_x2_lay_3_pool, inp_q1_x_2_lay_4_incept_all_output])
        inp_q1_x_3_lay_4_all_output= layers.add([inp_q1_x3_lay_3_pool, inp_q1_x_3_lay_4_incept_all_output])
        
        inp_q2_x_1_lay_4_all_output= layers.add([inp_q2_x1_lay_3_pool, inp_q2_x_1_lay_4_incept_all_output])
        inp_q2_x_2_lay_4_all_output= layers.add([inp_q2_x2_lay_3_pool, inp_q2_x_2_lay_4_incept_all_output])
        inp_q2_x_3_lay_4_all_output= layers.add([inp_q2_x3_lay_3_pool, inp_q2_x_3_lay_4_incept_all_output])
        
        inp_q3_x_1_lay_4_all_output= layers.add([inp_q3_x1_lay_3_pool, inp_q3_x_1_lay_4_incept_all_output])
        inp_q3_x_2_lay_4_all_output= layers.add([inp_q3_x2_lay_3_pool, inp_q3_x_2_lay_4_incept_all_output])
        inp_q3_x_3_lay_4_all_output= layers.add([inp_q3_x3_lay_3_pool, inp_q3_x_3_lay_4_incept_all_output])

        inp_q4_x_1_lay_4_all_output= layers.add([inp_q4_x1_lay_3_pool, inp_q4_x_1_lay_4_incept_all_output])
        inp_q4_x_2_lay_4_all_output= layers.add([inp_q4_x2_lay_3_pool, inp_q4_x_2_lay_4_incept_all_output])
        inp_q4_x_3_lay_4_all_output= layers.add([inp_q4_x3_lay_3_pool, inp_q4_x_3_lay_4_incept_all_output])
        
        inp_q5_x_1_lay_4_all_output= layers.add([inp_q5_x1_lay_3_pool, inp_q5_x_1_lay_4_incept_all_output])
        inp_q5_x_2_lay_4_all_output= layers.add([inp_q5_x2_lay_3_pool, inp_q5_x_2_lay_4_incept_all_output])
        inp_q5_x_3_lay_4_all_output= layers.add([inp_q5_x3_lay_3_pool, inp_q5_x_3_lay_4_incept_all_output])
        
        inp_q6_x_1_lay_4_all_output= layers.add([inp_q6_x1_lay_3_pool, inp_q6_x_1_lay_4_incept_all_output])
        inp_q6_x_2_lay_4_all_output= layers.add([inp_q6_x2_lay_3_pool, inp_q6_x_2_lay_4_incept_all_output])
        inp_q6_x_3_lay_4_all_output= layers.add([inp_q6_x3_lay_3_pool, inp_q6_x_3_lay_4_incept_all_output])
        
        inp_q7_x_1_lay_4_all_output= layers.add([inp_q7_x1_lay_3_pool, inp_q7_x_1_lay_4_incept_all_output])
        inp_q7_x_2_lay_4_all_output= layers.add([inp_q7_x2_lay_3_pool, inp_q7_x_2_lay_4_incept_all_output])
        inp_q7_x_3_lay_4_all_output= layers.add([inp_q7_x3_lay_3_pool, inp_q7_x_3_lay_4_incept_all_output])
 
        #parallel = keras.models.Model(inputs,all_output, name='parallel') 
        tower_q0_x1 = self.layer_4_final(inp_q0_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q0_x2 = self.layer_4_final(inp_q0_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q0_x3 = self.layer_4_final(inp_q0_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q1_x1 = self.layer_4_final(inp_q1_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q1_x2 = self.layer_4_final(inp_q1_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q1_x3 = self.layer_4_final(inp_q1_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

        tower_q2_x1 = self.layer_4_final(inp_q2_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q2_x2 = self.layer_4_final(inp_q2_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q2_x3 = self.layer_4_final(inp_q2_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q3_x1 = self.layer_4_final(inp_q3_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q3_x2 = self.layer_4_final(inp_q3_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q3_x3 = self.layer_4_final(inp_q3_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

        tower_q4_x1 = self.layer_4_final(inp_q4_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q4_x2 = self.layer_4_final(inp_q4_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q4_x3 = self.layer_4_final(inp_q4_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q5_x1 = self.layer_4_final(inp_q5_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q5_x2 = self.layer_4_final(inp_q5_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q5_x3 = self.layer_4_final(inp_q5_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

        tower_q6_x1 = self.layer_4_final(inp_q6_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q6_x2 = self.layer_4_final(inp_q6_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q6_x3 = self.layer_4_final(inp_q6_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q7_x1 = self.layer_4_final(inp_q7_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q7_x2 = self.layer_4_final(inp_q7_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q7_x3 = self.layer_4_final(inp_q7_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
                      
        merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3,tower_q1_x1,tower_q1_x2,tower_q1_x3,tower_q2_x1, tower_q2_x2, tower_q2_x3,tower_q3_x1, tower_q3_x2, tower_q3_x3,tower_q4_x1, tower_q4_x2, tower_q4_x3,tower_q5_x1, tower_q5_x2, tower_q5_x3,tower_q6_x1, tower_q6_x2, tower_q6_x3,tower_q7_x1, tower_q7_x2, tower_q7_x3], axis=1)

        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation=self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation=self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        if self.reducer_7x1_selector==0:
            model_name = ''.join(['model_brain_incept_RESIDUAL_quat_with_k3_to_7_with_only_7x1_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        elif self.reducer_7x1_selector== 1:
            model_name = ''.join(['model_brain_incept_RESIDUAL_quat_with_k3_to_7_with_7x1_and_3x3_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        else:
            model_name = ''.join(['model_brain_incept_RESIDUAL_quat_with_k3_to_7_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        print("")
        print("")
        print(model_name)
        print("")
        print("")
        model = keras.models.Model(inputs=[inp_q0_x1,inp_q0_x2,inp_q0_x3 ,inp_q1_x1,inp_q1_x2,inp_q1_x3,inp_q2_x1,inp_q2_x2,inp_q2_x3,inp_q3_x1,inp_q3_x2,inp_q3_x3,inp_q4_x1,inp_q4_x2,inp_q4_x3,inp_q5_x1,inp_q5_x2,inp_q5_x3,inp_q6_x1,inp_q6_x2,inp_q6_x3,inp_q7_x1,inp_q7_x2,inp_q7_x3], outputs=outputs, name=model_name)
        
        return model,model_name    

class model_par_brain_inception_only_vin_1:

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu'):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.square_height=128
        self.square_width=128
        self.tower_min_max_only = tower_min_max_only

    def layer_1_reducer(self,lay_1_all_output,d_lay_1_to_2,m_p=4):
        
        layer_1_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_1_all_output)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        return incept_1_to_3,incept_1_to_5,layer_1_pool

    def layer_2_reducer(self,lay_2_all_output,d_lay_1_to_2,m_p=3):
        
        layer_2_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_2_all_output)       
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        return incept_1_to_3,incept_1_to_5,layer_2_pool

    def layer_3_reducer(self,lay_3_all_output,d_lay_3_to_4,m_p=2):
        
        layer_3_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_3_all_output)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_1_to_5=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        return incept_1_to_3,incept_1_to_5,layer_3_pool

    def layer_4_final(self,lay_4_all_output,d_lay_3_to_4,m_p=2):
        layer_4_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_4_all_output)
        incept_1_to_final =layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_4_pool)
        return incept_1_to_final

    def model_maker(self,f_inc=1,f_d=1):

        d1=128*f_inc
        d3=64*f_inc
        d5=32*f_inc
        d_max=32*f_inc  
        d_lay_1_to_2 = 32*f_d
        d_lay_3_to_4 = 64*f_d
        
        m_p_l1=4#maxpool layer 1 size
        m_p_l2=3
        m_p_l3=2
        m_p_l4=2
        
        channels=self.channels
        
        inputs = keras.Input(shape=(self.square_height, self.square_width, self.channels))
        '''Inputs intialisation'''
        inp_q0_x1 = keras.Input(shape=(self.square_height,self.square_width, channels))
        inp_q0_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q0_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))   
        
        inp_q1_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
           		
        inp_q2_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q3_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q4_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        inp_q5_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q6_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q7_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x3 = keras.Input(shape=(self.square_height, self.square_width, channels)) 
        
        lay_1_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
        lay_1_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_1_incept_max_pool)
        
        lay_1_all_output = layers.concatenate([lay_1_incept_1, lay_1_incept_3,lay_1_incept_5,lay_1_incept_max_pool_depth], axis=3)
        '''layer 1 general network'''    
        layer_1_INCEPT_Net = keras.models.Model(inputs, lay_1_all_output, name='layer_1_INCEPT')     
        

        '''place the size of NN layer_2 inputs'''
        #since the stride is 1 ans same padding these equation works
        lay_1_height=int(self.square_height/m_p_l1)
        lay_1_width = int(self.square_width/m_p_l1)
        
        
        lay_1_inp_incept_1_to_3 = keras.Input(shape=(lay_1_height,lay_1_width, d_lay_1_to_2))
        lay_1_inp_incept_1_to_5 = keras.Input(shape=(lay_1_height,lay_1_width, d_lay_1_to_2))
        inp_lay_1_pool = keras.Input(shape=(lay_1_height,lay_1_width,d1+d3+d5+d_max))
        
        lay_2_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_1_pool)
        lay_2_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_1_inp_incept_1_to_3)
        lay_2_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_1_inp_incept_1_to_5)
        lay_2_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_1_pool)
        lay_2_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_2_incept_max_pool)
        
        lay_2_all_output = layers.concatenate([lay_2_incept_1, lay_2_incept_3,lay_2_incept_5,lay_2_incept_max_pool_depth], axis=3)
        '''layer 2 general network'''    
        layer_2_INCEPT_Net = keras.models.Model(inputs=[lay_1_inp_incept_1_to_3,lay_1_inp_incept_1_to_5,inp_lay_1_pool],outputs= lay_2_all_output, name='layer_2_INCEPT')     

        '''place the size of NN layer_3 inputs'''
        #since the stride is 1 ans same padding these equation works
        lay_2_height=int(lay_1_height/m_p_l2)
        lay_2_width = int(lay_1_height/m_p_l2)
        
        lay_2_inp_incept_1_to_3 = keras.Input(shape=(lay_2_height,lay_1_width, d_lay_1_to_2))
        lay_2_inp_incept_1_to_5 = keras.Input(shape=(lay_2_width,lay_1_width, d_lay_1_to_2))      
        inp_lay_2_pool = keras.Input(shape=(lay_2_height,lay_1_width,d1+d3+d5+d_max))

        lay_3_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_2_pool)
        lay_3_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_2_inp_incept_1_to_3)
        lay_3_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_2_inp_incept_1_to_5)
        lay_3_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_2_pool)
        lay_3_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_3_incept_max_pool)
        
        lay_3_all_output = layers.concatenate([lay_3_incept_1, lay_3_incept_3,lay_3_incept_5,lay_3_incept_max_pool_depth], axis=3)
        '''layer 3 general network'''    
        layer_3_INCEPT_Net = keras.models.Model(inputs=[lay_2_inp_incept_1_to_3,lay_2_inp_incept_1_to_5,inp_lay_2_pool],outputs= lay_3_all_output, name='layer_3_INCEPT')     

        '''place the size of NN layer_4 inputs'''
        #since the stride is 1 ans same padding these equation works
        lay_3_height=int(lay_2_height/m_p_l3)
        lay_3_width = int(lay_2_height/m_p_l3)
        
        lay_3_inp_incept_1_to_3 = keras.Input(shape=(lay_3_height, lay_3_width,d_lay_3_to_4))
        lay_3_inp_incept_1_to_5 = keras.Input(shape=(lay_3_height, lay_3_width,d_lay_3_to_4))
        inp_lay_3_pool = keras.Input(shape=(lay_3_height, lay_3_width,d1+d3+d5+d_max))       
        
        lay_4_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_3_pool)
        lay_4_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_3_inp_incept_1_to_3)
        lay_4_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_3_inp_incept_1_to_5)
        lay_4_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_3_pool)
        lay_4_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_4_incept_max_pool)
        
        lay_4_all_output = layers.concatenate([lay_4_incept_1, lay_4_incept_3,lay_4_incept_5,lay_4_incept_max_pool_depth], axis=3)
    
        '''layer 3 general network'''    
        layer_4_INCEPT_Net = keras.models.Model(inputs=[lay_3_inp_incept_1_to_3,lay_3_inp_incept_1_to_5,inp_lay_3_pool], outputs=lay_4_all_output, name='layer_4_INCEPT')     
        
        '''Applying layer_1 in projections'''
        #layer1 output of projection 1
        inp_q0_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q0_x1) 
        inp_q0_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q0_x2) 
        inp_q0_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q0_x3) 
        
        inp_q1_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q1_x1) 
        inp_q1_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q1_x2) 
        inp_q1_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q1_x3) 

        inp_q2_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q2_x1) 
        inp_q2_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q2_x2) 
        inp_q2_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q2_x3) 

        inp_q3_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q3_x1) 
        inp_q3_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q3_x2) 
        inp_q3_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q3_x3)     
        
        inp_q4_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q4_x1) 
        inp_q4_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q4_x2) 
        inp_q4_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q4_x3) 
        
        inp_q5_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q5_x1) 
        inp_q5_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q5_x2) 
        inp_q5_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q5_x3) 

        inp_q6_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q6_x1) 
        inp_q6_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q6_x2) 
        inp_q6_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q6_x3) 

        inp_q7_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q7_x1) 
        inp_q7_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q7_x2) 
        inp_q7_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q7_x3) 
        
        lay_1_inp_q0_x1_incept_1_to_3,lay_1_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_1_pool = self.layer_1_reducer(inp_q0_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q0_x2_incept_1_to_3,lay_1_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_1_pool = self.layer_1_reducer(inp_q0_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q0_x3_incept_1_to_3,lay_1_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_1_pool = self.layer_1_reducer(inp_q0_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
       
        lay_1_inp_q1_x1_incept_1_to_3,lay_1_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_1_pool = self.layer_1_reducer(inp_q1_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q1_x2_incept_1_to_3,lay_1_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_1_pool = self.layer_1_reducer(inp_q1_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q1_x3_incept_1_to_3,lay_1_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_1_pool = self.layer_1_reducer(inp_q1_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)

        lay_1_inp_q2_x1_incept_1_to_3,lay_1_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_1_pool = self.layer_1_reducer(inp_q2_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q2_x2_incept_1_to_3,lay_1_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_1_pool = self.layer_1_reducer(inp_q2_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q2_x3_incept_1_to_3,lay_1_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_1_pool = self.layer_1_reducer(inp_q2_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)

        lay_1_inp_q3_x1_incept_1_to_3,lay_1_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_1_pool = self.layer_1_reducer(inp_q3_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q3_x2_incept_1_to_3,lay_1_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_1_pool = self.layer_1_reducer(inp_q3_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q3_x3_incept_1_to_3,lay_1_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_1_pool = self.layer_1_reducer(inp_q3_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)              
        
        lay_1_inp_q4_x1_incept_1_to_3,lay_1_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_1_pool = self.layer_1_reducer(inp_q4_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q4_x2_incept_1_to_3,lay_1_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_1_pool = self.layer_1_reducer(inp_q4_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q4_x3_incept_1_to_3,lay_1_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_1_pool = self.layer_1_reducer(inp_q4_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
       
        lay_1_inp_q5_x1_incept_1_to_3,lay_1_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_1_pool = self.layer_1_reducer(inp_q5_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q5_x2_incept_1_to_3,lay_1_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_1_pool = self.layer_1_reducer(inp_q5_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q5_x3_incept_1_to_3,lay_1_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_1_pool = self.layer_1_reducer(inp_q5_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)

        lay_1_inp_q6_x1_incept_1_to_3,lay_1_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_1_pool = self.layer_1_reducer(inp_q6_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q6_x2_incept_1_to_3,lay_1_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_1_pool = self.layer_1_reducer(inp_q6_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q6_x3_incept_1_to_3,lay_1_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_1_pool = self.layer_1_reducer(inp_q6_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)

        lay_1_inp_q7_x1_incept_1_to_3,lay_1_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_1_pool = self.layer_1_reducer(inp_q7_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q7_x2_incept_1_to_3,lay_1_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_1_pool = self.layer_1_reducer(inp_q7_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q7_x3_incept_1_to_3,lay_1_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_1_pool = self.layer_1_reducer(inp_q7_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)        
        '''Applying layer_2 in projections'''
        #layer1 output of projection 1
        inp_q0_x_1_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q0_x1_incept_1_to_3,lay_1_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_1_pool]) 
        inp_q0_x_2_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q0_x2_incept_1_to_3,lay_1_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_1_pool]) 
        inp_q0_x_3_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q0_x3_incept_1_to_3,lay_1_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_1_pool]) 

        inp_q1_x_1_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q1_x1_incept_1_to_3,lay_1_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_1_pool]) 
        inp_q1_x_2_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q1_x2_incept_1_to_3,lay_1_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_1_pool]) 
        inp_q1_x_3_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q1_x3_incept_1_to_3,lay_1_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_1_pool]) 
        
        inp_q2_x_1_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q2_x1_incept_1_to_3,lay_1_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_1_pool]) 
        inp_q2_x_2_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q2_x2_incept_1_to_3,lay_1_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_1_pool]) 
        inp_q2_x_3_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q2_x3_incept_1_to_3,lay_1_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_1_pool]) 

        inp_q3_x_1_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q3_x1_incept_1_to_3,lay_1_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_1_pool]) 
        inp_q3_x_2_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q3_x2_incept_1_to_3,lay_1_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_1_pool]) 
        inp_q3_x_3_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q3_x3_incept_1_to_3,lay_1_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_1_pool])    

        inp_q4_x_1_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q4_x1_incept_1_to_3,lay_1_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_1_pool]) 
        inp_q4_x_2_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q4_x2_incept_1_to_3,lay_1_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_1_pool]) 
        inp_q4_x_3_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q4_x3_incept_1_to_3,lay_1_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_1_pool]) 

        inp_q5_x_1_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q5_x1_incept_1_to_3,lay_1_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_1_pool]) 
        inp_q5_x_2_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q5_x2_incept_1_to_3,lay_1_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_1_pool]) 
        inp_q5_x_3_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q5_x3_incept_1_to_3,lay_1_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_1_pool]) 
        
        inp_q6_x_1_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q6_x1_incept_1_to_3,lay_1_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_1_pool]) 
        inp_q6_x_2_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q6_x2_incept_1_to_3,lay_1_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_1_pool]) 
        inp_q6_x_3_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q6_x3_incept_1_to_3,lay_1_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_1_pool]) 

        inp_q7_x_1_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q7_x1_incept_1_to_3,lay_1_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_1_pool]) 
        inp_q7_x_2_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q7_x2_incept_1_to_3,lay_1_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_1_pool]) 
        inp_q7_x_3_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp_q7_x3_incept_1_to_3,lay_1_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_1_pool])    

        #layer2 output of projection 1
        lay_2_inp_q0_x1_incept_1_to_3,lay_2_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_2_pool = self.layer_2_reducer(inp_q0_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q0_x2_incept_1_to_3,lay_2_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_2_pool = self.layer_2_reducer(inp_q0_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q0_x3_incept_1_to_3,lay_2_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_2_pool = self.layer_2_reducer(inp_q0_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q1_x1_incept_1_to_3,lay_2_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_2_pool = self.layer_2_reducer(inp_q1_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q1_x2_incept_1_to_3,lay_2_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_2_pool = self.layer_2_reducer(inp_q1_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q1_x3_incept_1_to_3,lay_2_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_2_pool = self.layer_2_reducer(inp_q1_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

        lay_2_inp_q2_x1_incept_1_to_3,lay_2_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_2_pool = self.layer_2_reducer(inp_q2_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q2_x2_incept_1_to_3,lay_2_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_2_pool = self.layer_2_reducer(inp_q2_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q2_x3_incept_1_to_3,lay_2_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_2_pool = self.layer_2_reducer(inp_q2_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q3_x1_incept_1_to_3,lay_2_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_2_pool = self.layer_2_reducer(inp_q3_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q3_x2_incept_1_to_3,lay_2_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_2_pool = self.layer_2_reducer(inp_q3_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q3_x3_incept_1_to_3,lay_2_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_2_pool = self.layer_2_reducer(inp_q3_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

        lay_2_inp_q4_x1_incept_1_to_3,lay_2_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_2_pool = self.layer_2_reducer(inp_q4_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q4_x2_incept_1_to_3,lay_2_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_2_pool = self.layer_2_reducer(inp_q4_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q4_x3_incept_1_to_3,lay_2_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_2_pool = self.layer_2_reducer(inp_q4_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q5_x1_incept_1_to_3,lay_2_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_2_pool = self.layer_2_reducer(inp_q5_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q5_x2_incept_1_to_3,lay_2_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_2_pool = self.layer_2_reducer(inp_q5_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q5_x3_incept_1_to_3,lay_2_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_2_pool = self.layer_2_reducer(inp_q5_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

        lay_2_inp_q6_x1_incept_1_to_3,lay_2_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_2_pool = self.layer_2_reducer(inp_q6_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q6_x2_incept_1_to_3,lay_2_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_2_pool = self.layer_2_reducer(inp_q6_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q6_x3_incept_1_to_3,lay_2_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_2_pool = self.layer_2_reducer(inp_q6_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q7_x1_incept_1_to_3,lay_2_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_2_pool = self.layer_2_reducer(inp_q7_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q7_x2_incept_1_to_3,lay_2_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_2_pool = self.layer_2_reducer(inp_q7_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q7_x3_incept_1_to_3,lay_2_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_2_pool = self.layer_2_reducer(inp_q7_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

        '''Applying layer_3 in projections'''
        #layer3 output of projection 1
        inp_q0_x_1_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q0_x1_incept_1_to_3,lay_2_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_2_pool]) 
        inp_q0_x_2_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q0_x2_incept_1_to_3,lay_2_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_2_pool]) 
        inp_q0_x_3_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q0_x3_incept_1_to_3,lay_2_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_2_pool]) 
      
        inp_q1_x_1_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q1_x1_incept_1_to_3,lay_2_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_2_pool]) 
        inp_q1_x_2_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q1_x2_incept_1_to_3,lay_2_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_2_pool]) 
        inp_q1_x_3_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q1_x3_incept_1_to_3,lay_2_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_2_pool]) 
        
        inp_q2_x_1_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q2_x1_incept_1_to_3,lay_2_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_2_pool]) 
        inp_q2_x_2_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q2_x2_incept_1_to_3,lay_2_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_2_pool]) 
        inp_q2_x_3_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q2_x3_incept_1_to_3,lay_2_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_2_pool]) 
      
        inp_q3_x_1_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q3_x1_incept_1_to_3,lay_2_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_2_pool]) 
        inp_q3_x_2_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q3_x2_incept_1_to_3,lay_2_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_2_pool]) 
        inp_q3_x_3_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q3_x3_incept_1_to_3,lay_2_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_2_pool]) 

        inp_q4_x_1_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q4_x1_incept_1_to_3,lay_2_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_2_pool]) 
        inp_q4_x_2_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q4_x2_incept_1_to_3,lay_2_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_2_pool]) 
        inp_q4_x_3_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q4_x3_incept_1_to_3,lay_2_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_2_pool]) 
      
        inp_q5_x_1_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q5_x1_incept_1_to_3,lay_2_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_2_pool]) 
        inp_q5_x_2_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q5_x2_incept_1_to_3,lay_2_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_2_pool]) 
        inp_q5_x_3_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q5_x3_incept_1_to_3,lay_2_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_2_pool]) 
        
        inp_q6_x_1_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q6_x1_incept_1_to_3,lay_2_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_2_pool]) 
        inp_q6_x_2_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q6_x2_incept_1_to_3,lay_2_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_2_pool]) 
        inp_q6_x_3_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q6_x3_incept_1_to_3,lay_2_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_2_pool]) 
      
        inp_q7_x_1_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q7_x1_incept_1_to_3,lay_2_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_2_pool]) 
        inp_q7_x_2_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q7_x2_incept_1_to_3,lay_2_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_2_pool]) 
        inp_q7_x_3_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp_q7_x3_incept_1_to_3,lay_2_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_2_pool]) 
        
        #layer3 output of projection 1
        lay_3_inp_q0_x1_incept_1_to_3,lay_3_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_3_pool = self.layer_3_reducer(inp_q0_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q0_x2_incept_1_to_3,lay_3_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_3_pool = self.layer_3_reducer(inp_q0_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q0_x3_incept_1_to_3,lay_3_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_3_pool = self.layer_3_reducer(inp_q0_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q1_x1_incept_1_to_3,lay_3_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_3_pool = self.layer_3_reducer(inp_q1_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q1_x2_incept_1_to_3,lay_3_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_3_pool = self.layer_3_reducer(inp_q1_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q1_x3_incept_1_to_3,lay_3_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_3_pool = self.layer_3_reducer(inp_q1_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
       
        lay_3_inp_q2_x1_incept_1_to_3,lay_3_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_3_pool = self.layer_3_reducer(inp_q2_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q2_x2_incept_1_to_3,lay_3_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_3_pool = self.layer_3_reducer(inp_q2_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q2_x3_incept_1_to_3,lay_3_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_3_pool = self.layer_3_reducer(inp_q2_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q3_x1_incept_1_to_3,lay_3_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_3_pool = self.layer_3_reducer(inp_q3_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q3_x2_incept_1_to_3,lay_3_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_3_pool = self.layer_3_reducer(inp_q3_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q3_x3_incept_1_to_3,lay_3_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_3_pool = self.layer_3_reducer(inp_q3_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)

        lay_3_inp_q4_x1_incept_1_to_3,lay_3_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_3_pool = self.layer_3_reducer(inp_q4_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q4_x2_incept_1_to_3,lay_3_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_3_pool = self.layer_3_reducer(inp_q4_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q4_x3_incept_1_to_3,lay_3_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_3_pool = self.layer_3_reducer(inp_q4_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q5_x1_incept_1_to_3,lay_3_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_3_pool = self.layer_3_reducer(inp_q5_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q5_x2_incept_1_to_3,lay_3_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_3_pool = self.layer_3_reducer(inp_q5_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q5_x3_incept_1_to_3,lay_3_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_3_pool = self.layer_3_reducer(inp_q5_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
       
        lay_3_inp_q6_x1_incept_1_to_3,lay_3_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_3_pool = self.layer_3_reducer(inp_q6_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q6_x2_incept_1_to_3,lay_3_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_3_pool = self.layer_3_reducer(inp_q6_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q6_x3_incept_1_to_3,lay_3_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_3_pool = self.layer_3_reducer(inp_q6_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q7_x1_incept_1_to_3,lay_3_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_3_pool = self.layer_3_reducer(inp_q7_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q7_x2_incept_1_to_3,lay_3_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_3_pool = self.layer_3_reducer(inp_q7_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q7_x3_incept_1_to_3,lay_3_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_3_pool = self.layer_3_reducer(inp_q7_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        '''Applying layer_4 in projections'''
        inp_q0_x_1_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q0_x1_incept_1_to_3,lay_3_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_3_pool]) 
        inp_q0_x_2_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q0_x2_incept_1_to_3,lay_3_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_3_pool]) 
        inp_q0_x_3_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q0_x3_incept_1_to_3,lay_3_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_3_pool]) 
        
        inp_q1_x_1_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q1_x1_incept_1_to_3,lay_3_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_3_pool]) 
        inp_q1_x_2_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q1_x2_incept_1_to_3,lay_3_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_3_pool]) 
        inp_q1_x_3_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q1_x3_incept_1_to_3,lay_3_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_3_pool]) 
       
        inp_q2_x_1_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q2_x1_incept_1_to_3,lay_3_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_3_pool]) 
        inp_q2_x_2_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q2_x2_incept_1_to_3,lay_3_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_3_pool]) 
        inp_q2_x_3_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q2_x3_incept_1_to_3,lay_3_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_3_pool]) 
        
        inp_q3_x_1_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q3_x1_incept_1_to_3,lay_3_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_3_pool]) 
        inp_q3_x_2_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q3_x2_incept_1_to_3,lay_3_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_3_pool]) 
        inp_q3_x_3_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q3_x3_incept_1_to_3,lay_3_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_3_pool]) 

        inp_q4_x_1_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q4_x1_incept_1_to_3,lay_3_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_3_pool]) 
        inp_q4_x_2_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q4_x2_incept_1_to_3,lay_3_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_3_pool]) 
        inp_q4_x_3_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q4_x3_incept_1_to_3,lay_3_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_3_pool]) 
        
        inp_q5_x_1_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q5_x1_incept_1_to_3,lay_3_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_3_pool]) 
        inp_q5_x_2_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q5_x2_incept_1_to_3,lay_3_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_3_pool]) 
        inp_q5_x_3_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q5_x3_incept_1_to_3,lay_3_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_3_pool]) 
       
        inp_q6_x_1_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q6_x1_incept_1_to_3,lay_3_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_3_pool]) 
        inp_q6_x_2_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q6_x2_incept_1_to_3,lay_3_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_3_pool]) 
        inp_q6_x_3_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q6_x3_incept_1_to_3,lay_3_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_3_pool]) 
        
        inp_q7_x_1_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q7_x1_incept_1_to_3,lay_3_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_3_pool]) 
        inp_q7_x_2_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q7_x2_incept_1_to_3,lay_3_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_3_pool]) 
        inp_q7_x_3_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp_q7_x3_incept_1_to_3,lay_3_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_3_pool]) 

        #parallel = keras.models.Model(inputs,all_output, name='parallel') 
        tower_q0_x1 = self.layer_4_final(inp_q0_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q0_x2 = self.layer_4_final(inp_q0_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q0_x3 = self.layer_4_final(inp_q0_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q1_x1 = self.layer_4_final(inp_q1_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q1_x2 = self.layer_4_final(inp_q1_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q1_x3 = self.layer_4_final(inp_q1_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

        tower_q2_x1 = self.layer_4_final(inp_q2_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q2_x2 = self.layer_4_final(inp_q2_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q2_x3 = self.layer_4_final(inp_q2_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q3_x1 = self.layer_4_final(inp_q3_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q3_x2 = self.layer_4_final(inp_q3_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q3_x3 = self.layer_4_final(inp_q3_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

        tower_q4_x1 = self.layer_4_final(inp_q4_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q4_x2 = self.layer_4_final(inp_q4_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q4_x3 = self.layer_4_final(inp_q4_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q5_x1 = self.layer_4_final(inp_q5_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q5_x2 = self.layer_4_final(inp_q5_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q5_x3 = self.layer_4_final(inp_q5_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

        tower_q6_x1 = self.layer_4_final(inp_q6_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q6_x2 = self.layer_4_final(inp_q6_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q6_x3 = self.layer_4_final(inp_q6_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q7_x1 = self.layer_4_final(inp_q7_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q7_x2 = self.layer_4_final(inp_q7_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q7_x3 = self.layer_4_final(inp_q7_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
                      
        merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3,tower_q1_x1,tower_q1_x2,tower_q1_x3,tower_q2_x1, tower_q2_x2, tower_q2_x3,tower_q3_x1, tower_q3_x2, tower_q3_x3,tower_q4_x1, tower_q4_x2, tower_q4_x3,tower_q5_x1, tower_q5_x2, tower_q5_x3,tower_q6_x1, tower_q6_x2, tower_q6_x3,tower_q7_x1, tower_q7_x2, tower_q7_x3], axis=1)

        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation=self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation=self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        
        model_name = ''.join(['model_brain_inception_only_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        model = keras.models.Model(inputs=[inp_q0_x1,inp_q0_x2,inp_q0_x3 ,inp_q1_x1,inp_q1_x2,inp_q1_x3,inp_q2_x1,inp_q2_x2,inp_q2_x3,inp_q3_x1,inp_q3_x2,inp_q3_x3,inp_q4_x1,inp_q4_x2,inp_q4_x3,inp_q5_x1,inp_q5_x2,inp_q5_x3,inp_q6_x1,inp_q6_x2,inp_q6_x3,inp_q7_x1,inp_q7_x2,inp_q7_x3], outputs=outputs, name=model_name)
        
        return model,model_name    

class model_brain_incept_RESIDUAL_quat_vin_1(model_par_brain_inception_only_vin_1):

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu'):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.square_height=128
        self.square_width=128
        self.tower_min_max_only = tower_min_max_only

    def model_maker(self,f_inc=1,f_d=1):

        d1=128*f_inc
        d3=64*f_inc
        d5=32*f_inc
        d_max=32*f_inc  
        d_lay_1_to_2 = 32*f_d
        d_lay_3_to_4 = 64*f_d
        
        m_p_l1=4#maxpool layer 1 size
        m_p_l2=3
        m_p_l3=2
        m_p_l4=2
        
        channels=self.channels
        
        inputs = keras.Input(shape=(self.square_height, self.square_width, self.channels))
        '''Inputs intialisation'''
        inp_q0_x1 = keras.Input(shape=(self.square_height,self.square_width, channels))
        inp_q0_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q0_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))   
        
        inp_q1_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
           		
        inp_q2_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q3_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q4_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        inp_q5_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q6_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q7_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x3 = keras.Input(shape=(self.square_height, self.square_width, channels)) 
        
        lay_1_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
        lay_1_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_1_incept_max_pool)
        
        lay_1_all_output = layers.concatenate([lay_1_incept_1, lay_1_incept_3,lay_1_incept_5,lay_1_incept_max_pool_depth], axis=3)
        '''layer 1 general network'''    
        layer_1_INCEPT_Net = keras.models.Model(inputs, lay_1_all_output, name='layer_1_INCEPT')     
        

        '''place the size of NN layer_2 inputs'''
        #since the stride is 1 ans same padding these equation works
        lay_1_height=int(self.square_height/m_p_l1)
        lay_1_width = int(self.square_width/m_p_l1)
        
        
        lay_1_inp_incept_1_to_3 = keras.Input(shape=(lay_1_height,lay_1_width, d_lay_1_to_2))
        lay_1_inp_incept_1_to_5 = keras.Input(shape=(lay_1_height,lay_1_width, d_lay_1_to_2))
        inp_lay_1_pool = keras.Input(shape=(lay_1_height,lay_1_width,d1+d3+d5+d_max))
        
        lay_2_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_1_pool)
        lay_2_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_1_inp_incept_1_to_3)
        lay_2_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_1_inp_incept_1_to_5)
        lay_2_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_1_pool)
        lay_2_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_2_incept_max_pool)
        
        lay_2_all_output = layers.concatenate([lay_2_incept_1, lay_2_incept_3,lay_2_incept_5,lay_2_incept_max_pool_depth], axis=3)
        '''layer 2 general network'''    
        layer_2_INCEPT_Net = keras.models.Model(inputs=[lay_1_inp_incept_1_to_3,lay_1_inp_incept_1_to_5,inp_lay_1_pool],outputs= lay_2_all_output, name='layer_2_INCEPT')     

        '''place the size of NN layer_3 inputs'''
        #since the stride is 1 ans same padding these equation works
        lay_2_height=int(lay_1_height/m_p_l2)
        lay_2_width = int(lay_1_height/m_p_l2)
        
        lay_2_inp_incept_1_to_3 = keras.Input(shape=(lay_2_height,lay_1_width, d_lay_1_to_2))
        lay_2_inp_incept_1_to_5 = keras.Input(shape=(lay_2_width,lay_1_width, d_lay_1_to_2))      
        inp_lay_2_pool = keras.Input(shape=(lay_2_height,lay_1_width,d1+d3+d5+d_max))

        lay_3_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_2_pool)
        lay_3_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_2_inp_incept_1_to_3)
        lay_3_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_2_inp_incept_1_to_5)
        lay_3_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_2_pool)
        lay_3_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_3_incept_max_pool)
        
        lay_3_all_output = layers.concatenate([lay_3_incept_1, lay_3_incept_3,lay_3_incept_5,lay_3_incept_max_pool_depth], axis=3)
        '''layer 3 general network'''    
        layer_3_INCEPT_Net = keras.models.Model(inputs=[lay_2_inp_incept_1_to_3,lay_2_inp_incept_1_to_5,inp_lay_2_pool],outputs= lay_3_all_output, name='layer_3_INCEPT')     

        '''place the size of NN layer_4 inputs'''
        #since the stride is 1 ans same padding these equation works
        lay_3_height=int(lay_2_height/m_p_l3)
        lay_3_width = int(lay_2_height/m_p_l3)
        
        lay_3_inp_incept_1_to_3 = keras.Input(shape=(lay_3_height, lay_3_width,d_lay_3_to_4))
        lay_3_inp_incept_1_to_5 = keras.Input(shape=(lay_3_height, lay_3_width,d_lay_3_to_4))
        inp_lay_3_pool = keras.Input(shape=(lay_3_height, lay_3_width,d1+d3+d5+d_max))       
        
        lay_4_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_3_pool)
        lay_4_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_3_inp_incept_1_to_3)
        lay_4_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_3_inp_incept_1_to_5)
        lay_4_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_3_pool)
        lay_4_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_4_incept_max_pool)
        
        lay_4_all_output = layers.concatenate([lay_4_incept_1, lay_4_incept_3,lay_4_incept_5,lay_4_incept_max_pool_depth], axis=3)
    
        '''layer 3 general network'''    
        layer_4_INCEPT_Net = keras.models.Model(inputs=[lay_3_inp_incept_1_to_3,lay_3_inp_incept_1_to_5,inp_lay_3_pool], outputs=lay_4_all_output, name='layer_4_INCEPT')     
        
        '''Applying layer_1 in projections'''
        #layer1 output of projection 1
        inp_q0_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q0_x1) 
        inp_q0_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q0_x2) 
        inp_q0_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q0_x3) 
        
        inp_q1_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q1_x1) 
        inp_q1_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q1_x2) 
        inp_q1_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q1_x3) 

        inp_q2_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q2_x1) 
        inp_q2_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q2_x2) 
        inp_q2_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q2_x3) 

        inp_q3_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q3_x1) 
        inp_q3_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q3_x2) 
        inp_q3_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q3_x3)     
        
        inp_q4_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q4_x1) 
        inp_q4_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q4_x2) 
        inp_q4_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q4_x3) 
        
        inp_q5_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q5_x1) 
        inp_q5_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q5_x2) 
        inp_q5_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q5_x3) 

        inp_q6_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q6_x1) 
        inp_q6_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q6_x2) 
        inp_q6_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q6_x3) 

        inp_q7_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q7_x1) 
        inp_q7_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q7_x2) 
        inp_q7_x3_lay_1_all_output = layer_1_INCEPT_Net(inp_q7_x3) 
        
        lay_1_inp_q0_x1_incept_1_to_3,lay_1_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_1_pool = self.layer_1_reducer(inp_q0_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q0_x2_incept_1_to_3,lay_1_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_1_pool = self.layer_1_reducer(inp_q0_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q0_x3_incept_1_to_3,lay_1_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_1_pool = self.layer_1_reducer(inp_q0_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
       
        lay_1_inp_q1_x1_incept_1_to_3,lay_1_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_1_pool = self.layer_1_reducer(inp_q1_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q1_x2_incept_1_to_3,lay_1_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_1_pool = self.layer_1_reducer(inp_q1_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q1_x3_incept_1_to_3,lay_1_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_1_pool = self.layer_1_reducer(inp_q1_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)

        lay_1_inp_q2_x1_incept_1_to_3,lay_1_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_1_pool = self.layer_1_reducer(inp_q2_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q2_x2_incept_1_to_3,lay_1_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_1_pool = self.layer_1_reducer(inp_q2_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q2_x3_incept_1_to_3,lay_1_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_1_pool = self.layer_1_reducer(inp_q2_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)

        lay_1_inp_q3_x1_incept_1_to_3,lay_1_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_1_pool = self.layer_1_reducer(inp_q3_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q3_x2_incept_1_to_3,lay_1_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_1_pool = self.layer_1_reducer(inp_q3_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q3_x3_incept_1_to_3,lay_1_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_1_pool = self.layer_1_reducer(inp_q3_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)              
        
        lay_1_inp_q4_x1_incept_1_to_3,lay_1_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_1_pool = self.layer_1_reducer(inp_q4_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q4_x2_incept_1_to_3,lay_1_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_1_pool = self.layer_1_reducer(inp_q4_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q4_x3_incept_1_to_3,lay_1_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_1_pool = self.layer_1_reducer(inp_q4_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
       
        lay_1_inp_q5_x1_incept_1_to_3,lay_1_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_1_pool = self.layer_1_reducer(inp_q5_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q5_x2_incept_1_to_3,lay_1_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_1_pool = self.layer_1_reducer(inp_q5_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q5_x3_incept_1_to_3,lay_1_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_1_pool = self.layer_1_reducer(inp_q5_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)

        lay_1_inp_q6_x1_incept_1_to_3,lay_1_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_1_pool = self.layer_1_reducer(inp_q6_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q6_x2_incept_1_to_3,lay_1_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_1_pool = self.layer_1_reducer(inp_q6_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q6_x3_incept_1_to_3,lay_1_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_1_pool = self.layer_1_reducer(inp_q6_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)

        lay_1_inp_q7_x1_incept_1_to_3,lay_1_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_1_pool = self.layer_1_reducer(inp_q7_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q7_x2_incept_1_to_3,lay_1_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_1_pool = self.layer_1_reducer(inp_q7_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp_q7_x3_incept_1_to_3,lay_1_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_1_pool = self.layer_1_reducer(inp_q7_x3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)        
        '''Applying layer_2 in projections'''
        #layer1 output of projection 1
        inp_q0_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q0_x1_incept_1_to_3,lay_1_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_1_pool]) 
        inp_q0_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q0_x2_incept_1_to_3,lay_1_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_1_pool]) 
        inp_q0_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q0_x3_incept_1_to_3,lay_1_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_1_pool]) 

        inp_q1_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q1_x1_incept_1_to_3,lay_1_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_1_pool]) 
        inp_q1_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q1_x2_incept_1_to_3,lay_1_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_1_pool]) 
        inp_q1_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q1_x3_incept_1_to_3,lay_1_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_1_pool]) 
        
        inp_q2_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q2_x1_incept_1_to_3,lay_1_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_1_pool]) 
        inp_q2_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q2_x2_incept_1_to_3,lay_1_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_1_pool]) 
        inp_q2_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q2_x3_incept_1_to_3,lay_1_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_1_pool]) 

        inp_q3_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q3_x1_incept_1_to_3,lay_1_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_1_pool]) 
        inp_q3_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q3_x2_incept_1_to_3,lay_1_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_1_pool]) 
        inp_q3_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q3_x3_incept_1_to_3,lay_1_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_1_pool])    

        inp_q4_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q4_x1_incept_1_to_3,lay_1_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_1_pool]) 
        inp_q4_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q4_x2_incept_1_to_3,lay_1_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_1_pool]) 
        inp_q4_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q4_x3_incept_1_to_3,lay_1_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_1_pool]) 

        inp_q5_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q5_x1_incept_1_to_3,lay_1_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_1_pool]) 
        inp_q5_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q5_x2_incept_1_to_3,lay_1_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_1_pool]) 
        inp_q5_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q5_x3_incept_1_to_3,lay_1_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_1_pool]) 
        
        inp_q6_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q6_x1_incept_1_to_3,lay_1_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_1_pool]) 
        inp_q6_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q6_x2_incept_1_to_3,lay_1_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_1_pool]) 
        inp_q6_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q6_x3_incept_1_to_3,lay_1_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_1_pool]) 

        inp_q7_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q7_x1_incept_1_to_3,lay_1_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_1_pool]) 
        inp_q7_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q7_x2_incept_1_to_3,lay_1_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_1_pool]) 
        inp_q7_x_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q7_x3_incept_1_to_3,lay_1_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_1_pool])    

        #make residUAL NETWORK
        inp_q0_x_1_lay_2_all_output= layers.add([inp_q0_x1_lay_1_pool, inp_q0_x_1_lay_2_incept_all_output])
        inp_q0_x_2_lay_2_all_output= layers.add([inp_q0_x2_lay_1_pool, inp_q0_x_2_lay_2_incept_all_output])
        inp_q0_x_3_lay_2_all_output= layers.add([inp_q0_x3_lay_1_pool, inp_q0_x_3_lay_2_incept_all_output])
        
        inp_q1_x_1_lay_2_all_output= layers.add([inp_q1_x1_lay_1_pool, inp_q1_x_1_lay_2_incept_all_output])
        inp_q1_x_2_lay_2_all_output= layers.add([inp_q1_x2_lay_1_pool, inp_q1_x_2_lay_2_incept_all_output])
        inp_q1_x_3_lay_2_all_output= layers.add([inp_q1_x3_lay_1_pool, inp_q1_x_3_lay_2_incept_all_output])
        
        inp_q2_x_1_lay_2_all_output= layers.add([inp_q2_x1_lay_1_pool, inp_q2_x_1_lay_2_incept_all_output])
        inp_q2_x_2_lay_2_all_output= layers.add([inp_q2_x2_lay_1_pool, inp_q2_x_2_lay_2_incept_all_output])
        inp_q2_x_3_lay_2_all_output= layers.add([inp_q2_x3_lay_1_pool, inp_q2_x_3_lay_2_incept_all_output])
        
        inp_q3_x_1_lay_2_all_output= layers.add([inp_q3_x1_lay_1_pool, inp_q3_x_1_lay_2_incept_all_output])
        inp_q3_x_2_lay_2_all_output= layers.add([inp_q3_x2_lay_1_pool, inp_q3_x_2_lay_2_incept_all_output])
        inp_q3_x_3_lay_2_all_output= layers.add([inp_q3_x3_lay_1_pool, inp_q3_x_3_lay_2_incept_all_output])

        inp_q4_x_1_lay_2_all_output= layers.add([inp_q4_x1_lay_1_pool, inp_q4_x_1_lay_2_incept_all_output])
        inp_q4_x_2_lay_2_all_output= layers.add([inp_q4_x2_lay_1_pool, inp_q4_x_2_lay_2_incept_all_output])
        inp_q4_x_3_lay_2_all_output= layers.add([inp_q4_x3_lay_1_pool, inp_q4_x_3_lay_2_incept_all_output])
        
        inp_q5_x_1_lay_2_all_output= layers.add([inp_q5_x1_lay_1_pool, inp_q5_x_1_lay_2_incept_all_output])
        inp_q5_x_2_lay_2_all_output= layers.add([inp_q5_x2_lay_1_pool, inp_q5_x_2_lay_2_incept_all_output])
        inp_q5_x_3_lay_2_all_output= layers.add([inp_q5_x3_lay_1_pool, inp_q5_x_3_lay_2_incept_all_output])
        
        inp_q6_x_1_lay_2_all_output= layers.add([inp_q6_x1_lay_1_pool, inp_q6_x_1_lay_2_incept_all_output])
        inp_q6_x_2_lay_2_all_output= layers.add([inp_q6_x2_lay_1_pool, inp_q6_x_2_lay_2_incept_all_output])
        inp_q6_x_3_lay_2_all_output= layers.add([inp_q6_x3_lay_1_pool, inp_q6_x_3_lay_2_incept_all_output])
        
        inp_q7_x_1_lay_2_all_output= layers.add([inp_q7_x1_lay_1_pool, inp_q7_x_1_lay_2_incept_all_output])
        inp_q7_x_2_lay_2_all_output= layers.add([inp_q7_x2_lay_1_pool, inp_q7_x_2_lay_2_incept_all_output])
        inp_q7_x_3_lay_2_all_output= layers.add([inp_q7_x3_lay_1_pool, inp_q7_x_3_lay_2_incept_all_output])
       
        #layer2 output of projection 1
        lay_2_inp_q0_x1_incept_1_to_3,lay_2_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_2_pool = self.layer_2_reducer(inp_q0_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q0_x2_incept_1_to_3,lay_2_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_2_pool = self.layer_2_reducer(inp_q0_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q0_x3_incept_1_to_3,lay_2_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_2_pool = self.layer_2_reducer(inp_q0_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q1_x1_incept_1_to_3,lay_2_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_2_pool = self.layer_2_reducer(inp_q1_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q1_x2_incept_1_to_3,lay_2_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_2_pool = self.layer_2_reducer(inp_q1_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q1_x3_incept_1_to_3,lay_2_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_2_pool = self.layer_2_reducer(inp_q1_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

        lay_2_inp_q2_x1_incept_1_to_3,lay_2_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_2_pool = self.layer_2_reducer(inp_q2_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q2_x2_incept_1_to_3,lay_2_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_2_pool = self.layer_2_reducer(inp_q2_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q2_x3_incept_1_to_3,lay_2_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_2_pool = self.layer_2_reducer(inp_q2_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q3_x1_incept_1_to_3,lay_2_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_2_pool = self.layer_2_reducer(inp_q3_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q3_x2_incept_1_to_3,lay_2_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_2_pool = self.layer_2_reducer(inp_q3_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q3_x3_incept_1_to_3,lay_2_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_2_pool = self.layer_2_reducer(inp_q3_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

        lay_2_inp_q4_x1_incept_1_to_3,lay_2_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_2_pool = self.layer_2_reducer(inp_q4_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q4_x2_incept_1_to_3,lay_2_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_2_pool = self.layer_2_reducer(inp_q4_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q4_x3_incept_1_to_3,lay_2_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_2_pool = self.layer_2_reducer(inp_q4_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q5_x1_incept_1_to_3,lay_2_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_2_pool = self.layer_2_reducer(inp_q5_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q5_x2_incept_1_to_3,lay_2_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_2_pool = self.layer_2_reducer(inp_q5_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q5_x3_incept_1_to_3,lay_2_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_2_pool = self.layer_2_reducer(inp_q5_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

        lay_2_inp_q6_x1_incept_1_to_3,lay_2_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_2_pool = self.layer_2_reducer(inp_q6_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q6_x2_incept_1_to_3,lay_2_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_2_pool = self.layer_2_reducer(inp_q6_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q6_x3_incept_1_to_3,lay_2_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_2_pool = self.layer_2_reducer(inp_q6_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        
        lay_2_inp_q7_x1_incept_1_to_3,lay_2_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_2_pool = self.layer_2_reducer(inp_q7_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q7_x2_incept_1_to_3,lay_2_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_2_pool = self.layer_2_reducer(inp_q7_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp_q7_x3_incept_1_to_3,lay_2_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_2_pool = self.layer_2_reducer(inp_q7_x_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

        '''Applying layer_3 in projections'''
        #layer3 output of projection 1
        inp_q0_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q0_x1_incept_1_to_3,lay_2_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_2_pool]) 
        inp_q0_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q0_x2_incept_1_to_3,lay_2_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_2_pool]) 
        inp_q0_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q0_x3_incept_1_to_3,lay_2_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_2_pool]) 
      
        inp_q1_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q1_x1_incept_1_to_3,lay_2_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_2_pool]) 
        inp_q1_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q1_x2_incept_1_to_3,lay_2_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_2_pool]) 
        inp_q1_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q1_x3_incept_1_to_3,lay_2_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_2_pool]) 
        
        inp_q2_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q2_x1_incept_1_to_3,lay_2_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_2_pool]) 
        inp_q2_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q2_x2_incept_1_to_3,lay_2_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_2_pool]) 
        inp_q2_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q2_x3_incept_1_to_3,lay_2_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_2_pool]) 
      
        inp_q3_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q3_x1_incept_1_to_3,lay_2_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_2_pool]) 
        inp_q3_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q3_x2_incept_1_to_3,lay_2_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_2_pool]) 
        inp_q3_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q3_x3_incept_1_to_3,lay_2_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_2_pool]) 

        inp_q4_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q4_x1_incept_1_to_3,lay_2_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_2_pool]) 
        inp_q4_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q4_x2_incept_1_to_3,lay_2_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_2_pool]) 
        inp_q4_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q4_x3_incept_1_to_3,lay_2_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_2_pool]) 
      
        inp_q5_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q5_x1_incept_1_to_3,lay_2_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_2_pool]) 
        inp_q5_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q5_x2_incept_1_to_3,lay_2_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_2_pool]) 
        inp_q5_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q5_x3_incept_1_to_3,lay_2_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_2_pool]) 
        
        inp_q6_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q6_x1_incept_1_to_3,lay_2_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_2_pool]) 
        inp_q6_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q6_x2_incept_1_to_3,lay_2_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_2_pool]) 
        inp_q6_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q6_x3_incept_1_to_3,lay_2_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_2_pool]) 
      
        inp_q7_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q7_x1_incept_1_to_3,lay_2_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_2_pool]) 
        inp_q7_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q7_x2_incept_1_to_3,lay_2_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_2_pool]) 
        inp_q7_x_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q7_x3_incept_1_to_3,lay_2_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_2_pool]) 
  
        #make residUAL NETWORK
        inp_q0_x_1_lay_3_all_output= layers.add([inp_q0_x1_lay_2_pool, inp_q0_x_1_lay_3_incept_all_output])
        inp_q0_x_2_lay_3_all_output= layers.add([inp_q0_x2_lay_2_pool, inp_q0_x_2_lay_3_incept_all_output])
        inp_q0_x_3_lay_3_all_output= layers.add([inp_q0_x3_lay_2_pool, inp_q0_x_3_lay_3_incept_all_output])
        
        inp_q1_x_1_lay_3_all_output= layers.add([inp_q1_x1_lay_2_pool, inp_q1_x_1_lay_3_incept_all_output])
        inp_q1_x_2_lay_3_all_output= layers.add([inp_q1_x2_lay_2_pool, inp_q1_x_2_lay_3_incept_all_output])
        inp_q1_x_3_lay_3_all_output= layers.add([inp_q1_x3_lay_2_pool, inp_q1_x_3_lay_3_incept_all_output])
        
        inp_q2_x_1_lay_3_all_output= layers.add([inp_q2_x1_lay_2_pool, inp_q2_x_1_lay_3_incept_all_output])
        inp_q2_x_2_lay_3_all_output= layers.add([inp_q2_x2_lay_2_pool, inp_q2_x_2_lay_3_incept_all_output])
        inp_q2_x_3_lay_3_all_output= layers.add([inp_q2_x3_lay_2_pool, inp_q2_x_3_lay_3_incept_all_output])
        
        inp_q3_x_1_lay_3_all_output= layers.add([inp_q3_x1_lay_2_pool, inp_q3_x_1_lay_3_incept_all_output])
        inp_q3_x_2_lay_3_all_output= layers.add([inp_q3_x2_lay_2_pool, inp_q3_x_2_lay_3_incept_all_output])
        inp_q3_x_3_lay_3_all_output= layers.add([inp_q3_x3_lay_2_pool, inp_q3_x_3_lay_3_incept_all_output])

        inp_q4_x_1_lay_3_all_output= layers.add([inp_q4_x1_lay_2_pool, inp_q4_x_1_lay_3_incept_all_output])
        inp_q4_x_2_lay_3_all_output= layers.add([inp_q4_x2_lay_2_pool, inp_q4_x_2_lay_3_incept_all_output])
        inp_q4_x_3_lay_3_all_output= layers.add([inp_q4_x3_lay_2_pool, inp_q4_x_3_lay_3_incept_all_output])
        
        inp_q5_x_1_lay_3_all_output= layers.add([inp_q5_x1_lay_2_pool, inp_q5_x_1_lay_3_incept_all_output])
        inp_q5_x_2_lay_3_all_output= layers.add([inp_q5_x2_lay_2_pool, inp_q5_x_2_lay_3_incept_all_output])
        inp_q5_x_3_lay_3_all_output= layers.add([inp_q5_x3_lay_2_pool, inp_q5_x_3_lay_3_incept_all_output])
        
        inp_q6_x_1_lay_3_all_output= layers.add([inp_q6_x1_lay_2_pool, inp_q6_x_1_lay_3_incept_all_output])
        inp_q6_x_2_lay_3_all_output= layers.add([inp_q6_x2_lay_2_pool, inp_q6_x_2_lay_3_incept_all_output])
        inp_q6_x_3_lay_3_all_output= layers.add([inp_q6_x3_lay_2_pool, inp_q6_x_3_lay_3_incept_all_output])
        
        inp_q7_x_1_lay_3_all_output= layers.add([inp_q7_x1_lay_2_pool, inp_q7_x_1_lay_3_incept_all_output])
        inp_q7_x_2_lay_3_all_output= layers.add([inp_q7_x2_lay_2_pool, inp_q7_x_2_lay_3_incept_all_output])
        inp_q7_x_3_lay_3_all_output= layers.add([inp_q7_x3_lay_2_pool, inp_q7_x_3_lay_3_incept_all_output])        
        #layer3 output of projection 1
        lay_3_inp_q0_x1_incept_1_to_3,lay_3_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_3_pool = self.layer_3_reducer(inp_q0_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q0_x2_incept_1_to_3,lay_3_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_3_pool = self.layer_3_reducer(inp_q0_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q0_x3_incept_1_to_3,lay_3_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_3_pool = self.layer_3_reducer(inp_q0_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q1_x1_incept_1_to_3,lay_3_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_3_pool = self.layer_3_reducer(inp_q1_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q1_x2_incept_1_to_3,lay_3_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_3_pool = self.layer_3_reducer(inp_q1_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q1_x3_incept_1_to_3,lay_3_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_3_pool = self.layer_3_reducer(inp_q1_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
       
        lay_3_inp_q2_x1_incept_1_to_3,lay_3_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_3_pool = self.layer_3_reducer(inp_q2_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q2_x2_incept_1_to_3,lay_3_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_3_pool = self.layer_3_reducer(inp_q2_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q2_x3_incept_1_to_3,lay_3_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_3_pool = self.layer_3_reducer(inp_q2_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q3_x1_incept_1_to_3,lay_3_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_3_pool = self.layer_3_reducer(inp_q3_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q3_x2_incept_1_to_3,lay_3_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_3_pool = self.layer_3_reducer(inp_q3_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q3_x3_incept_1_to_3,lay_3_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_3_pool = self.layer_3_reducer(inp_q3_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)

        lay_3_inp_q4_x1_incept_1_to_3,lay_3_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_3_pool = self.layer_3_reducer(inp_q4_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q4_x2_incept_1_to_3,lay_3_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_3_pool = self.layer_3_reducer(inp_q4_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q4_x3_incept_1_to_3,lay_3_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_3_pool = self.layer_3_reducer(inp_q4_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q5_x1_incept_1_to_3,lay_3_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_3_pool = self.layer_3_reducer(inp_q5_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q5_x2_incept_1_to_3,lay_3_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_3_pool = self.layer_3_reducer(inp_q5_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q5_x3_incept_1_to_3,lay_3_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_3_pool = self.layer_3_reducer(inp_q5_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
       
        lay_3_inp_q6_x1_incept_1_to_3,lay_3_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_3_pool = self.layer_3_reducer(inp_q6_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q6_x2_incept_1_to_3,lay_3_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_3_pool = self.layer_3_reducer(inp_q6_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q6_x3_incept_1_to_3,lay_3_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_3_pool = self.layer_3_reducer(inp_q6_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        lay_3_inp_q7_x1_incept_1_to_3,lay_3_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_3_pool = self.layer_3_reducer(inp_q7_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q7_x2_incept_1_to_3,lay_3_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_3_pool = self.layer_3_reducer(inp_q7_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp_q7_x3_incept_1_to_3,lay_3_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_3_pool = self.layer_3_reducer(inp_q7_x_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        
        '''Applying layer_4 in projections'''
        inp_q0_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q0_x1_incept_1_to_3,lay_3_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_3_pool]) 
        inp_q0_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q0_x2_incept_1_to_3,lay_3_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_3_pool]) 
        inp_q0_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q0_x3_incept_1_to_3,lay_3_inp_q0_x3_incept_1_to_5,inp_q0_x3_lay_3_pool]) 
        
        inp_q1_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q1_x1_incept_1_to_3,lay_3_inp_q1_x1_incept_1_to_5,inp_q1_x1_lay_3_pool]) 
        inp_q1_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q1_x2_incept_1_to_3,lay_3_inp_q1_x2_incept_1_to_5,inp_q1_x2_lay_3_pool]) 
        inp_q1_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q1_x3_incept_1_to_3,lay_3_inp_q1_x3_incept_1_to_5,inp_q1_x3_lay_3_pool]) 
       
        inp_q2_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q2_x1_incept_1_to_3,lay_3_inp_q2_x1_incept_1_to_5,inp_q2_x1_lay_3_pool]) 
        inp_q2_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q2_x2_incept_1_to_3,lay_3_inp_q2_x2_incept_1_to_5,inp_q2_x2_lay_3_pool]) 
        inp_q2_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q2_x3_incept_1_to_3,lay_3_inp_q2_x3_incept_1_to_5,inp_q2_x3_lay_3_pool]) 
        
        inp_q3_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q3_x1_incept_1_to_3,lay_3_inp_q3_x1_incept_1_to_5,inp_q3_x1_lay_3_pool]) 
        inp_q3_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q3_x2_incept_1_to_3,lay_3_inp_q3_x2_incept_1_to_5,inp_q3_x2_lay_3_pool]) 
        inp_q3_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q3_x3_incept_1_to_3,lay_3_inp_q3_x3_incept_1_to_5,inp_q3_x3_lay_3_pool]) 

        inp_q4_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q4_x1_incept_1_to_3,lay_3_inp_q4_x1_incept_1_to_5,inp_q4_x1_lay_3_pool]) 
        inp_q4_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q4_x2_incept_1_to_3,lay_3_inp_q4_x2_incept_1_to_5,inp_q4_x2_lay_3_pool]) 
        inp_q4_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q4_x3_incept_1_to_3,lay_3_inp_q4_x3_incept_1_to_5,inp_q4_x3_lay_3_pool]) 
        
        inp_q5_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q5_x1_incept_1_to_3,lay_3_inp_q5_x1_incept_1_to_5,inp_q5_x1_lay_3_pool]) 
        inp_q5_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q5_x2_incept_1_to_3,lay_3_inp_q5_x2_incept_1_to_5,inp_q5_x2_lay_3_pool]) 
        inp_q5_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q5_x3_incept_1_to_3,lay_3_inp_q5_x3_incept_1_to_5,inp_q5_x3_lay_3_pool]) 
       
        inp_q6_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q6_x1_incept_1_to_3,lay_3_inp_q6_x1_incept_1_to_5,inp_q6_x1_lay_3_pool]) 
        inp_q6_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q6_x2_incept_1_to_3,lay_3_inp_q6_x2_incept_1_to_5,inp_q6_x2_lay_3_pool]) 
        inp_q6_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q6_x3_incept_1_to_3,lay_3_inp_q6_x3_incept_1_to_5,inp_q6_x3_lay_3_pool]) 
        
        inp_q7_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q7_x1_incept_1_to_3,lay_3_inp_q7_x1_incept_1_to_5,inp_q7_x1_lay_3_pool]) 
        inp_q7_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q7_x2_incept_1_to_3,lay_3_inp_q7_x2_incept_1_to_5,inp_q7_x2_lay_3_pool]) 
        inp_q7_x_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q7_x3_incept_1_to_3,lay_3_inp_q7_x3_incept_1_to_5,inp_q7_x3_lay_3_pool]) 
        
        #make residUAL NETWORK
        inp_q0_x_1_lay_4_all_output= layers.add([inp_q0_x1_lay_3_pool, inp_q0_x_1_lay_4_incept_all_output])
        inp_q0_x_2_lay_4_all_output= layers.add([inp_q0_x2_lay_3_pool, inp_q0_x_2_lay_4_incept_all_output])
        inp_q0_x_3_lay_4_all_output= layers.add([inp_q0_x3_lay_3_pool, inp_q0_x_3_lay_4_incept_all_output])
        
        inp_q1_x_1_lay_4_all_output= layers.add([inp_q1_x1_lay_3_pool, inp_q1_x_1_lay_4_incept_all_output])
        inp_q1_x_2_lay_4_all_output= layers.add([inp_q1_x2_lay_3_pool, inp_q1_x_2_lay_4_incept_all_output])
        inp_q1_x_3_lay_4_all_output= layers.add([inp_q1_x3_lay_3_pool, inp_q1_x_3_lay_4_incept_all_output])
        
        inp_q2_x_1_lay_4_all_output= layers.add([inp_q2_x1_lay_3_pool, inp_q2_x_1_lay_4_incept_all_output])
        inp_q2_x_2_lay_4_all_output= layers.add([inp_q2_x2_lay_3_pool, inp_q2_x_2_lay_4_incept_all_output])
        inp_q2_x_3_lay_4_all_output= layers.add([inp_q2_x3_lay_3_pool, inp_q2_x_3_lay_4_incept_all_output])
        
        inp_q3_x_1_lay_4_all_output= layers.add([inp_q3_x1_lay_3_pool, inp_q3_x_1_lay_4_incept_all_output])
        inp_q3_x_2_lay_4_all_output= layers.add([inp_q3_x2_lay_3_pool, inp_q3_x_2_lay_4_incept_all_output])
        inp_q3_x_3_lay_4_all_output= layers.add([inp_q3_x3_lay_3_pool, inp_q3_x_3_lay_4_incept_all_output])

        inp_q4_x_1_lay_4_all_output= layers.add([inp_q4_x1_lay_3_pool, inp_q4_x_1_lay_4_incept_all_output])
        inp_q4_x_2_lay_4_all_output= layers.add([inp_q4_x2_lay_3_pool, inp_q4_x_2_lay_4_incept_all_output])
        inp_q4_x_3_lay_4_all_output= layers.add([inp_q4_x3_lay_3_pool, inp_q4_x_3_lay_4_incept_all_output])
        
        inp_q5_x_1_lay_4_all_output= layers.add([inp_q5_x1_lay_3_pool, inp_q5_x_1_lay_4_incept_all_output])
        inp_q5_x_2_lay_4_all_output= layers.add([inp_q5_x2_lay_3_pool, inp_q5_x_2_lay_4_incept_all_output])
        inp_q5_x_3_lay_4_all_output= layers.add([inp_q5_x3_lay_3_pool, inp_q5_x_3_lay_4_incept_all_output])
        
        inp_q6_x_1_lay_4_all_output= layers.add([inp_q6_x1_lay_3_pool, inp_q6_x_1_lay_4_incept_all_output])
        inp_q6_x_2_lay_4_all_output= layers.add([inp_q6_x2_lay_3_pool, inp_q6_x_2_lay_4_incept_all_output])
        inp_q6_x_3_lay_4_all_output= layers.add([inp_q6_x3_lay_3_pool, inp_q6_x_3_lay_4_incept_all_output])
        
        inp_q7_x_1_lay_4_all_output= layers.add([inp_q7_x1_lay_3_pool, inp_q7_x_1_lay_4_incept_all_output])
        inp_q7_x_2_lay_4_all_output= layers.add([inp_q7_x2_lay_3_pool, inp_q7_x_2_lay_4_incept_all_output])
        inp_q7_x_3_lay_4_all_output= layers.add([inp_q7_x3_lay_3_pool, inp_q7_x_3_lay_4_incept_all_output])
 
        #parallel = keras.models.Model(inputs,all_output, name='parallel') 
        tower_q0_x1 = self.layer_4_final(inp_q0_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q0_x2 = self.layer_4_final(inp_q0_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q0_x3 = self.layer_4_final(inp_q0_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q1_x1 = self.layer_4_final(inp_q1_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q1_x2 = self.layer_4_final(inp_q1_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q1_x3 = self.layer_4_final(inp_q1_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

        tower_q2_x1 = self.layer_4_final(inp_q2_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q2_x2 = self.layer_4_final(inp_q2_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q2_x3 = self.layer_4_final(inp_q2_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q3_x1 = self.layer_4_final(inp_q3_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q3_x2 = self.layer_4_final(inp_q3_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q3_x3 = self.layer_4_final(inp_q3_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

        tower_q4_x1 = self.layer_4_final(inp_q4_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q4_x2 = self.layer_4_final(inp_q4_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q4_x3 = self.layer_4_final(inp_q4_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q5_x1 = self.layer_4_final(inp_q5_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q5_x2 = self.layer_4_final(inp_q5_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q5_x3 = self.layer_4_final(inp_q5_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

        tower_q6_x1 = self.layer_4_final(inp_q6_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q6_x2 = self.layer_4_final(inp_q6_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q6_x3 = self.layer_4_final(inp_q6_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        
        tower_q7_x1 = self.layer_4_final(inp_q7_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q7_x2 = self.layer_4_final(inp_q7_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_q7_x3 = self.layer_4_final(inp_q7_x_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
                      
        merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3,tower_q1_x1,tower_q1_x2,tower_q1_x3,tower_q2_x1, tower_q2_x2, tower_q2_x3,tower_q3_x1, tower_q3_x2, tower_q3_x3,tower_q4_x1, tower_q4_x2, tower_q4_x3,tower_q5_x1, tower_q5_x2, tower_q5_x3,tower_q6_x1, tower_q6_x2, tower_q6_x3,tower_q7_x1, tower_q7_x2, tower_q7_x3], axis=1)

        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation=self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation=self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        
        model_name = ''.join(['model_brain_inception_Residual_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        model = keras.models.Model(inputs=[inp_q0_x1,inp_q0_x2,inp_q0_x3 ,inp_q1_x1,inp_q1_x2,inp_q1_x3,inp_q2_x1,inp_q2_x2,inp_q2_x3,inp_q3_x1,inp_q3_x2,inp_q3_x3,inp_q4_x1,inp_q4_x2,inp_q4_x3,inp_q5_x1,inp_q5_x2,inp_q5_x3,inp_q6_x1,inp_q6_x2,inp_q6_x3,inp_q7_x1,inp_q7_x2,inp_q7_x3], outputs=outputs, name=model_name)
        
        return model,model_name    



class model_par_inception_only_mentor_quat_vin_1:

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu'):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only
        self.square_height=128
        self.square_width=128
        
    def model_maker(self,f_inc=1,f_d=1):
        
        channels=self.channels
        
        d1=128*f_inc
        d3=64*f_inc
        d5=32*f_inc
        d_max=32*f_inc  
        d_lay_1_to_2 = 32*f_d
        d_lay_3_to_4 = 64*f_d
        
        inputs = keras.Input(shape=(128,128, channels))     
         
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= self.activation)(inputs)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= self.activation)(inputs)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= self.activation)(inputs)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_1_pool = layers.MaxPooling2D(pool_size=(4, 4))(all_output)
        
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_1_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_1_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_1_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_1_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_2_pool = layers.MaxPooling2D(pool_size=(3, 3))(all_output)
        
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_2_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_2_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_2_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_2_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(all_output)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_3_pool)
        incept_1_to_5=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_3_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_3_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_3_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_4_pool = layers.MaxPooling2D(pool_size=(2, 2))(all_output)
        #
        incept_1_to_final =layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_4_pool)
        parallel = keras.models.Model(inputs, incept_1_to_final, name='parallel')     
        #parallel = keras.models.Model(inputs,all_output, name='parallel') 
        inp_q0_x1 = keras.Input(shape=(self.square_height,self.square_width, channels))
        inp_q0_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q0_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        inp_q1_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
           		
        inp_q2_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q3_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q4_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        inp_q5_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q6_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q7_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        tower_q0_x1 = parallel(inp_q0_x1)
        tower_q0_x2 = parallel(inp_q0_x2)
        tower_q0_x3 = parallel(inp_q0_x3)
        
        tower_q1_x1 = parallel(inp_q1_x1)
        tower_q1_x2 = parallel(inp_q1_x2)
        tower_q1_x3 = parallel(inp_q1_x3)
        		
        tower_q2_x1 = parallel(inp_q2_x1)
        tower_q2_x2 = parallel(inp_q2_x2)
        tower_q2_x3 = parallel(inp_q2_x3)
        
        tower_q3_x1 = parallel(inp_q3_x1)
        tower_q3_x2 = parallel(inp_q3_x2)
        tower_q3_x3 = parallel(inp_q3_x3)
        		
        tower_q4_x1 = parallel(inp_q4_x1)
        tower_q4_x2 = parallel(inp_q4_x2)
        tower_q4_x3 = parallel(inp_q4_x3)
        		
        tower_q5_x1 = parallel(inp_q5_x1)
        tower_q5_x2 = parallel(inp_q5_x2)
        tower_q5_x3 = parallel(inp_q5_x3)
        		
        tower_q6_x1 = parallel(inp_q6_x1)
        tower_q6_x2 = parallel(inp_q6_x2)
        tower_q6_x3 = parallel(inp_q6_x3)
        		
        tower_q7_x1 = parallel(inp_q7_x1)
        tower_q7_x2 = parallel(inp_q7_x2)
        tower_q7_x3 = parallel(inp_q7_x3)
        
        # merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3], axis=1)
        merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3,tower_q1_x1,tower_q1_x2,tower_q1_x3,tower_q2_x1, tower_q2_x2, tower_q2_x3,tower_q3_x1, tower_q3_x2, tower_q3_x3,tower_q4_x1, tower_q4_x2, tower_q4_x3,tower_q5_x1, tower_q5_x2, tower_q5_x3,tower_q6_x1, tower_q6_x2, tower_q6_x3,tower_q7_x1, tower_q7_x2, tower_q7_x3], axis=1)
        model_name = ''.join(['model_par_inception_only_mentor_vin_1_quat_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation= self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation= self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        
        model = keras.models.Model(inputs=[inp_q0_x1,inp_q0_x2,inp_q0_x3 ,inp_q1_x1,inp_q1_x2,inp_q1_x3,inp_q2_x1,inp_q2_x2,inp_q2_x3,inp_q3_x1,inp_q3_x2,inp_q3_x3,inp_q4_x1,inp_q4_x2,inp_q4_x3,inp_q5_x1,inp_q5_x2,inp_q5_x3,inp_q6_x1,inp_q6_x2,inp_q6_x3,inp_q7_x1,inp_q7_x2,inp_q7_x3], outputs=outputs, name=model_name)

        return model,model_name    


class model_par_inception_residual_mentor_quat_vin_1:

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu'):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only
        self.square_height=128
        self.square_width=128
        
    def model_maker(self,f_inc=1,f_d=1):
        
        channels=self.channels
        
        d1=128*f_inc
        d3=64*f_inc
        d5=32*f_inc
        d_max=32*f_inc  
        d_lay_1_to_2 = 32*f_d
        d_lay_3_to_4 = 64*f_d
        
        inputs = keras.Input(shape=(128,128, channels))
         
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= self.activation)(inputs)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= self.activation)(inputs)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= self.activation)(inputs)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_1_pool = layers.MaxPooling2D(pool_size=(4, 4))(all_output)
        
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_1_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_1_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_1_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_1_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        
        layer_1_2_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_1_pool)
        layer_21_residual = layers.add([layer_1_2_pool, all_output])
        
        layer_2_pool = layers.MaxPooling2D(pool_size=(3, 3))(layer_21_residual)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_2_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_2_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_2_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_2_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_2_3_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_2_pool)
        layer_32_residual = layers.add([layer_2_3_pool, all_output])
        
        layer_3_pool = layers.MaxPooling2D(pool_size=(3, 3))(layer_32_residual)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_3_pool)
        incept_1_to_5=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_3_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_3_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_3_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_3_4_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_3_pool)
        layer_34_residual = layers.add([layer_3_4_pool, all_output])
        
        layer_4_pool = layers.MaxPooling2D(pool_size=(2, 2))(layer_34_residual)
        #
        incept_1_to_final =layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation= self.activation)(layer_4_pool)
        parallel = keras.models.Model(inputs, incept_1_to_final, name='parallel')     
        #parallel = keras.models.Model(inputs,all_output, name='parallel') 
        inp_q0_x1 = keras.Input(shape=(self.square_height,self.square_width, channels))
        inp_q0_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q0_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        inp_q1_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q1_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
           		
        inp_q2_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q2_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q3_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q3_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q4_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q4_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        inp_q5_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q5_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q6_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q6_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        		
        inp_q7_x1 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x2 = keras.Input(shape=(self.square_height, self.square_width, channels))
        inp_q7_x3 = keras.Input(shape=(self.square_height, self.square_width, channels))
        
        
        tower_q0_x1 = parallel(inp_q0_x1)
        tower_q0_x2 = parallel(inp_q0_x2)
        tower_q0_x3 = parallel(inp_q0_x3)
        
        tower_q1_x1 = parallel(inp_q1_x1)
        tower_q1_x2 = parallel(inp_q1_x2)
        tower_q1_x3 = parallel(inp_q1_x3)
        		
        tower_q2_x1 = parallel(inp_q2_x1)
        tower_q2_x2 = parallel(inp_q2_x2)
        tower_q2_x3 = parallel(inp_q2_x3)
        
        tower_q3_x1 = parallel(inp_q3_x1)
        tower_q3_x2 = parallel(inp_q3_x2)
        tower_q3_x3 = parallel(inp_q3_x3)
        		
        tower_q4_x1 = parallel(inp_q4_x1)
        tower_q4_x2 = parallel(inp_q4_x2)
        tower_q4_x3 = parallel(inp_q4_x3)
        		
        tower_q5_x1 = parallel(inp_q5_x1)
        tower_q5_x2 = parallel(inp_q5_x2)
        tower_q5_x3 = parallel(inp_q5_x3)
        		
        tower_q6_x1 = parallel(inp_q6_x1)
        tower_q6_x2 = parallel(inp_q6_x2)
        tower_q6_x3 = parallel(inp_q6_x3)
        		
        tower_q7_x1 = parallel(inp_q7_x1)
        tower_q7_x2 = parallel(inp_q7_x2)
        tower_q7_x3 = parallel(inp_q7_x3)
        
        # merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3], axis=1)
        merged = layers.concatenate([tower_q0_x1, tower_q0_x2, tower_q0_x3,tower_q1_x1,tower_q1_x2,tower_q1_x3,tower_q2_x1, tower_q2_x2, tower_q2_x3,tower_q3_x1, tower_q3_x2, tower_q3_x3,tower_q4_x1, tower_q4_x2, tower_q4_x3,tower_q5_x1, tower_q5_x2, tower_q5_x3,tower_q6_x1, tower_q6_x2, tower_q6_x3,tower_q7_x1, tower_q7_x2, tower_q7_x3], axis=1)
        model_name = ''.join(['model_par_inception_residual_mentor_vin_1_quat_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation= self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation= self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        
        model = keras.models.Model(inputs=[inp_q0_x1,inp_q0_x2,inp_q0_x3 ,inp_q1_x1,inp_q1_x2,inp_q1_x3,inp_q2_x1,inp_q2_x2,inp_q2_x3,inp_q3_x1,inp_q3_x2,inp_q3_x3,inp_q4_x1,inp_q4_x2,inp_q4_x3,inp_q5_x1,inp_q5_x2,inp_q5_x3,inp_q6_x1,inp_q6_x2,inp_q6_x3,inp_q7_x1,inp_q7_x2,inp_q7_x3], outputs=outputs, name=model_name)
    
        return model,model_name    