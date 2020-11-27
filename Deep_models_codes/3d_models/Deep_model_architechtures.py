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

class model_brain_inception_only_vin_1:

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu'):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.square_height=200
        self.square_width=200
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
        
        inputs = keras.Input(shape=(self.square_height, self.square_width, self.channels))
        
        inp1 = keras.Input(shape=(self.square_height, self.square_width,  self.channels))
        inp2 = keras.Input(shape=(self.square_height, self.square_width,  self.channels))
        inp3 = keras.Input(shape=(self.square_height, self.square_width,  self.channels))    
         
        lay_1_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
        lay_1_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_1_incept_max_pool)
        
        lay_1_all_output = layers.concatenate([lay_1_incept_1, lay_1_incept_3,lay_1_incept_5,lay_1_incept_max_pool_depth], axis=3)
        '''layer 1 general network'''    
        layer_1_INCEPT_Net = keras.models.Model(inputs, lay_1_all_output, name='layer_1_INCEPT')     
        
        '''Applying layer_1 in projections'''
        #layer1 output of projection 1
        inp_1_lay_1_all_output = layer_1_INCEPT_Net(inp1) 
        inp_2_lay_1_all_output = layer_1_INCEPT_Net(inp2) 
        inp_3_lay_1_all_output = layer_1_INCEPT_Net(inp3) 

        lay_1_inp1_incept_1_to_3,lay_1_inp1_incept_1_to_5,inp1_lay_1_pool = self.layer_1_reducer(inp_1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp2_incept_1_to_3,lay_1_inp2_incept_1_to_5,inp2_lay_1_pool = self.layer_1_reducer(inp_2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp3_incept_1_to_3,lay_1_inp3_incept_1_to_5,inp3_lay_1_pool = self.layer_1_reducer(inp_3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
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

        '''Applying layer_2 in projections'''
        #layer1 output of projection 1
        inp_1_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp1_incept_1_to_3,lay_1_inp1_incept_1_to_5,inp1_lay_1_pool]) 
        inp_2_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp2_incept_1_to_3,lay_1_inp2_incept_1_to_5,inp2_lay_1_pool]) 
        inp_3_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp3_incept_1_to_3,lay_1_inp3_incept_1_to_5,inp3_lay_1_pool]) 

        #layer2 output of projection 1
        lay_2_inp1_incept_1_to_3,lay_2_inp1_incept_1_to_5,inp1_lay_2_pool = self.layer_2_reducer(inp_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp2_incept_1_to_3,lay_2_inp2_incept_1_to_5,inp2_lay_2_pool = self.layer_2_reducer(inp_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp3_incept_1_to_3,lay_2_inp3_incept_1_to_5,inp3_lay_2_pool = self.layer_2_reducer(inp_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

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

        '''Applying layer_3 in projections'''
        #layer3 output of projection 1
        inp_1_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp1_incept_1_to_3,lay_2_inp1_incept_1_to_5,inp1_lay_2_pool]) 
        inp_2_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp2_incept_1_to_3,lay_2_inp2_incept_1_to_5,inp2_lay_2_pool]) 
        inp_3_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp3_incept_1_to_3,lay_2_inp3_incept_1_to_5,inp3_lay_2_pool]) 
      
        #layer3 output of projection 1
        lay_3_inp1_incept_1_to_3,lay_3_inp1_incept_1_to_5,inp1_lay_3_pool = self.layer_3_reducer(inp_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp2_incept_1_to_3,lay_3_inp2_incept_1_to_5,inp2_lay_3_pool = self.layer_3_reducer(inp_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp3_incept_1_to_3,lay_3_inp3_incept_1_to_5,inp3_lay_3_pool = self.layer_3_reducer(inp_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)

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
        '''Applying layer_4 in projections'''
        inp_1_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp1_incept_1_to_3,lay_3_inp1_incept_1_to_5,inp1_lay_3_pool]) 
        inp_2_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp2_incept_1_to_3,lay_3_inp2_incept_1_to_5,inp2_lay_3_pool]) 
        inp_3_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp3_incept_1_to_3,lay_3_inp3_incept_1_to_5,inp3_lay_3_pool]) 

        
        #parallel = keras.models.Model(inputs,all_output, name='parallel') 
        tower_1 = self.layer_4_final(inp_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_2 = self.layer_4_final(inp_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_3 = self.layer_4_final(inp_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_brain_inception_only_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_brain_inception_only_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation=self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation=self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        
        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name    


class model_brain_incept_RESIDUAL_only_vin_1(model_brain_inception_only_vin_1):

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu'):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.square_height=200
        self.square_width=200
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
        
        inputs = keras.Input(shape=(self.square_height, self.square_width, self.channels))
        
        inp1 = keras.Input(shape=(self.square_height, self.square_width,  self.channels))
        inp2 = keras.Input(shape=(self.square_height, self.square_width,  self.channels))
        inp3 = keras.Input(shape=(self.square_height, self.square_width,  self.channels))    
         
        lay_1_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(inputs)
        lay_1_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
        lay_1_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_1_incept_max_pool)
        
        lay_1_all_output = layers.concatenate([lay_1_incept_1, lay_1_incept_3,lay_1_incept_5,lay_1_incept_max_pool_depth], axis=3)
        '''layer 1 general network'''    
        layer_1_INCEPT_Net = keras.models.Model(inputs, lay_1_all_output, name='layer_1_INCEPT')     
        
        '''Applying layer_1 in projections'''
        #layer1 output of projection 1
        inp_1_lay_1_all_output = layer_1_INCEPT_Net(inp1) 
        inp_2_lay_1_all_output = layer_1_INCEPT_Net(inp2) 
        inp_3_lay_1_all_output = layer_1_INCEPT_Net(inp3) 

        lay_1_inp1_incept_1_to_3,lay_1_inp1_incept_1_to_5,inp1_lay_1_pool = self.layer_1_reducer(inp_1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp2_incept_1_to_3,lay_1_inp2_incept_1_to_5,inp2_lay_1_pool = self.layer_1_reducer(inp_2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
        lay_1_inp3_incept_1_to_3,lay_1_inp3_incept_1_to_5,inp3_lay_1_pool = self.layer_1_reducer(inp_3_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
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

        '''Applying layer_2 in projections'''
        #layer1 output of projection 1
        inp_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp1_incept_1_to_3,lay_1_inp1_incept_1_to_5,inp1_lay_1_pool]) 
        inp_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp2_incept_1_to_3,lay_1_inp2_incept_1_to_5,inp2_lay_1_pool]) 
        inp_3_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp3_incept_1_to_3,lay_1_inp3_incept_1_to_5,inp3_lay_1_pool]) 
        #make residUAL NETWORK
        inp_1_lay_2_all_output= layers.add([inp1_lay_1_pool, inp_1_lay_2_incept_all_output])
        inp_2_lay_2_all_output= layers.add([inp2_lay_1_pool, inp_2_lay_2_incept_all_output])
        inp_3_lay_2_all_output= layers.add([inp3_lay_1_pool, inp_3_lay_2_incept_all_output])

        #layer2 output of projection 1
        lay_2_inp1_incept_1_to_3,lay_2_inp1_incept_1_to_5,inp1_lay_2_pool = self.layer_2_reducer(inp_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp2_incept_1_to_3,lay_2_inp2_incept_1_to_5,inp2_lay_2_pool = self.layer_2_reducer(inp_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
        lay_2_inp3_incept_1_to_3,lay_2_inp3_incept_1_to_5,inp3_lay_2_pool = self.layer_2_reducer(inp_3_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)

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

        '''Applying layer_3 in projections'''
        #layer3 output of projection 1
        inp_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp1_incept_1_to_3,lay_2_inp1_incept_1_to_5,inp1_lay_2_pool]) 
        inp_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp2_incept_1_to_3,lay_2_inp2_incept_1_to_5,inp2_lay_2_pool]) 
        inp_3_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp3_incept_1_to_3,lay_2_inp3_incept_1_to_5,inp3_lay_2_pool]) 
      
        #make residUAL NETWORK
        inp_1_lay_3_all_output = layers.add([inp1_lay_2_pool, inp_1_lay_3_incept_all_output])
        inp_2_lay_3_all_output = layers.add([inp2_lay_2_pool, inp_2_lay_3_incept_all_output])
        inp_3_lay_3_all_output = layers.add([inp3_lay_2_pool, inp_3_lay_3_incept_all_output])

        #layer3 output of projection 1
        lay_3_inp1_incept_1_to_3,lay_3_inp1_incept_1_to_5,inp1_lay_3_pool = self.layer_3_reducer(inp_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp2_incept_1_to_3,lay_3_inp2_incept_1_to_5,inp2_lay_3_pool = self.layer_3_reducer(inp_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
        lay_3_inp3_incept_1_to_3,lay_3_inp3_incept_1_to_5,inp3_lay_3_pool = self.layer_3_reducer(inp_3_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)

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
        '''Applying layer_4 in projections'''
        inp_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp1_incept_1_to_3,lay_3_inp1_incept_1_to_5,inp1_lay_3_pool]) 
        inp_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp2_incept_1_to_3,lay_3_inp2_incept_1_to_5,inp2_lay_3_pool]) 
        inp_3_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp3_incept_1_to_3,lay_3_inp3_incept_1_to_5,inp3_lay_3_pool]) 
        #make residUAL NETWORK
        inp_1_lay_4_all_output = layers.add([inp1_lay_3_pool, inp_1_lay_4_incept_all_output])
        inp_2_lay_4_all_output = layers.add([inp2_lay_3_pool, inp_2_lay_4_incept_all_output])
        inp_3_lay_4_all_output = layers.add([inp3_lay_3_pool, inp_3_lay_4_incept_all_output])

        
        #parallel = keras.models.Model(inputs,all_output, name='parallel') 
        tower_1 = self.layer_4_final(inp_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_2 = self.layer_4_final(inp_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
        tower_3 = self.layer_4_final(inp_3_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_brain_incept_RESIDUAL_only_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_brain_incept_RESIDUAL_only_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation=self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation=self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        
        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name    
    
    
class model_inception_only_mentor_vin_1:

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu'):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only

    def parallel(self,inputs,f_inc,f_d):

        d1=128*f_inc
        d3=64*f_inc
        d5=32*f_inc
        d_max=32*f_inc  
        d_lay_1_to_2 = 32*f_d
        d_lay_3_to_4 = 64*f_d
         
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_1_pool = layers.MaxPooling2D(pool_size=(4, 4))(all_output)
        
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_1_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_2_pool = layers.MaxPooling2D(pool_size=(3, 3))(all_output)
        
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_2_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(all_output)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_1_to_5=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_3_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_4_pool = layers.MaxPooling2D(pool_size=(2, 2))(all_output)
        #
        incept_1_to_final =layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_4_pool)
        return incept_1_to_final
    
    def model_maker(self,f_inc=1,f_d=1):

        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,f_inc,f_d)
        tower_2 = self.parallel(inp2,f_inc,f_d)
        tower_3 = self.parallel(inp3,f_inc,f_d)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_inception_only_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_inception_only_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation=self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation=self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        
        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name    

class model_inception_residual_mentor_vin_1:

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu'):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only

    def parallel(self,inputs,f_inc,f_d):

        d1=128*f_inc
        d3=64*f_inc
        d5=32*f_inc
        d_max=32*f_inc  
        d_lay_1_to_2 = 32*f_d
        d_lay_3_to_4 = 64*f_d
         
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_1_pool = layers.MaxPooling2D(pool_size=(4, 4))(all_output)
        
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_1_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        
        layer_1_2_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        layer_21_residual = layers.add([layer_1_2_pool, all_output])
        
        layer_2_pool = layers.MaxPooling2D(pool_size=(3, 3))(layer_21_residual)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_2_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_2_3_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        layer_32_residual = layers.add([layer_2_3_pool, all_output])
        
        layer_3_pool = layers.MaxPooling2D(pool_size=(3, 3))(layer_32_residual)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_1_to_5=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_3_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_3_4_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        layer_34_residual = layers.add([layer_3_4_pool, all_output])
        
        layer_4_pool = layers.MaxPooling2D(pool_size=(3, 3))(layer_34_residual)
        #
        incept_1_to_final =layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_4_pool)
        return incept_1_to_final
    
    def model_maker(self,f_inc=1,f_d=1):

        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,f_inc,f_d)
        tower_2 = self.parallel(inp2,f_inc,f_d)
        tower_3 = self.parallel(inp3,f_inc,f_d)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_inception_residual_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_inception_residual_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation=self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation=self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        
        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name    
    
'''parallel inception models'''
class model_par_inception_only_mentor_vin_1:

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu'):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only

    def model_maker(self,f_inc=1,f_d=1):

        d1=128*f_inc
        d3=64*f_inc
        d5=32*f_inc
        d_max=32*f_inc  
        d_lay_1_to_2 = 32*f_d
        d_lay_3_to_4 = 64*f_d

        inputs = keras.Input(shape=(200, 200, self.channels))
        
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))    
         
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_1_pool = layers.MaxPooling2D(pool_size=(4, 4))(all_output)
        
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_1_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_2_pool = layers.MaxPooling2D(pool_size=(3, 3))(all_output)
        
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_2_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(all_output)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_1_to_5=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_3_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_4_pool = layers.MaxPooling2D(pool_size=(2, 2))(all_output)
        #
        incept_1_to_final =layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_4_pool)
        
        parallel = keras.models.Model(inputs, incept_1_to_final, name='parallel')     
        #parallel = keras.models.Model(inputs,all_output, name='parallel') 
        tower_1 = parallel(inp1)
        tower_2 = parallel(inp2)
        tower_3 = parallel(inp3)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_par_inception_only_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_par_inception_only_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation=self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation=self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        
        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name    

class model_par_inception_residual_mentor_vin_1:

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu'):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only

    def model_maker(self,f_inc=1,f_d=1):

        d1=128*f_inc
        d3=64*f_inc
        d5=32*f_inc
        d_max=32*f_inc  
        d_lay_1_to_2 = 32*f_d
        d_lay_3_to_4 = 64*f_d

        inputs = keras.Input(shape=(200, 200, self.channels))
        
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))      
         
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(inputs)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_1_pool = layers.MaxPooling2D(pool_size=(4, 4))(all_output)
        
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_1_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        
        layer_1_2_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        layer_21_residual = layers.add([layer_1_2_pool, all_output])
        
        layer_2_pool = layers.MaxPooling2D(pool_size=(3, 3))(layer_21_residual)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_2_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_2_3_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        layer_32_residual = layers.add([layer_2_3_pool, all_output])
        
        layer_3_pool = layers.MaxPooling2D(pool_size=(3, 3))(layer_32_residual)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_1_to_5=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        
        incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_3)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(incept_1_to_5)
        incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_3_pool)
        incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)
        
        all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
        layer_3_4_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        layer_34_residual = layers.add([layer_3_4_pool, all_output])
        
        layer_4_pool = layers.MaxPooling2D(pool_size=(3, 3))(layer_34_residual)
        #
        incept_1_to_final =layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_4_pool)
        parallel = keras.models.Model(inputs, incept_1_to_final, name='parallel')     
        #parallel = keras.models.Model(inputs,all_output, name='parallel') 
        tower_1 = parallel(inp1)
        tower_2 = parallel(inp2)
        tower_3 = parallel(inp3)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_par_inception_residual_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_par_inception_residual_mentor_vin_1_act_',self.activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation=self.activation)(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation=self.activation)(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        
        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name    

class model_vin_4:
    
    def __init__(self,channels=17,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only
    
    def parallel(self,inputs,k1,k2,k3,k4):
    
        xy = layers.Conv2D(32, (k1,k1),strides=(1,1),padding='same',activation='relu')(inputs)
        block_1_xy_output = layers.MaxPooling2D(pool_size=(4, 4))(xy)
        
        xy = layers.Conv2D(32, (k2,k2),strides=(1,1),padding='same',activation='relu')(block_1_xy_output)
        block_2_output = layers.add([xy, block_1_xy_output])
    
        block_2_xy_output = layers.MaxPooling2D(pool_size=(2, 2))(block_2_output)
    
        xy = layers.Conv2D(64, (k3,k3), activation='relu', padding='same')(block_2_xy_output)
        '''to be removed'''
        xy = layers.MaxPooling2D(pool_size=(2, 2))(xy)
        xy = layers.Conv2D(64, (k4,k4), activation='relu', padding='same')(xy)
        block_3_xy_output = layers.MaxPooling2D(pool_size=(2, 2))(xy)
        return block_3_xy_output
 
#    def model_maker_same_kernal_projections(self,p1=7,p2=5,p3=3):
    def model_maker(self,p1=3,p2=5):
        
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,p2,p2,p1,p1)
        tower_2 = self.parallel(inp2,p2,p2,p1,p1)
        tower_3 = self.parallel(inp3,p2,p2,p1,p1)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name =''.join(['MIN_MAX_model_vin_4_p1_',str(p1),'_p2_',str(p2),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name =''.join(['model_vin_4_p1_',str(p1),'_p2_',str(p2),'.h5'])
      
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model_name = 'model_vin_4_same.h5'
        model = keras.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name='model_vin_4_same')
        
        return model,model_name
    
#    def model_maker_same_kernal_combo_projections(self,p11=7,p12=3,p21=5,p22=5,p31=3,p32=7):
#        
#        inp1 = keras.Input(shape=(200, 200, self.channels))
#        inp2 = keras.Input(shape=(200, 200, self.channels))
#        inp3 = keras.Input(shape=(200, 200, self.channels))
#        
#        tower_1 = self.parallel(inp1,p11,p11,p12,p12)
#        tower_2 = self.parallel(inp2,p21,p21,p22,p22)
#        tower_3 = self.parallel(inp3,p31,p32,p31,p32)
#        
#        merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
#        merged = layers.Flatten()(merged)
#        all_co = layers.Dense(100, activation='relu')(merged)
#        all_co = layers.Dropout(0.5)(all_co)
#        all_co = layers.Dense(50, activation='relu')(all_co)
#        all_co= layers.Dropout(0.5)(all_co)
#        outputs = layers.Dense(3, activation='softmax')(all_co)
#
#        model_name = 'model_vin_4_combo.h5'
#        model = keras.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name='model_vin_4_combo')
#
#        return model,model_name


class model_accidental:
    
    def __init__(self,channels=17,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only
    
    def parallel(self,inputs,k1,k2,k3,k4,d1):
    
        xy = layers.Conv2D(d1, (k1,k1),strides=(1,1),padding='same',activation='relu')(inputs)
        block_1_xy_output = layers.MaxPooling2D(pool_size=(4, 4))(xy)
        
        xy = layers.Conv2D(d1, (k2,k2),strides=(1,1),padding='same',activation='relu')(block_1_xy_output)
        block_2_xy_output = layers.MaxPooling2D(pool_size=(2, 2))(xy)
    
        xy = layers.Conv2D(2*d1, (k3,k3), activation='relu', padding='same')(block_2_xy_output)
        xy = layers.MaxPooling2D(pool_size=(2, 2))(xy)

        xy = layers.Conv2D(2*d1, (k4,k4), activation='relu', padding='same')(xy)
        block_3_xy_output = layers.MaxPooling2D(pool_size=(2, 2))(xy)
        return block_3_xy_output
 
    def model_maker(self,p1=3,p2=3,p3=3,d1=32):
        
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,p1,p2,p3,p3,d1)
        tower_2 = self.parallel(inp2,p1,p2,p3,p3,d1)
        tower_3 = self.parallel(inp3,p1,p2,p3,p3,d1)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name =''.join(['MIN_MAX_model_accidental_p1_',str(p1),'_p2_',str(p2),'_p3_',str(p3),'_d1_',str(d1),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name =''.join(['model_accidental_p1_',str(p1),'_p2_',str(p2),'_p3_',str(p3),'_d1_',str(d1),'.h5'])
            
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name

'''Models with inception'''
class model_inception_layer_1:
    
    def __init__(self,channels=19,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only

    def parallel(self,inputs,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4):
    
        incept_3 = layers.Conv2D(d1, (k_b1,k_b1),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (k_b2,k_b2),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (k_b3,k_b3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (k_b4,k_b4),strides=(1,1),padding='same',activation='relu')(inputs)
#        incept_10 = layers.Conv2D(d1, (k_b5,k_b5),strides=(1,1),padding='same',activation='relu')(inputs)

        data = layers.concatenate([incept_3, incept_5, incept_7,incept_9], axis=3)
        data = layers.MaxPooling2D(pool_size=(4, 4))(data)
        
        data_2 = layers.Conv2D(d1*4, (k2,k2),strides=(1,1),padding='same',activation='relu')(data)
        data = layers.add([data_2, data])
    
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
    
        data = layers.Conv2D(64, (k3,k3), activation='relu', padding='same')(data)
        '''to be removed'''
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        data = layers.Conv2D(64, (k4,k4), activation='relu', padding='same')(data)
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        return data

    def model_maker(self,d1=8,k2=3,k3=3,k4=3,k_b1=3,k_b2=3,k_b3=3,k_b4=5,k_b5=5):
         
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4)
        tower_2 = self.parallel(inp2,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4)
        tower_3 = self.parallel(inp3,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4)

        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name =''.join(['MIN_MAX_model_inception_layer_1_d1_',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'_kb1_',str(k_b1),'_kb2_',str(k_b2),'_kb3_',str(k_b3),'_kb4_',str(k_b4),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name =''.join(['model_inception_layer_1_d1_',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'_kb1_',str(k_b1),'_kb2_',str(k_b2),'_kb3_',str(k_b3),'_kb4_',str(k_b4),'.h5'])
        

        merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name

'''Models with inception'''
class model_inception_layer_1_normal:
    
    def __init__(self,channels=19,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only

    def parallel(self,inputs,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4):
    
        incept_3 = layers.Conv2D(d1, (k_b1,k_b1),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (k_b2,k_b2),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (k_b3,k_b3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (k_b4,k_b4),strides=(1,1),padding='same',activation='relu')(inputs)
        
        data = layers.concatenate([incept_3, incept_5, incept_7,incept_9], axis=3)
        data = layers.MaxPooling2D(pool_size=(4, 4))(data)
        
        data = layers.Conv2D(d1*4, (k2,k2),strides=(1,1),padding='same',activation='relu')(data)

        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
    
        data = layers.Conv2D(64, (k3,k3), activation='relu', padding='same')(data)
        '''to be removed'''
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        data = layers.Conv2D(64, (k4,k4), activation='relu', padding='same')(data)
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        return data

    def model_maker(self,d1=8,k2=3,k3=3,k4=3,k_b1=3,k_b2=3,k_b3=3,k_b4=5):
         
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4)
        tower_2 = self.parallel(inp2,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4)
        tower_3 = self.parallel(inp3,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4)

        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name =''.join(['MIN_MAX_model_inception_layer_1_d1_',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'_kb1_',str(k_b1),'_kb2_',str(k_b2),'_kb3_',str(k_b3),'_kb4_',str(k_b4),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name =''.join(['model_inception_layer_1_d1_',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'_kb1_',str(k_b1),'_kb2_',str(k_b2),'_kb3_',str(k_b3),'_kb4_',str(k_b4),'.h5'])
        

        merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name
'''Models with inception_alias'''
class model_inception_layer_1_alias_1:
    
    def __init__(self,channels=19,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only

    def parallel(self,inputs,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5):
    
        incept_3 = layers.Conv2D(d1, (k_b1,k_b1),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (k_b2,k_b2),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (k_b3,k_b3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (k_b4,k_b4),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_10 = layers.Conv2D(d1, (k_b5,k_b5),strides=(1,1),padding='same',activation='relu')(inputs)
#        incept_10 = layers.Conv2D(d1, (k_b5,k_b5),strides=(1,1),padding='same',activation='relu')(inputs)

        data = layers.concatenate([incept_3, incept_5, incept_7,incept_9,incept_10], axis=3)
        data = layers.MaxPooling2D(pool_size=(4, 4))(data)
        
        data_2 = layers.Conv2D(d1*5, (k2,k2),strides=(1,1),padding='same',activation='relu')(data)
        data = layers.add([data_2, data])
    
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
    
        data = layers.Conv2D(64, (k3,k3), activation='relu', padding='same')(data)
        '''to be removed'''
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        data = layers.Conv2D(64, (k4,k4), activation='relu', padding='same')(data)
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        return data

    def model_maker(self,d1=8,k2=3,k3=3,k4=3,k_b1=3,k_b2=3,k_b3=3,k_b4=5,k_b5=5):
         
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5)
        tower_2 = self.parallel(inp2,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5)
        tower_3 = self.parallel(inp3,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5)

        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name =''.join(['MIN_MAX_model_inception_layer_1_d1_',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'_kb1_',str(k_b1),'_kb2_',str(k_b2),'_kb3_',str(k_b3),'_kb4_',str(k_b4),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name =''.join(['model_inception_layer_1_d1_',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'_kb1_',str(k_b1),'_kb2_',str(k_b2),'_kb3_',str(k_b3),'_kb4_',str(k_b4),'.h5'])
        

        merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name
'''Models with inception_alias'''
class model_inception_layer_1_normal_alias_1:
    
    def __init__(self,channels=19,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only

    def parallel(self,inputs,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5):
    
        incept_3 = layers.Conv2D(d1, (k_b1,k_b1),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (k_b2,k_b2),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (k_b3,k_b3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (k_b4,k_b4),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_10 = layers.Conv2D(d1, (k_b5,k_b5),strides=(1,1),padding='same',activation='relu')(inputs)
#        incept_10 = layers.Conv2D(d1, (k_b5,k_b5),strides=(1,1),padding='same',activation='relu')(inputs)

        data = layers.concatenate([incept_3, incept_5, incept_7,incept_9,incept_10], axis=3)
        data = layers.MaxPooling2D(pool_size=(4, 4))(data)
        
        data = layers.Conv2D(d1*5, (k2,k2),strides=(1,1),padding='same',activation='relu')(data)
    
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
    
        data = layers.Conv2D(64, (k3,k3), activation='relu', padding='same')(data)
        '''to be removed'''
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        data = layers.Conv2D(64, (k4,k4), activation='relu', padding='same')(data)
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        return data

    def model_maker(self,d1=8,k2=3,k3=3,k4=3,k_b1=3,k_b2=3,k_b3=3,k_b4=5,k_b5=5):
         
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5)
        tower_2 = self.parallel(inp2,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5)
        tower_3 = self.parallel(inp3,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5)

        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name =''.join(['MIN_MAX_model_inception_layer_1_d1_',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'_kb1_',str(k_b1),'_kb2_',str(k_b2),'_kb3_',str(k_b3),'_kb4_',str(k_b4),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name =''.join(['model_inception_layer_1_d1_',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'_kb1_',str(k_b1),'_kb2_',str(k_b2),'_kb3_',str(k_b3),'_kb4_',str(k_b4),'.h5'])
        

        merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name
'''Models with inception_alias'''
class model_inception_layer_1_normal_alias_2:
    
    def __init__(self,channels=19,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only

    def parallel(self,inputs,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5,k_b6):
    
        incept_3 = layers.Conv2D(d1, (k_b1,k_b1),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (k_b2,k_b2),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (k_b3,k_b3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (k_b4,k_b4),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_10 = layers.Conv2D(d1, (k_b5,k_b5),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_11 = layers.Conv2D(d1, (k_b6,k_b6),strides=(1,1),padding='same',activation='relu')(inputs)

        data = layers.concatenate([incept_3, incept_5, incept_7,incept_9,incept_10,incept_11], axis=3)
        data = layers.MaxPooling2D(pool_size=(4, 4))(data)
        
        data = layers.Conv2D(d1*6, (k2,k2),strides=(1,1),padding='same',activation='relu')(data)
    
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
    
        data = layers.Conv2D(64, (k3,k3), activation='relu', padding='same')(data)
        '''to be removed'''
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        data = layers.Conv2D(64, (k4,k4), activation='relu', padding='same')(data)
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        return data

    def model_maker(self,d1=8,k2=3,k3=3,k4=3,k_b1=3,k_b2=3,k_b3=3,k_b4=5,k_b5=5,k_b6=5):
         
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5,k_b6)
        tower_2 = self.parallel(inp2,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5,k_b6)
        tower_3 = self.parallel(inp3,d1,k2,k3,k4,k_b1,k_b2,k_b3,k_b4,k_b5,k_b6)

        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name =''.join(['MIN_MAX_model_inception_layer_1_d1_',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'_kb1_',str(k_b1),'_kb2_',str(k_b2),'_kb3_',str(k_b3),'_kb4_',str(k_b4),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name =''.join(['model_inception_layer_1_d1_',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'_kb1_',str(k_b1),'_kb2_',str(k_b2),'_kb3_',str(k_b3),'_kb4_',str(k_b4),'.h5'])
        

        merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name
'''model-1'''
class model_inception_vin_1:
    
    def __init__(self,channels=17):
        'Initialization'
        self.channels = channels
        

    def parallel(self,inputs,d1,k2,k3,k4):
    
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(inputs)
        data = layers.concatenate([incept_3, incept_5, incept_7,incept_9], axis=3)
        data = layers.MaxPooling2D(pool_size=(4, 4))(data)
        
        data_2 = layers.Conv2D(d1*4, (k2,k2),strides=(1,1),padding='same',activation='relu')(data)
        data = layers.add([data_2, data])
    
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
    
        data = layers.Conv2D(64, (k3,k3), activation='relu', padding='same')(data)
        '''to be removed'''
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        data = layers.Conv2D(64, (k4,k4), activation='relu', padding='same')(data)
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        return data

    def model_maker(self,d1=8,k2=3,k3=3,k4=3):
         
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,d1,k2,k3,k4)
        tower_2 = self.parallel(inp2,d1,k2,k3,k4)
        tower_3 = self.parallel(inp3,d1,k2,k3,k4)

        merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model_name = 'model_inception_vin_1.h5'
        model = keras.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name='model_inception_vin_1')
        
        return model,model_name


'''model-2'''
class model_inception_all__depths__inception_vin_1:
    
    def __init__(self,channels=17):
        'Initialization'
        self.channels = channels
        
    def parallel(self,inputs,d1,d2,d3,d4):
    
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(inputs)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_9)
        
        incept_3 = layers.Conv2D(d2, (3,3),strides=(1,1),padding='same',activation='relu')(incept_3_pool)
        incept_5 = layers.Conv2D(d2, (5,5),strides=(1,1),padding='same',activation='relu')(incept_5_pool)
        incept_7 = layers.Conv2D(d2, (7,7),strides=(1,1),padding='same',activation='relu')(incept_7_pool)
        incept_9 = layers.Conv2D(d2, (9,9),strides=(1,1),padding='same',activation='relu')(incept_9_pool)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
        
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation='relu')(incept_3_pool)
        incept_5 = layers.Conv2D(d3, (5,5),strides=(1,1),padding='same',activation='relu')(incept_5_pool)
        incept_7 = layers.Conv2D(d3, (7,7),strides=(1,1),padding='same',activation='relu')(incept_7_pool)
        incept_9 = layers.Conv2D(d3, (9,9),strides=(1,1),padding='same',activation='relu')(incept_9_pool)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
        
        incept_3 = layers.Conv2D(d4, (3,3),strides=(1,1),padding='same',activation='relu')(incept_3_pool)
        incept_5 = layers.Conv2D(d4, (5,5),strides=(1,1),padding='same',activation='relu')(incept_5_pool)
        incept_7 = layers.Conv2D(d4, (7,7),strides=(1,1),padding='same',activation='relu')(incept_7_pool)
        incept_9 = layers.Conv2D(d4, (9,9),strides=(1,1),padding='same',activation='relu')(incept_9_pool)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
         
        all_output = layers.concatenate([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool], axis=3)
    
        return all_output

    def model_maker(self,d1=8,d2=8,d3=16,d4=16):
         
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,d1,d2,d3,d4)
        tower_2 = self.parallel(inp2,d1,d2,d3,d4)
        tower_3 = self.parallel(inp3,d1,d2,d3,d4)
        
        merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
    
        model_name = 'model_inception_all__depths__inception_vin_1.h5'
        model = keras.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name='model_inception_all__depths__inception_vin_1')
        return model,model_name
    
'''model type 3'''
class model_inception_all__depths__inception_complic_vin_1:
    
    def __init__(self,channels=17):
        'Initialization'
        self.channels = channels
        
    def parallel(self,inputs,d1,d3):
    
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(inputs)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_9)
        
        all_output = layers.concatenate([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool], axis=3)
    
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
          
        all_output_2 = layers.concatenate([incept_3, incept_5, incept_7,incept_9], axis=3)
        all_output=  layers.add([all_output,all_output_2])
        all_output = layers.MaxPooling2D(pool_size=(2,2))(all_output)
     
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d3, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d3, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d3, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
        
        all_output = layers.concatenate([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool], axis=3)
        
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d3, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d3, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d3, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
        
        all_output_4 = layers.concatenate([incept_3, incept_5, incept_7,incept_9], axis=3)
        all_output=  layers.add([all_output,all_output_4])
        all_output = layers.MaxPooling2D(pool_size=(2,2))(all_output)
    
        return all_output

    def model_maker(self,d1=8,d3=16):
         
        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        tower_1 = self.parallel(inp1,d1,d3)
        tower_2 = self.parallel(inp2,d1,d3)
        tower_3 = self.parallel(inp3,d1,d3)
        
        merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)
        model_name = 'model_inception_all__depths__inception_complic_vin_1.h5'
        model = keras.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name='model_inception_all__depths__inception_complic_vin_1')
        return model,model_name

'''Models have parallel towers'''

'''Models with inception'''
'''model parallel tower-1'''
class model_par_inception_vin_1:
    
    def __init__(self,channels=17,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only
        

    def model_maker(self,d1=8,k2=3,k3=3,k4=3):
        
        inputs = keras.Input(shape=(200, 200, self.channels))

        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(inputs)
        data = layers.concatenate([incept_3, incept_5, incept_7,incept_9], axis=3)
        data = layers.MaxPooling2D(pool_size=(4, 4))(data)
        
        data_2 = layers.Conv2D(d1*4, (k2,k2),strides=(1,1),padding='same',activation='relu')(data)
        data = layers.add([data_2, data])
    
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
    
        data = layers.Conv2D(64, (k3,k3), activation='relu', padding='same')(data)
        '''to be removed'''
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        data = layers.Conv2D(64, (k4,k4), activation='relu', padding='same')(data)
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        
        parallel = keras.models.Model(inputs, data, name='parallel')     

        tower_1 = parallel(inp1)
        tower_2 = parallel(inp2)
        tower_3 = parallel(inp3)
       
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_inception_vin_1_d1',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_inception_vin_1_d1',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'.h5'])

        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name
'''model parallel tower-2'''
class model_par_inception_w_o_addition_vin_1:
    
    def __init__(self,channels=17,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only
        

    def model_maker(self,d1=8,k2=3,k3=3,k4=3):
        
        inputs = keras.Input(shape=(200, 200, self.channels))

        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(inputs)
        data = layers.concatenate([incept_3, incept_5, incept_7,incept_9], axis=3)
        data = layers.MaxPooling2D(pool_size=(4, 4))(data)
        
        data = layers.Conv2D(d1*4, (k2,k2),strides=(1,1),padding='same',activation='relu')(data)   
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
    
        data = layers.Conv2D(64, (k3,k3), activation='relu', padding='same')(data)
        '''to be removed'''
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        data = layers.Conv2D(64, (k4,k4), activation='relu', padding='same')(data)
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        
        parallel = keras.models.Model(inputs, data, name='parallel')     

        tower_1 = parallel(inp1)
        tower_2 = parallel(inp2)
        tower_3 = parallel(inp3)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_inception_w_o_addition_vin_1_d1',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_inception_w_o_addition_vin_1_d1',str(d1),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'.h5'])

        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name

'''model parallel tower -3'''
class model_par_inception_all__depths__inception_vin_1:
    
    def __init__(self,channels=17,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only
        
    def model_maker(self,d1=4,d2=4,d3=8,d4=8):
        inputs = keras.Input(shape=(200, 200, self.channels))

        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))    
       
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(inputs)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_9)
        
        incept_3 = layers.Conv2D(d2, (3,3),strides=(1,1),padding='same',activation='relu')(incept_3_pool)
        incept_5 = layers.Conv2D(d2, (5,5),strides=(1,1),padding='same',activation='relu')(incept_5_pool)
        incept_7 = layers.Conv2D(d2, (7,7),strides=(1,1),padding='same',activation='relu')(incept_7_pool)
        incept_9 = layers.Conv2D(d2, (9,9),strides=(1,1),padding='same',activation='relu')(incept_9_pool)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
        
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation='relu')(incept_3_pool)
        incept_5 = layers.Conv2D(d3, (5,5),strides=(1,1),padding='same',activation='relu')(incept_5_pool)
        incept_7 = layers.Conv2D(d3, (7,7),strides=(1,1),padding='same',activation='relu')(incept_7_pool)
        incept_9 = layers.Conv2D(d3, (9,9),strides=(1,1),padding='same',activation='relu')(incept_9_pool)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
        
        incept_3 = layers.Conv2D(d4, (3,3),strides=(1,1),padding='same',activation='relu')(incept_3_pool)
        incept_5 = layers.Conv2D(d4, (5,5),strides=(1,1),padding='same',activation='relu')(incept_5_pool)
        incept_7 = layers.Conv2D(d4, (7,7),strides=(1,1),padding='same',activation='relu')(incept_7_pool)
        incept_9 = layers.Conv2D(d4, (9,9),strides=(1,1),padding='same',activation='relu')(incept_9_pool)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
         
        data = layers.concatenate([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool], axis=3)

    
        parallel = keras.models.Model(inputs, data, name='parallel')     

        tower_1 = parallel(inp1)
        tower_2 = parallel(inp2)
        tower_3 = parallel(inp3)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_inception_all__depths__inception_vin_1_d1',str(d1),'_d2_',str(d2),'_d3_',str(d3),'_d4_',str(d4),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_inception_all__depths__inception_vin_1_d1',str(d1),'_d2_',str(d2),'_d3_',str(d3),'_d4_',str(d4),'.h5'])
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
    
        return model,model_name


'''model parallel tower -4'''
class model_par_parallel_inception_all__depths_min_max_inception_vin_1:
    
    def __init__(self,channels=17,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only
        
    def model_maker(self,d1=8,d2=8,d3=16,d4=16):
        inputs = keras.Input(shape=(200, 200, self.channels))

        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))    
       
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(inputs)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_9)
        
        all_output_max = layers.maximum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output_min =layers.minimum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output = layers.concatenate([all_output_max, all_output_min], axis=3)
        
        incept_3 = layers.Conv2D(d2, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d2, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d2, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d2, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
        
        all_output_max = layers.maximum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output_min =layers.minimum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output = layers.concatenate([all_output_max, all_output_min], axis=3)
        
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d3, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d3, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d3, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
        
        all_output_max = layers.maximum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output_min =layers.minimum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output = layers.concatenate([all_output_max, all_output_min], axis=3)
        
        incept_3 = layers.Conv2D(d4, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d4, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d4, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d4, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
         
        all_output_max = layers.maximum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output_min =layers.minimum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output = layers.concatenate([all_output_max, all_output_min], axis=3)
    
        parallel = keras.models.Model(inputs, all_output, name='parallel')     

        tower_1 = parallel(inp1)
        tower_2 = parallel(inp2)
        tower_3 = parallel(inp3)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_inception_all__depths__inception_vin_1_d1',str(d1),'_d2_',str(d2),'_d3_',str(d3),'_d4_',str(d4),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_inception_all__depths__inception_vin_1_d1',str(d1),'_d2_',str(d2),'_d3_',str(d3),'_d4_',str(d4),'.h5'])
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
    
        return model,model_name

    
'''model parallel tower -5'''
class model_par_inception_all__depths__inception_complic_vin_1:
    
    def __init__(self,channels=17,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only
        

    def model_maker(self,d1=8,d3=16):
                
        inputs = keras.Input(shape=(200, 200, self.channels))

        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))   
        
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(inputs)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_9)
        
        all_output = layers.concatenate([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool], axis=3)
    
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
          
        all_output_2 = layers.concatenate([incept_3, incept_5, incept_7,incept_9], axis=3)
        all_output=  layers.add([all_output,all_output_2])
        all_output = layers.MaxPooling2D(pool_size=(2,2))(all_output)
     
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d3, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d3, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d3, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
        
        all_output = layers.concatenate([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool], axis=3)
        
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d3, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d3, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d3, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
        
        all_output_4 = layers.concatenate([incept_3, incept_5, incept_7,incept_9], axis=3)
        all_output=  layers.add([all_output,all_output_4])
        all_output = layers.MaxPooling2D(pool_size=(2,2))(all_output)

        
        parallel = keras.models.Model(inputs, all_output, name='parallel')     

        tower_1 = parallel(inp1)
        tower_2 = parallel(inp2)
        tower_3 = parallel(inp3)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_inception_all__depths__inception_complic_vin_1_d1',str(d1),'_d3_',str(d3),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_inception_all__depths__inception_complic_vin_1_d1',str(d1),'_d3_',str(d3),'.h5'])

        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
    
        return model,model_name

'''model parallel tower -6'''
class model_par_parallel_inception_all__depths_min_max_inception_complic_vin_1:
    
    def __init__(self,channels=17,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only
        

    def model_maker(self,d1=8,d3=16):
                
        inputs = keras.Input(shape=(200, 200, self.channels))

        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))   
        
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(inputs)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_9)
        
        all_output_max = layers.maximum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output_min =layers.minimum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output = layers.concatenate([all_output_max, all_output_min], axis=3)
        
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d1, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d1, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d1, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
        
        all_output_2_max = layers.maximum([incept_3, incept_5, incept_7,incept_9]) 
        all_output_2_min =layers.minimum([incept_3, incept_5, incept_7,incept_9]) 
        all_output_2 = layers.concatenate([all_output_2_max, all_output_2_min], axis=3)
        
        all_output=  layers.add([all_output,all_output_2])
        all_output = layers.MaxPooling2D(pool_size=(2,2))(all_output)
         
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d3, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d3, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d3, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)
        incept_7_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_7)
        incept_9_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_9)
        
        all_output_max = layers.maximum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output_min =layers.minimum([incept_3_pool, incept_5_pool, incept_7_pool,incept_9_pool])
        all_output = layers.concatenate([all_output_max, all_output_min], axis=3)
        
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d3, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_7 = layers.Conv2D(d3, (7,7),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_9 = layers.Conv2D(d3, (9,9),strides=(1,1),padding='same',activation='relu')(all_output)
        
        all_output_2_max = layers.maximum([incept_3, incept_5, incept_7,incept_9]) 
        all_output_2_min =layers.minimum([incept_3, incept_5, incept_7,incept_9]) 
        all_output_2 = layers.concatenate([all_output_2_max, all_output_2_min], axis=3)
        
        all_output=  layers.add([all_output,all_output_2])
        all_output = layers.MaxPooling2D(pool_size=(2,2))(all_output)

        
        parallel = keras.models.Model(inputs, all_output, name='parallel')     

        tower_1 = parallel(inp1)
        tower_2 = parallel(inp2)
        tower_3 = parallel(inp3)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_parallel_inception_all__depths_min_max_inception_complic_vin_1_d1',str(d1),'_d3_',str(d3),'.h5'])
        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_parallel_inception_all__depths_min_max_inception_complic_vin_1_d1',str(d1),'_d3_',str(d3),'.h5'])

        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
    
        return model,model_name
    
class model_par_inception_w_o_addition_vin_exp:
    
    def __init__(self,channels=17,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only       

    def model_maker(self,d1=96,d2=32,k2=3,k3=3,k4=3):#or d1=64 (32 x 3)
        
        inputs = keras.Input(shape=(200, 200, self.channels))

        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))
        
        incept_3 = layers.Conv2D(d1, (3,3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d2, (5,5),strides=(1,1),padding='same',activation='relu')(inputs)
        data = layers.concatenate([incept_3, incept_5], axis=3)
        data = layers.MaxPooling2D(pool_size=(4, 4))(data)
        
        data = layers.Conv2D(d1+d2, (k2,k2),strides=(1,1),padding='same',activation='relu')(data)   
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
    
        data = layers.Conv2D(d1+d2, (k3,k3), activation='relu', padding='same')(data)
        '''to be removed'''
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        data = layers.Conv2D(64, (k4,k4), activation='relu', padding='same')(data)
        data = layers.MaxPooling2D(pool_size=(2, 2))(data)
        
        parallel = keras.models.Model(inputs, data, name='parallel')     

        tower_1 = parallel(inp1)
        tower_2 = parallel(inp2)
        tower_3 = parallel(inp3)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_par_inception_w_o_addition_vin_exp_d1_',str(d1),'_d2',str(d2),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_par_inception_w_o_addition_vin_exp_d1_',str(d1),'_d2_',str(d2),'_k2_',str(k2),'_k3_',str(k3),'_k4_',str(k4),'.h5'])

        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
        
        return model,model_name

class model_par_parallel_inception_all_k3x3_k_5_depths_min_max_inception_vin_1:
    
    def __init__(self,channels=17,tower_min_max_only=False):
        'Initialization'
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only
        
    def model_maker(self,d3=96,d5=32):
        inputs = keras.Input(shape=(200, 200, self.channels))

        inp1 = keras.Input(shape=(200, 200, self.channels))
        inp2 = keras.Input(shape=(200, 200, self.channels))
        inp3 = keras.Input(shape=(200, 200, self.channels))    
       
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation='relu')(inputs)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation='relu')(inputs)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(4, 4))(incept_5)

        all_output = layers.concatenate([incept_3_pool, incept_5_pool], axis=3)
        
        incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)

        all_output = layers.concatenate([incept_3_pool, incept_5_pool], axis=3)
        
        incept_3 = layers.Conv2D(2*d3, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(2*d5, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)

        all_output = layers.concatenate([incept_3_pool, incept_5_pool], axis=3)
        
        incept_3 = layers.Conv2D(2*d3, (3,3),strides=(1,1),padding='same',activation='relu')(all_output)
        incept_5 = layers.Conv2D(2*d5, (5,5),strides=(1,1),padding='same',activation='relu')(all_output)
        
        incept_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_3)
        incept_5_pool = layers.MaxPooling2D(pool_size=(2, 2))(incept_5)

        all_output = layers.concatenate([incept_3_pool, incept_5_pool], axis=3)
    
        parallel = keras.models.Model(inputs, all_output, name='parallel')     

        tower_1 = parallel(inp1)
        tower_2 = parallel(inp2)
        tower_3 = parallel(inp3)
        
        if self.tower_min_max_only:
            tower_max = layers.maximum([tower_1,tower_2,tower_3])
            tower_min=layers.minimum([tower_max, tower_2,tower_3])
            #tower_average=layers.average([tower_1, tower_2,tower_3])
            #tower_mul= layers.Multiply([tower_1, tower_2,tower_3])#since multiplication is expensive
            merged = layers.concatenate([tower_max, tower_min], axis=1)
            model_name = ''.join(['MIN_MAX_model_par_parallel_inception_all_k3x3_k_5_depths_min_max_inception_vin_1_d3_',str(d3),'_d5_',str(d5),'.h5'])

        else:
            merged = layers.concatenate([tower_1, tower_2, tower_3], axis=1)
            model_name = ''.join(['model_par_parallel_inception_all_k3x3_k_5_depths_min_max_inception_vin_1_d3_',str(d3),'_d5_',str(d5),'.h5'])
        
        merged = layers.Flatten()(merged)
        all_co = layers.Dense(100, activation='relu')(merged)
        all_co = layers.Dropout(0.5)(all_co)
        all_co = layers.Dense(50, activation='relu')(all_co)
        all_co= layers.Dropout(0.5)(all_co)
        outputs = layers.Dense(3, activation='softmax')(all_co)

        model = keras.models.Model(inputs=[inp1, inp2,inp3], outputs=outputs, name=model_name)
    
        return model,model_name    