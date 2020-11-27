# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %A.Nishanth C00294860
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

class model_par_brain_inception_only_vin_1:

    def __init__(self, channels=17,tower_min_max_only=False,activation='relu'):
        '''Initialization
        
        activation can be 'selu','swish','nishy_vin1'
        '''
        self.activation = activation
        self.channels = channels
        self.tower_min_max_only = tower_min_max_only

    def layer_1_reducer(self,lay_1_all_output,d_lay_1_to_2):
        
        layer_1_pool = layers.MaxPooling2D(pool_size=(4, 4))(all_output)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_1_pool)
        return incept_1_to_3,incept_1_to_5,layer_1_pool

    def layer_2_reducer(self,lay_2_all_output,d_lay_1_to_2):
        
        layer_2_pool = layers.MaxPooling2D(pool_size=(3, 3))(lay_2_all_output)       
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_2_pool)
        return incept_1_to_3,incept_1_to_5,layer_2_pool

    def layer_3_reducer(self,lay_3_all_output,d_lay_3_to_4):
        
        layer_3_pool = layers.MaxPooling2D(pool_size=(2, 2))(lay_3_all_output)
        #to reduce the depth representation
        incept_1_to_3=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        incept_1_to_5=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_3_pool)
        return incept_1_to_3,incept_1_to_5,layer_3_pool

    def layer_4_final(self,lay_4_all_output,d_lay_3_to_4):
        layer_4_pool = layers.MaxPooling2D(pool_size=(2, 2))(all_output)
        incept_1_to_final =layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=self.activation)(layer_4_pool)
        return incept_1_to_final

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
       
        lay_1_inp1_incept_1_to_3,lay_1_inp1_incept_1_to_5,inp1_lay_1_pool = self.layer_1_reducer(inp_1_lay_1_all_output,d_lay_1_to_2)
        lay_1_inp2_incept_1_to_3,lay_1_inp2_incept_1_to_5,inp2_lay_1_pool = self.layer_1_reducer(inp_2_lay_1_all_output,d_lay_1_to_2)

        '''place the size of NN layer_2 inputs'''
        lay_1_inp_incept_1_to_3 = keras.Input(shape=(200, 200, self.channels))
        lay_1_inp_incept_1_to_5 = keras.Input(shape=(200, 200, self.channels))
        inp_lay_1_pool = keras.Input(shape=(200, 200, self.channels))
        
        lay_2_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_1_pool)
        lay_2_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_1_inp_incept_1_to_3)
        lay_2_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_1_inp_incept_1_to_5)
        lay_2_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_1_pool)
        lay_2_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_2_incept_max_pool)
        
        lay_2_all_output = layers.concatenate([lay_2_incept_1, lay_2_incept_3,lay_2_incept_5,lay_2_incept_max_pool_depth], axis=3)
        '''layer 2 general network'''    
        layer_2_INCEPT_Net = keras.models.Model(inputs=[lay_1_inp_incept_1_to_3,lay_1_inp_incept_1_to_5,inp_lay_1_pool], lay_2_all_output, name='layer_2_INCEPT')     

        '''Applying layer_2 in projections'''
        #layer1 output of projection 1
        inp_1_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp1_incept_1_to_3,lay_1_inp1_incept_1_to_5,inp1_lay_1_pool]) 
        inp_2_lay_2_all_output = layer_2_INCEPT_Net([lay_1_inp2_incept_1_to_3,lay_1_inp2_incept_1_to_5,inp2_lay_1_pool]) 
        
        #layer2 output of projection 1
        lay_2_inp1_incept_1_to_3,lay_2_inp1_incept_1_to_5,inp1_lay_2_pool = self.layer_2_reducer(inp_1_lay_2_all_output,d_lay_1_to_2)
        lay_2_inp2_incept_1_to_3,lay_2_inp2_incept_1_to_5,inp2_lay_2_pool = self.layer_2_reducer(inp_2_lay_2_all_output,d_lay_1_to_2)

        '''place the size of NN layer_3 inputs'''
        lay_2_inp_incept_1_to_3 = keras.Input(shape=(200, 200, self.channels))
        lay_2_inp_incept_1_to_5 = keras.Input(shape=(200, 200, self.channels))
        inp_lay_2_pool = keras.Input(shape=(200, 200, self.channels))

        lay_3_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_2_pool)
        lay_3_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_2_inp_incept_1_to_3)
        lay_3_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_2_inp_incept_1_to_5)
        lay_3_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_2_pool)
        lay_3_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_3_incept_max_pool)
        
        lay_3_all_output = layers.concatenate([lay_3_incept_1, lay_3_incept_3,lay_3_incept_5,lay_3_incept_max_pool_depth], axis=3)
        '''layer 3 general network'''    
        layer_3_INCEPT_Net = keras.models.Model(inputs=[lay_2_inp_incept_1_to_3,lay_2_inp_incept_1_to_5,inp_lay_2_pool], lay_3_all_output, name='layer_3_INCEPT')     

        '''Applying layer_3 in projections'''
        #layer3 output of projection 1
        inp_1_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp1_incept_1_to_3,lay_2_inp1_incept_2_to_5,inp1_lay_2_pool]) 
        inp_2_lay_3_all_output = layer_3_INCEPT_Net([lay_2_inp2_incept_1_to_3,lay_2_inp2_incept_2_to_5,inp2_lay_2_pool]) 
        
        #layer3 output of projection 1
        lay_3_inp1_incept_1_to_3,lay_3_inp1_incept_1_to_5,inp1_lay_3_pool = self.layer_3_reducer(inp_1_lay_3_all_output,d_lay_3_to_4)
        lay_3_inp2_incept_1_to_3,lay_3_inp2_incept_1_to_5,inp2_lay_3_pool = self.layer_3_reducer(inp_2_lay_3_all_output,d_lay_3_to_4)

        '''place the size of NN layer_4 inputs'''
        lay_3_inp_incept_1_to_3 = keras.Input(shape=(200, 200, self.channels))
        lay_3_inp_incept_1_to_5 = keras.Input(shape=(200, 200, self.channels))
        inp_lay_3_pool = keras.Input(shape=(200, 200, self.channels))
        
        
        lay_4_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=self.activation)(inp_lay_3_pool)
        lay_4_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=self.activation)(lay_3_inp_incept_1_to_3)
        lay_4_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=self.activation)(lay_3_inp_incept_1_to_5)
        lay_4_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_3_pool)
        lay_4_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_4_incept_max_pool)
        
        lay_4_all_output = layers.concatenate([lay_4_incept_1, lay_4_incept_3,lay_4_incept_5,lay_4_incept_max_pool_depth], axis=3)
    
        '''layer 3 general network'''    
        layer_4_INCEPT_Net = keras.models.Model(inputs=[lay_3_inp_incept_1_to_3,lay_3_inp_incept_1_to_5,inp_lay_3_pool], lay_4_all_output, name='layer_4_INCEPT')     
        '''Applying layer_4 in projections'''
        inp_1_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp1_incept_1_to_3,lay_3_inp1_incept_1_to_5,inp1_lay_3_pool]) 
        inp_2_lay_4_all_output = layer_4_INCEPT_Net([lay_3_inp2_incept_1_to_3,lay_3_inp2_incept_1_to_5,inp2_lay_3_pool]) 
        
        
        #parallel = keras.models.Model(inputs,all_output, name='parallel') 
        tower_1 = self.incept_1_to_final(inp_1_lay_4_all_output,d_lay_3_to_4)
        tower_2 = self.incept_1_to_final(inp_2_lay_4_all_output,d_lay_3_to_4)

        
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
    
'''Add residual as well'''