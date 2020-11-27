#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:08:36 2020

@author: c00294860
"""
from tensorflow import keras
from tensorflow.keras import layers


'''Defining the swish -activation function'''
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation

from contextlib import redirect_stdout

import os

main_ch_dir = '/home/C00294860/Documents/BIBM/Model_with_parallel_and_projections/results/'
model_save_name='model_brain_incept_RESIDUAL_quat_vin_1'

#channels=17
channels=21

image_saving_dir=''.join([main_ch_dir,str(channels),'_channels/results_images/'])
summery_saving_dir=''.join([main_ch_dir,str(channels),'_channels/'])


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
 


def layer_1_reducer(lay_1_all_output,d_lay_1_to_2,m_p=4):
    
    layer_1_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_1_all_output)
    #to reduce the depth representation
    incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=activation)(layer_1_pool)
    incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=activation)(layer_1_pool)
    return incept_1_to_3,incept_1_to_5,layer_1_pool

def layer_2_reducer(lay_2_all_output,d_lay_1_to_2,m_p=3):
    
    layer_2_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_2_all_output)       
    #to reduce the depth representation
    incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=activation)(layer_2_pool)
    incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation=activation)(layer_2_pool)
    return incept_1_to_3,incept_1_to_5,layer_2_pool

def layer_3_reducer(lay_3_all_output,d_lay_3_to_4,m_p=2):
    
    layer_3_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_3_all_output)
    #to reduce the depth representation
    incept_1_to_3=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=activation)(layer_3_pool)
    incept_1_to_5=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=activation)(layer_3_pool)
    return incept_1_to_3,incept_1_to_5,layer_3_pool

def layer_4_final(lay_4_all_output,d_lay_3_to_4,m_p=2):
    layer_4_pool = layers.MaxPooling2D(pool_size=(m_p, m_p))(lay_4_all_output)
    incept_1_to_final =layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation=activation)(layer_4_pool)
    return incept_1_to_final


activation = 'swish'
square_height=128
square_width=128


f_inc=1
f_d=1

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
inputs = keras.Input(shape=(128,128, channels))
 
'''Inputs intialisation'''
inp_q0_x1 = keras.Input(shape=(square_height,square_width, channels))
inp_q0_x2 = keras.Input(shape=(square_height, square_width, channels))

lay_1_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=activation)(inputs)
lay_1_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=activation)(inputs)
lay_1_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=activation)(inputs)
lay_1_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
lay_1_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_1_incept_max_pool)

lay_1_all_output = layers.concatenate([lay_1_incept_1, lay_1_incept_3,lay_1_incept_5,lay_1_incept_max_pool_depth], axis=3)
'''layer 1 general network'''    
layer_1_INCEPT_Net = keras.models.Model(inputs, lay_1_all_output, name='layer_1_INCEPT')     
        

os.chdir('/')
os.chdir(summery_saving_dir)  
model_feature_extract_name=''.join([model_save_name,'_projection_feature_extracter'])

with open(''.join([model_feature_extract_name,'.txt']), 'w') as f:
    with redirect_stdout(f):
        print('')
        print('layer_1_INCEPT_Net')        
        print('')
        layer_1_INCEPT_Net.summary() 
f.close()

os.chdir('/')
os.chdir(image_saving_dir)  
keras.utils.plot_model(layer_1_INCEPT_Net,''.join([model_feature_extract_name,'_layer_1_INCEPT_Net.png']) )
#%
'''place the size of NN layer_2 inputs'''
#since the stride is 1 ans same padding these equation works
lay_1_height=int(square_height/m_p_l1)
lay_1_width = int(square_width/m_p_l1)


lay_1_inp_incept_1_to_3 = keras.Input(shape=(lay_1_height,lay_1_width, d_lay_1_to_2))
lay_1_inp_incept_1_to_5 = keras.Input(shape=(lay_1_height,lay_1_width, d_lay_1_to_2))
inp_lay_1_pool = keras.Input(shape=(lay_1_height,lay_1_width,d1+d3+d5+d_max))

lay_2_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=activation)(inp_lay_1_pool)
lay_2_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=activation)(lay_1_inp_incept_1_to_3)
lay_2_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=activation)(lay_1_inp_incept_1_to_5)
lay_2_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_1_pool)
lay_2_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_2_incept_max_pool)

lay_2_all_output = layers.concatenate([lay_2_incept_1, lay_2_incept_3,lay_2_incept_5,lay_2_incept_max_pool_depth], axis=3)
'''layer 2 general network'''    
layer_2_INCEPT_Net = keras.models.Model(inputs=[lay_1_inp_incept_1_to_3,lay_1_inp_incept_1_to_5,inp_lay_1_pool],outputs= lay_2_all_output, name='layer_2_INCEPT')     
os.chdir('/')
os.chdir(summery_saving_dir) 
with open(''.join([model_feature_extract_name,'.txt']), 'a') as f:
    with redirect_stdout(f):
        print('')
        print('layer_2_INCEPT_Net')        
        print('')
        layer_2_INCEPT_Net.summary() 
f.close()
os.chdir('/')
os.chdir(image_saving_dir)  
keras.utils.plot_model(layer_2_INCEPT_Net,''.join([model_feature_extract_name,'_layer_2_INCEPT_Net.png']) )


'''place the size of NN layer_3 inputs'''
#since the stride is 1 ans same padding these equation works
lay_2_height=int(lay_1_height/m_p_l2)
lay_2_width = int(lay_1_height/m_p_l2)

lay_2_inp_incept_1_to_3 = keras.Input(shape=(lay_2_height,lay_1_width, d_lay_1_to_2))
lay_2_inp_incept_1_to_5 = keras.Input(shape=(lay_2_width,lay_1_width, d_lay_1_to_2))      
inp_lay_2_pool = keras.Input(shape=(lay_2_height,lay_1_width,d1+d3+d5+d_max))

lay_3_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=activation)(inp_lay_2_pool)
lay_3_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=activation)(lay_2_inp_incept_1_to_3)
lay_3_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=activation)(lay_2_inp_incept_1_to_5)
lay_3_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_2_pool)
lay_3_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_3_incept_max_pool)

lay_3_all_output = layers.concatenate([lay_3_incept_1, lay_3_incept_3,lay_3_incept_5,lay_3_incept_max_pool_depth], axis=3)
'''layer 3 general network'''    
layer_3_INCEPT_Net = keras.models.Model(inputs=[lay_2_inp_incept_1_to_3,lay_2_inp_incept_1_to_5,inp_lay_2_pool],outputs= lay_3_all_output, name='layer_3_INCEPT')     
os.chdir('/')
os.chdir(summery_saving_dir) 
with open(''.join([model_feature_extract_name,'.txt']), 'a') as f:
    with redirect_stdout(f):
        print('')
        print('layer_3_INCEPT_Net')        
        print('')
        layer_3_INCEPT_Net.summary() 
f.close()
os.chdir('/')
os.chdir(image_saving_dir)  
keras.utils.plot_model(layer_3_INCEPT_Net,''.join([model_feature_extract_name,'_layer_3_INCEPT_Net.png']) )


'''place the size of NN layer_4 inputs'''
#since the stride is 1 ans same padding these equation works
lay_3_height=int(lay_2_height/m_p_l3)
lay_3_width = int(lay_2_height/m_p_l3)

lay_3_inp_incept_1_to_3 = keras.Input(shape=(lay_3_height, lay_3_width,d_lay_3_to_4))
lay_3_inp_incept_1_to_5 = keras.Input(shape=(lay_3_height, lay_3_width,d_lay_3_to_4))
inp_lay_3_pool = keras.Input(shape=(lay_3_height, lay_3_width,d1+d3+d5+d_max))       

lay_4_incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation=activation)(inp_lay_3_pool)
lay_4_incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation=activation)(lay_3_inp_incept_1_to_3)
lay_4_incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation=activation)(lay_3_inp_incept_1_to_5)
lay_4_incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inp_lay_3_pool)
lay_4_incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(lay_4_incept_max_pool)

lay_4_all_output = layers.concatenate([lay_4_incept_1, lay_4_incept_3,lay_4_incept_5,lay_4_incept_max_pool_depth], axis=3)

'''layer 3 general network'''    
layer_4_INCEPT_Net = keras.models.Model(inputs=[lay_3_inp_incept_1_to_3,lay_3_inp_incept_1_to_5,inp_lay_3_pool], outputs=lay_4_all_output, name='layer_4_INCEPT')     
os.chdir('/')
os.chdir(summery_saving_dir) 
with open(''.join([model_feature_extract_name,'.txt']), 'a') as f:
    with redirect_stdout(f):
        print('')
        print('layer_4_INCEPT_Net')        
        print('')
        layer_4_INCEPT_Net.summary() 
f.close()
os.chdir('/')
os.chdir(image_saving_dir)  
keras.utils.plot_model(layer_4_INCEPT_Net,''.join([model_feature_extract_name,'_layer_4_INCEPT_Net.png']) )
#%%  
'''Applying layer_1 in projections'''
#layer1 output of projection 1
inp_q0_x1_lay_1_all_output = layer_1_INCEPT_Net(inp_q0_x1) 
inp_q0_x2_lay_1_all_output = layer_1_INCEPT_Net(inp_q0_x2) 


lay_1_inp_q0_x1_incept_1_to_3,lay_1_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_1_pool = layer_1_reducer(inp_q0_x1_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)
lay_1_inp_q0_x2_incept_1_to_3,lay_1_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_1_pool = layer_1_reducer(inp_q0_x2_lay_1_all_output,d_lay_1_to_2,m_p=m_p_l1)

'''Applying layer_2 in projections'''
#layer1 output of projection 1
inp_q0_x_1_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q0_x1_incept_1_to_3,lay_1_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_1_pool]) 
inp_q0_x_2_lay_2_incept_all_output = layer_2_INCEPT_Net([lay_1_inp_q0_x2_incept_1_to_3,lay_1_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_1_pool]) 

#make residUAL NETWORK
inp_q0_x_1_lay_2_all_output= layers.add([inp_q0_x1_lay_1_pool, inp_q0_x_1_lay_2_incept_all_output])
inp_q0_x_2_lay_2_all_output= layers.add([inp_q0_x2_lay_1_pool, inp_q0_x_2_lay_2_incept_all_output])

#layer2 output of projection 1
lay_2_inp_q0_x1_incept_1_to_3,lay_2_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_2_pool = layer_2_reducer(inp_q0_x_1_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)
lay_2_inp_q0_x2_incept_1_to_3,lay_2_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_2_pool = layer_2_reducer(inp_q0_x_2_lay_2_all_output,d_lay_1_to_2,m_p=m_p_l2)


'''Applying layer_3 in projections'''
#layer3 output of projection 1
inp_q0_x_1_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q0_x1_incept_1_to_3,lay_2_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_2_pool]) 
inp_q0_x_2_lay_3_incept_all_output = layer_3_INCEPT_Net([lay_2_inp_q0_x2_incept_1_to_3,lay_2_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_2_pool]) 

  
#make residUAL NETWORK
inp_q0_x_1_lay_3_all_output= layers.add([inp_q0_x1_lay_2_pool, inp_q0_x_1_lay_3_incept_all_output])
inp_q0_x_2_lay_3_all_output= layers.add([inp_q0_x2_lay_2_pool, inp_q0_x_2_lay_3_incept_all_output])

     
#layer3 output of projection 1
lay_3_inp_q0_x1_incept_1_to_3,lay_3_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_3_pool = layer_3_reducer(inp_q0_x_1_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)
lay_3_inp_q0_x2_incept_1_to_3,lay_3_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_3_pool = layer_3_reducer(inp_q0_x_2_lay_3_all_output,d_lay_3_to_4,m_p=m_p_l3)

'''Applying layer_4 in projections'''
inp_q0_x_1_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q0_x1_incept_1_to_3,lay_3_inp_q0_x1_incept_1_to_5,inp_q0_x1_lay_3_pool]) 
inp_q0_x_2_lay_4_incept_all_output = layer_4_INCEPT_Net([lay_3_inp_q0_x2_incept_1_to_3,lay_3_inp_q0_x2_incept_1_to_5,inp_q0_x2_lay_3_pool]) 

#make residUAL NETWORK
inp_q0_x_1_lay_4_all_output= layers.add([inp_q0_x1_lay_3_pool, inp_q0_x_1_lay_4_incept_all_output])
inp_q0_x_2_lay_4_all_output= layers.add([inp_q0_x2_lay_3_pool, inp_q0_x_2_lay_4_incept_all_output])

#parallel = keras.models.Model(inputs,all_output, name='parallel') 
tower_q0_x1 = layer_4_final(inp_q0_x_1_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)
tower_q0_x2 = layer_4_final(inp_q0_x_2_lay_4_all_output,d_lay_3_to_4,m_p=m_p_l4)

              
merged = layers.concatenate([tower_q0_x1, tower_q0_x2], axis=1)

merged = layers.Flatten()(merged)
all_co = layers.Dense(100, activation=activation)(merged)
all_co = layers.Dropout(0.5)(all_co)
all_co = layers.Dense(50, activation=activation)(all_co)
all_co= layers.Dropout(0.5)(all_co)
outputs = layers.Dense(3, activation='softmax')(all_co)

model_name = ''.join(['model_brain_inception_Residual_mentor_vin_1_paper_1_act_',activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])
model = keras.models.Model(inputs=[inp_q0_x1,inp_q0_x2], outputs=outputs, name=model_name)
        
#os.chdir('/')
#os.chdir(summery_saving_dir)  
model_main_name=''.join(['main_',model_save_name])
#with open(''.join([model_main_name,'.txt']), 'w') as f:
#    with redirect_stdout(f):
#        model.summary() 

f.close()
os.chdir('/')
os.chdir(image_saving_dir)  
keras.utils.plot_model(model,''.join([model_main_name,'.png']) )

    