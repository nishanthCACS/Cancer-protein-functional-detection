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
model_name_save='model_par_inception_residual_mentor_quat'

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

inputs = keras.Input(shape=(128,128, channels))
 
incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= activation)(inputs)
incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= activation)(inputs)
incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= activation)(inputs)
incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(inputs)
incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)

all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
layer_1_pool = layers.MaxPooling2D(pool_size=(4, 4))(all_output)

#to reduce the depth representation
incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= activation)(layer_1_pool)
incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= activation)(layer_1_pool)

incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= activation)(layer_1_pool)
incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= activation)(incept_1_to_3)
incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= activation)(incept_1_to_5)
incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_1_pool)
incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)

all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)

layer_1_2_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation= activation)(layer_1_pool)
layer_21_residual = layers.add([layer_1_2_pool, all_output])

layer_2_pool = layers.MaxPooling2D(pool_size=(3, 3))(layer_21_residual)
#to reduce the depth representation
incept_1_to_3=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= activation)(layer_2_pool)
incept_1_to_5=layers.Conv2D(d_lay_1_to_2, (1,1),strides=(1,1),padding='same',activation= activation)(layer_2_pool)

incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= activation)(layer_2_pool)
incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= activation)(incept_1_to_3)
incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= activation)(incept_1_to_5)
incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_2_pool)
incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)

all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
layer_2_3_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation= activation)(layer_2_pool)
layer_32_residual = layers.add([layer_2_3_pool, all_output])

layer_3_pool = layers.MaxPooling2D(pool_size=(3, 3))(layer_32_residual)
#to reduce the depth representation
incept_1_to_3=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation= activation)(layer_3_pool)
incept_1_to_5=layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation= activation)(layer_3_pool)

incept_1 = layers.Conv2D(d1, (1,1),strides=(1,1),padding='same',activation= activation)(layer_3_pool)
incept_3 = layers.Conv2D(d3, (3,3),strides=(1,1),padding='same',activation= activation)(incept_1_to_3)
incept_5 = layers.Conv2D(d5, (5,5),strides=(1,1),padding='same',activation= activation)(incept_1_to_5)
incept_max_pool= layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(layer_3_pool)
incept_max_pool_depth=layers.Conv2D(d_max, (1,1),strides=(1,1),padding='same',activation='relu')(incept_max_pool)

all_output = layers.concatenate([incept_1, incept_3,incept_5,incept_max_pool_depth], axis=3)
layer_3_4_pool=layers.Conv2D(d1+d3+d5+d_max, (1,1),strides=(1,1),padding='same',activation= activation)(layer_3_pool)
layer_34_residual = layers.add([layer_3_4_pool, all_output])

layer_4_pool = layers.MaxPooling2D(pool_size=(2, 2))(layer_34_residual)
incept_1_to_final =layers.Conv2D(d_lay_3_to_4, (1,1),strides=(1,1),padding='same',activation= activation)(layer_4_pool)
parallel = keras.models.Model(inputs, incept_1_to_final, name='parallel')  

os.chdir('/')
os.chdir(summery_saving_dir)  
model_feature_extract_name=''.join([model_name_save,'_projection_feature_extracter'])
with open(''.join([model_feature_extract_name,'.txt']), 'w') as f:
    with redirect_stdout(f):
        parallel.summary() 

f.close()
os.chdir('/')
os.chdir(image_saving_dir)  
keras.utils.plot_model(parallel,''.join([model_feature_extract_name,'.png']) )
#%%
#parallel = keras.models.Model(inputs,all_output, name='parallel') 
inp_q0_x1 = keras.Input(shape=(square_height,square_width, channels))
inp_q0_x2 = keras.Input(shape=(square_height, square_width, channels))
inp_q0_x3 = keras.Input(shape=(square_height, square_width, channels))

inp_q1_x1 = keras.Input(shape=(square_height, square_width, channels))
inp_q1_x2 = keras.Input(shape=(square_height, square_width, channels))
inp_q1_x3 = keras.Input(shape=(square_height, square_width, channels))
   		
inp_q2_x1 = keras.Input(shape=(square_height, square_width, channels))
inp_q2_x2 = keras.Input(shape=(square_height, square_width, channels))
inp_q2_x3 = keras.Input(shape=(square_height, square_width, channels))
		
inp_q3_x1 = keras.Input(shape=(square_height, square_width, channels))
inp_q3_x2 = keras.Input(shape=(square_height, square_width, channels))
inp_q3_x3 = keras.Input(shape=(square_height, square_width, channels))
		
inp_q4_x1 = keras.Input(shape=(square_height, square_width, channels))
inp_q4_x2 = keras.Input(shape=(square_height, square_width, channels))
inp_q4_x3 = keras.Input(shape=(square_height, square_width, channels))

inp_q5_x1 = keras.Input(shape=(square_height, square_width, channels))
inp_q5_x2 = keras.Input(shape=(square_height, square_width, channels))
inp_q5_x3 = keras.Input(shape=(square_height, square_width, channels))
		
inp_q6_x1 = keras.Input(shape=(square_height, square_width, channels))
inp_q6_x2 = keras.Input(shape=(square_height, square_width, channels))
inp_q6_x3 = keras.Input(shape=(square_height, square_width, channels))
		
inp_q7_x1 = keras.Input(shape=(square_height, square_width, channels))
inp_q7_x2 = keras.Input(shape=(square_height, square_width, channels))
inp_q7_x3 = keras.Input(shape=(square_height, square_width, channels))


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
model_name = ''.join(['model_par_inception_residual_mentor_vin_1_quat_',activation,'_f_inc_',str(f_inc),'_f_d_',str(f_d),'.h5'])

merged = layers.Flatten()(merged)
all_co = layers.Dense(100, activation= activation)(merged)
all_co = layers.Dropout(0.5)(all_co)
all_co = layers.Dense(50, activation= activation)(all_co)
all_co= layers.Dropout(0.5)(all_co)
outputs = layers.Dense(3, activation='softmax')(all_co)

model = keras.models.Model(inputs=[inp_q0_x1,inp_q0_x2,inp_q0_x3 ,inp_q1_x1,inp_q1_x2,inp_q1_x3,inp_q2_x1,inp_q2_x2,inp_q2_x3,inp_q3_x1,inp_q3_x2,inp_q3_x3,inp_q4_x1,inp_q4_x2,inp_q4_x3,inp_q5_x1,inp_q5_x2,inp_q5_x3,inp_q6_x1,inp_q6_x2,inp_q6_x3,inp_q7_x1,inp_q7_x2,inp_q7_x3], outputs=outputs, name=model_name)

os.chdir('/')
os.chdir(summery_saving_dir)  
model_main_name=''.join(['main_',model_name_save])
with open(''.join([model_main_name,'.txt']), 'w') as f:
    with redirect_stdout(f):
        model.summary() 

f.close()
os.chdir('/')
os.chdir(image_saving_dir)  
keras.utils.plot_model(model,''.join([model_main_name,'.png']) )

    