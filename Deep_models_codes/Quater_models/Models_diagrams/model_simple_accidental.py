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
model_name_save='model_accidental_eq_different_NN'

channels=17
#channels=21

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
p1=3
d1=32


k1=p1
k2=p1
k3=p1
k4=p1
      
def parallel(inputs,k1=3,k2=3,k3=3,k4=3,d1=32):

    xy = layers.Conv2D(d1, (k1,k1),strides=(1,1),padding='same',activation= activation)(inputs)
    block_1_xy_output = layers.MaxPooling2D(pool_size=(4, 4))(xy)
    
    xy = layers.Conv2D(d1, (k2,k2),strides=(1,1),padding='same',activation= activation)(block_1_xy_output)
    block_2_xy_output = layers.MaxPooling2D(pool_size=(2, 2))(xy)

    xy = layers.Conv2D(2*d1, (k3,k3), padding='same',activation= activation)(block_2_xy_output)
    xy = layers.MaxPooling2D(pool_size=(2, 2))(xy)

    xy = layers.Conv2D(2*d1, (k4,k4), padding='same',activation= activation)(xy)
    block_3_xy_output = layers.MaxPooling2D(pool_size=(2, 2))(xy)
    return block_3_xy_output

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
  
model_name =''.join(['model_accidental_p1_',str(p1),'_d1_',str(d1),'.h5'])
    
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

    