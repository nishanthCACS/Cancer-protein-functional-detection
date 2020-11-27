# -*- coding: utf-8 -*-
"""
Created on %25-Feb-2019(12.29 P.m)

@author: %A.Nishanth C00294860
"""
import os
import csv

os.chdir('/')
os.chdir('C:/Users/nishy/Documents/Projects_UL/Continues BIBM/Results_for_R/ONGO_R')
files=os.listdir()
pdb_files=[]
for f in files:
#    print(f[-3:len(f)])
    if f[-3:len(f)]=='pdb':
        pdb_files.append(f)
#%%
c=0
for i in range(0,len(pdb_files)):
    f = open(pdb_files[i])
    csv_f = csv.reader(f)
    stack=[]
    for row in csv_f:
        for s in row:
            stack.append(s)
    
    if 'ONCO' in stack[0]:
        c=c+1
    else:
        print(stack[0])