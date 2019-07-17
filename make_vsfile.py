#!/usr/bin/env python3

"""
Make random velocity files for Nice sedimentary basin
"""

# Regular imports
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/flomin/Desktop/thesis/MyScripts/bash/randomfield/rand_code/')
from random2D import random2D
from random2D import make_vfile


# Velocity profile
vsi     = (180.,290.,200.,250.,300.,220.,290.)
vpi     = (440.,710.,489.,612.,734.,538.,710.)

# Main directory
main_dir = '/Users/flomin/Desktop/thesis/simulations/nice_bis/'

# folders
folders = ['a10','a20','a30','a50','a70','a100','a150']

# Directories
dirs = [ main_dir + i + '/' for i in folders ]

# Number of realisations
N = 10

# reference point
x0,z0 = 490.0 , -200.0

# Domain size (1140,1140)

# Create random fluctuations
rand_obj = [ random2D(d,n_real=N) for d in dirs ]

# Superpose velocities and write to text file
for i in range(N):
  for j in range(len(rand_obj)):
    param = { 'Z': rand_obj[j].m_ACF[i,:,:], 'vsi':vsi, 'vpi':vpi, 'x0':x0,
             'z0':z0, 'Nx':rand_obj[j]._Nx, 'Nz': rand_obj[j]._Nz,
             'dx':rand_obj[j]._dx, 'dz':rand_obj[j]._dz, 'ax':rand_obj[j]._ax,
             'az':rand_obj[j]._az,'eps':rand_obj[j]._eps,'n':i+1,
             'dirname':dirs[j] }
    make_vfile(**param)



