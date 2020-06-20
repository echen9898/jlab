import os
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy import *
from scipy import stats
from scipy.optimize import leastsq
from scipy.optimize import curve_fit


################################# READ DATA #################################

metals = dict() # {element : [ [0 degree], [20 degree] ]}

element_dirs = list() 
for file in os.listdir():
    if file not in ['.DS_Store', 'process.py']:
        element_dirs.append(file) 

for elem in element_dirs: 
    
    os.chdir(elem) 

    total_counts_0 = zeros((2048,))
    total_counts_20 = zeros((2048,))

    files_0 = list() # all 0 degree spectra
    files_20 = list() # all 20 degree spectra
    for file in os.listdir():
        if file not in ['.DS_Store']:
            if file[:1] == '0': # 0 degree spectrum
                files_0.append(file)
            if file[:1] == '2': # 20 degree spectrum
                files_20.append(file) 

    for n in files_0: # get counts for each 0 degree spectra, and add to total array

        counts = list()
        with open(n) as file:
            rows = file.readlines()
            for r in rows:
                try:
                    counts.append(int(r))
                except ValueError:
                    pass

        total_counts_0 += array(counts[:-5]) # shave off last 5 numerica settings (settings not data)

    for n in files_20: # get counts for each 0 degree spectra, and add to total array
        counts = list()
        with open(n) as file:
            rows = file.readlines()
            for r in rows:
                try:
                    counts.append(int(r))
                except ValueError:
                    pass
        
        total_counts_20 += array(counts[:-5]) # shave off last 5 numerica settings (settings not data)
    
    metals[elem] = [total_counts_0, total_counts_20] # summed spectra for each angle
    os.chdir('..')


################################# Z DEPENDENCE #################################

plt.scatter(list(range(0, 2048)), metals['Fe'][1], s=2, color='black')
plt.show()










