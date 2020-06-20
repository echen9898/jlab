import os
import numpy as np
import scipy as sp
from scipy import *
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import math


#################################### DATA TABLE ####################################

# format: dirs['./80G'] = [[window], parameters_0, [[volts],[counts]]]
dirs = {
    './80G' : [[620, 670], [50, 2.7, 0.2]],
    './85G' : [[700, 760], [100, 3, 0.2]],
    './90G' : [[800, 850], [70, 3.5, 0.3]],
    './95G' : [[850, 950], [110, 3.6, 0.3]],
    './100G' : [[1000, 1200], [300, 4, 0.3]],
    './105G' : [[1000, 1200], [120, 4.17, 0.4]],
    './110G' : [[1000, 1200], [160, 4.6, 0.2]]
}


############################### COMPILE MAX COUNTS ###############################

def list_peak_counts(dir_name, width=(0, 2048)):

    os.chdir(dir_name)

    filenames = list()
    for file in os.listdir():
        if file != '.DS_Store':
            filenames.append(file) 

    counts = dict()
    for name in filenames: # for each voltage

        readout = list()
        bins = list()

        with open(name) as file: 
            rows = file.readlines()

            bins = list()
            for r in rows:
                try:
                    bins.append(int(r))
                except ValueError:
                    pass

            peak = bins[width[0]:width[1]] # take portion of data in desired interval
            volt = name[:3]
            counts[volt] = sum(peak) # register the counts in this interval

    results = [[], []] # [[voltages], [counts]]
    for c in counts:
        results[0].append(float(c))
        results[1].append(counts[c])

    os.chdir('..')

    return results


############################### COMPILE MAX COUNTS ###############################

for directory in dirs: # make a list of count lists
    dirs[directory].append(list_peak_counts(directory, dirs[directory][0]))

############################### COMPUTE GAUSSIAN FIT ###############################

def func(x, a, mu, std): # evaluate function for an x value
    func = a * np.exp(-(x-mu)**2/(2*std**2)) # single gaussian
    return func

def chi_square(observed, volts, a, mu, std):
    chi_sq = 0
    for i in range(len(observed)):
        expected = func(volts[i], a, mu, std)
        chi_sq += (observed[i] - expected)**2/observed[i]
    return chi_sq

field = './110G' # select the field you'd like to calculate central voltage for

x = dirs[field][2][0] # voltages
x_continuous = arange(min(x), max(x), 0.01) # x axis for fit calculation
y = dirs[field][2][1] # counts
yerrs = sqrt(array(y)) # poisson random error on counts

params_0 = array(dirs[field][1]) # initialize parameters
fit = curve_fit(func, x, y, params_0, yerrs, maxfev=500000) # find the best fit with curve_fit

# ----- These are the best fit parameters returned -----
a, mu, std = fit[0]
chi2 = chi_square(y, x, a, mu, std)
n = len(y) # number of observed counts
df = n - len(fit[0]) # degrees of freedom
print('MEAN: ', mu) # mean
print('STD: ', std) # standard deviation
print('STD_M: ', std/sqrt(len(x))) # standard deviation on mean
print('CHI_SQ: ', chi2) # chi square
print('DEGREES: ', df) # degrees of freedom
print('CHI_SQ_PDF: ', chi2/df) # chi square pdf
print('P(X2): ', 1 - sp.stats.chi2.cdf(chi2, df)) # p-value

std_means = [0.13415442018201987, 
            0.12907190259404439, 
            0.12086974861576942, 
            0.11662153371238322, 
            0.1193046736935328, 
            0.14327468766968912, 
            0.1516384522385753] # central voltage uncertainties for each field (80 t0 110)

# ----- Plot results -----
plt.scatter(x, y, s=15, marker=',', color='black') # plot the histogram
plt.errorbar(x, y, yerrs, color='black', capsize=1.5, ls='none')
plt.plot(x_continuous, func(x_continuous, a, mu, std), color='red') # plot the fit

plt.title('Central Voltage Determination', fontsize=18)
plt.xlabel('Voltage (kV)', fontsize=15)
plt.ylabel('Total Counts', fontsize=15)

plt.show()



