import os
import numpy as np
import scipy as sp
from scipy import *
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import math

#----- constants -----#
# E0 = 5.486 # MeV -> original energy of beam before passing through the foil
E0 = 4.580 # MeV -> attenuated energy beam due to gold film over source

#----- get filenames -----#
files = list()
for file in os.listdir():
    if file not in {'.DS_Store', 'process.py'}:
        files.append(file) 

#----- read counts -----#
layers = dict() # {layer : [counts, fit p0, window, fit params]} -> MCA counts for each thickness

for name in files: # for each gold foil thickness (# layers)
    counts = list() 
    with open(name) as file:
        rows = file.readlines()
        for r in rows:
            counts.append(int(r))
    
    layers[int(name[:1])] = [counts]

#----- gaussian fit to each layer -----#

# add initial fit parameters to the layers dictionary
layers[0].append([78, 1605, 50])
layers[1].append([108, 1335, 80])
layers[2].append([70, 1147, 90])
layers[3].append([22, 677, 100])

# add window over which to perform gaussian fit
layers[0].append([1479, 1715])
layers[1].append([1180, 1477])
layers[2].append([950, 1320])
layers[3].append([412, 885])

def gaussian(x, a, mu, std): # gaussian fit function
    func = a * np.exp(-(x-mu)**2/(2*std**2)) 
    return func

def chi_square(observed, bins, a, mu, std):
    chi_sq = 0
    for i in range(len(observed)):
        expected = gaussian(bins[i], a, mu, std)
        chi_sq += (observed[i] - expected)**2/observed[i]
    return chi_sq

peak_bins = dict() # {layer : mean} -> bin corresponding to the energy peak for each thickness
x = list(range(0, 2026)) # number of bins in each .txt file

for l in layers: # find the central peak bin for each layer
    left, right = layers[l][2] # window bounds over which to perform fit

    y = array(layers[l][0][left:right]) # counts for layer l
    y_errs = sqrt(y) # poison variable error on each bin count

    p0 = array(layers[l][1]) # amplitude, mean, std
    fit = curve_fit(gaussian, x[left:right], y, p0, y_errs, maxfev=500000) # find the best fit with curve_fit

    a, mu, std = fit[0]
    layers[l].append(fit[0])
    peak_bins[l] = mu

#----- chi2 and results -----#
energies = {0 : E0} # {layer : energy} -> calculated using ratio of peak bins
energy_errs = {0 : 0} # {layer : energy uncertainty} -> calculated from mean uncertainty

l_0, r_0 = layers[0][2]
dc0 = peak_bins[0]/sqrt(sum(layers[0][0][l_0:r_0])) # uncertainty on c0

for l in layers: # print stats for each layer
    left, right = layers[l][2]
    a, mu, std = layers[l][3]

    c0 = peak_bins[0]
    c1 = peak_bins[l]
    dc1 = peak_bins[l]/sqrt(sum(layers[l][0][left:right])) # uncertainty on c1

    if l != 0:
        energy = (c1/c0) * energies[0]
        uncertainty = energy * sqrt( (dc1/c1)**2 + (dc0/c0)**2 )

        energies[l] = energy
        energy_errs[l] = uncertainty

    y_window = layers[l][0][left:right]

    chi2 =chi_square(y_window, x[left:right], a, mu, std) # perform chi2 over counts windw
    n = len(y_window) # number of points in window
    df = n-3 # 3 parameters in gaussian fit

    print('LAYER: ', l)
    print('BIN: ', peak_bins[l])
    print('COUNTS: ', n)
    print('BIN_ERROR: ', peak_bins[l]/sqrt(sum(y_window))) 
    print('ENERGY: ', energies[l]) # energy of attenuated beam (MeV) after passing through this thickness
    print('ENERGY_ERROR: ', energy_errs[l])
    print('DEGREES: ', df)
    print('CHI2: ', chi2)
    print('CHI2_PDF: ', chi2/df)
    print('P(x2): ', 1 - sp.stats.chi2.cdf(chi2, df))
    print('\n')

#----- thicknesses -----#
# True values in (g/cm2)
true = {
    0 : 0,
    1 : 0.0025,
    2 : 0.0050,
    3 : 0.0075
}

print('TRUE VALUES:')
for l in true:
    print(l, ': ', true[l])
print('\n')

if E0 == 5.486: # unattenuated beam

    # calculated projected ranges
    l_0 = 1.806e-02 # 5.486 MeV
    l_1 = 1.424e-02 # 4.579 MeV
    l_2 = 1.154e-02 # 3.883 MeV
    l_3 = 6.305e-03 # 2.329 MeV

    # calculated projected ranges (upper uncertainty)
    l_0_u = 1.806e-02 # no error
    l_1_u = 1.446e-02 # 4.634 MeV
    l_2_u = 1.172e-02 # 3.932 MeV
    l_3_u = 6.409e-03 # 2.365 MeV

    # calculated projected ranges (lower uncertainty)
    l_0_l = 1.806e-02 # no error
    l_1_l = 1.403e-02 # 4.526 MeV
    l_2_l = 1.136e-02 # 3.835 MeV
    l_3_l = 6.195e-03 # 2.293 MeV

if E0 == 4.580: # attenuated beam
    
    # calculated projected ranges
    l_0 = 1.424e-02 # 4.580 MeV
    l_1 = 1.131e-02 # 3.823 MeV
    l_2 = 9.237e-03 # 3.242 MeV
    l_3 = 5.202e-03 # 1.945 MeV

    # calculated projected ranges (upper uncertainty)
    l_0_u = 1.424e-02 # no error
    l_1_u = 1.148e-02 # 3.868 MeV
    l_2_u = 9.378e-03 # 3.283 MeV
    l_3_u = 5.285e-03 # 1.975 MeV

    # calculated projected ranges (lower uncertainty)
    l_0_l = 1.424e-02 # no error
    l_1_l = 1.115e-02 # 3.778 MeV
    l_2_l = 9.100e-03 # 3.202 MeV
    l_3_l = 5.117e-03 # 1.914 MeV

ranges = [l_0, l_1, l_2, l_3]
ranges_upper = [l_0_u, l_1_u, l_2_u, l_3_u] # upper uncertainty
ranges_lower = [l_0_l, l_1_l, l_2_l, l_3_l] # lower uncertainty

#----- differences -----#
print('UPPER')
calculated_u = dict() # calculated thicknesses (upper uncertainty)
for l in range(len(ranges_upper)):
    calculated_u[l] = l_0_u - ranges_upper[l]
    print(l, ': ', calculated_u[l])
print('\n')

print('EXTRACTED:')
calculated = dict() # calculated thicknesses
for l in range(len(ranges)):
    calculated[l] = l_0 - ranges[l]
    print(l, ': ', calculated[l])
print('\n')

print('LOWER')
calculated_l = dict() # calculated thicknesses (lower uncertainty)
for l in range(len(ranges_lower)):
    calculated_l[l] = l_0_l - ranges_lower[l]
    print(l, ': ', calculated_l[l])
print('\n')

print('ERRORS')
errs_l = dict() # uncertainty on thickness
for l in range(len(ranges_lower)):
    errs_l[l] = ((calculated_l[l] - calculated[l]) + (calculated[l] - calculated_u[l]))/2
    print(l, ': ', errs_l[l])
print('\n')

print('PERCENT_ERRORS: ')
for l in range(len(ranges_lower)):
    if l != 0:
        print(l, ': ', errs_l[l]/calculated[l])


#----- plot results -----#
plot = 't' # t = thicknesses, g = gaussian

t = 0 # thickness being plotted
left, right = layers[t][2] # window boundaries
a, mu, std = layers[t][3] # fit parameters

if plot == 'g': # gaussian fit to spectra
    plt.scatter(x, layers[t][0], s=2, marker='.', color='black') # plot the MCA counts
    plt.plot(x[left:right], gaussian(array(x[left:right]), a, mu, std), color='red') # plot the fit

if plot == 't': # plot thicknesses (convert dict -> arrays first)
    ls = [1, 2, 3] # x axis
    true = list(true.values())[1:] # true thicknesses
    calc = list(calculated.values())[1:] # extracted thicknesses
    calc_u = list(calculated_u.values())[1:] # upper
    calc_l = list(calculated_l.values())[1:] # lower

    du = array(calc_u) - array(calc)
    dl = array(calc) - array(calc_l)

    errs = (du + dl)/2 # average error

    plt.scatter(ls, true, s=12, marker=',', color='green', label='True values') # true values
    plt.scatter(ls, calc, s=8, color='black', label='Measurements') # extracted values
    plt.errorbar(ls, calc, errs, color='black', capsize=1.5, ls='none') # uncertainty

    plt.legend()
plt.show()








