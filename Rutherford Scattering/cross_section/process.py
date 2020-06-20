import os
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy import *
from scipy import stats
from scipy.optimize import leastsq
from scipy.optimize import curve_fit


################################# NORMALIZATION #################################

norms = dict() # {date : countrate} -> normalization countrates for each day

norm_windows = { # {date : window} -> energy window for each day of measurements
    7 : [1250, 1665],
    14 : [1367, 1787],
    19 : [1371, 1870],
    26 : [1383, 1892],
    28 : [1394, 1874]
}

dates = { # {angle : date} -> day each angle measurement was taken
    10 : 19,
    20 : 19,
    30 : 19,
    40 : 19,
    50 : 28,
    60 : 26
}

# windows for each date

os.chdir('norm')

#----- get filenames -----#

norm_files = list()
for file in os.listdir():
    if file != '.DS_Store':
        norm_files.append(file) 

#----- fill count rates -----#

for name in norm_files: # for each angle
    
    readout = list() # total readout - every line
    counts = list() # counts

    with open(name) as file: # open the file
        rows = file.readlines()

        for r in rows:
            readout.append(r)
            try:
                counts.append(int(r))
            except ValueError:
                pass
    
    #norms[int(name[:2])] = counts # for plotting purposes
    l, r = norm_windows[int(name[:2])]
    norms[int(name[:2])] = sum(counts[l:r])/120 # counts/sec

os.chdir('..')

#----- display norms -----#

# for d in norms:
#     print('DATE ', d, ': ', norms[d])

# plt.scatter(list(range(0, 2053)), norms[28], s=3, marker='.', color='black') # plot normalization spectra


################################# COUNTRATE vs. ANGLE #################################

#----- initialization -----#

offset = 0.03883333771325598 # offset from center beam profile in radians
angles = [10, 20, 30, 40, 50, 60] # degrees
angles_rad = array(angles) * 0.0174533 #- offset# angles in radians

durations = { # {angle : total measurement time (sec)}
    10 : 600,
    20 : 600,
    30 : 1200,
    40 : 4800,
    50 : 22834,
    60 : 59163
}

meas = dict() # {angle : total counts} -> sum all the spectra for each angle
rates = list() # list of I/I0 normalized countrates
rate_errs = list() # list of I/I0 uncertainties due to poisson counts

Z_g = 79 # gold atomic number
Z_a = 2 # alpha particle atomic number
E = 5.486 # MeV (unattenuated energy)
e = 1.60217662e-19 # coulombs

#----- fill count rates -----#

angle_dirs = list()
for file in os.listdir():
    if file not in ['.DS_Store', 'norm', 'info.txt', 'process.py']:
        angle_dirs.append(file) 

for angle in angle_dirs: # for each angle
    
    if int(angle) == 10: # exclude 10 degree point
        os.chdir(angle) # go into that files directory

        total_counts = zeros((374, )) # Matt's window

        files = list() # all spectra in this directory
        for file in os.listdir():
            if file not in ['.DS_Store']:
                files.append(file) 

        for n in files: # get counts for each spectra, and add to total array

            counts = list()
            with open(n) as file: # open the file
                rows = file.readlines()

                for r in rows:
                    try:
                        counts.append(int(r))
                    except ValueError:
                        pass
                        
            total_counts += array(counts[970:1344]) # window matt is using for all pionts - 10 degree gaussian fit window (bc collisions are elastic so energy is conserved)

        meas[int(angle)] = total_counts # summed spectra
        os.chdir('..')

    else:
        os.chdir(angle) # go into that files directory

        total_counts = zeros((2048, )) # initial array of zeros

        files = list() # all spectra in this directory
        for file in os.listdir():
            if file not in ['.DS_Store']:
                files.append(file) 

        for n in files: # get counts for each spectra, and add to total array

            counts = list()
            with open(n) as file: # open the file
                rows = file.readlines()

                for r in rows:
                    try:
                        counts.append(int(r))
                    except ValueError:
                        pass
                        
            total_counts += array(counts[:-5]) # shave off last 5 numerica settings (settings not data)

        meas[int(angle)] = total_counts # summed spectra
        os.chdir('..')

#----- display total spectra -----#

# plt.scatter(list(range(0, 2048)), meas[20], s=3, marker='.', color='black') # plot total spectra

#----- I/I0 -----#

for a in angles:
    # calculate I/I0
    I = sum(meas[a])/durations[a]
    I0 = norms[dates[a]]
    rates.append(I/I0)

    # calculate/propagate poisson error on I/I0
    dI = sqrt(sum(meas[a]))/durations[a]
    dI0 = sqrt(norms[dates[a]])/120 # each normalization is 120 seconds

    d_ratio = (I/I0) * sqrt( (dI/I)**2 + (dI0/I0)**2 ) # propagate dI and dI0
    rate_errs.append(d_ratio)

print('I/I0: ', rates)
print('\n')

################################# CONVOLUTION #################################

theta = arange(0.001, math.pi, 0.01) # 0 to pi range
phi = arange(3, 80, 0.01) * 0.0174533 # all phis from 3 to 80 degrees in rads
dth = theta[1] - theta[0] # spacing between thetas in rads

th0 =  0.11773504385807504 # half width of triangle in rads
# th0 = 0.115 # matts
dth0 = 0.03872789385856346 # uncertainty on half width of triangle

def calc_triangular(theta, phi, th0): # triangular response function
    if abs(theta - phi) < th0:
        return 1 - abs(theta - phi)/th0
    return 0

def r_model(x, c0): # rutherford model
    func = c0/(sin(x/2))**4
    return func

def t_model(x, c0): # thomson model
    func = c0 * (1 + cos(x)**2)
    return func

# errs = [0.0009491002011667106, 0.0001281878667501588, 3.201596467811618e-05, 9.867646162076535e-06, 4.1192077389075325e-06]

# errs = [0.3549901989611043, # using dth0 = 0.115
# 0.0007919525743769509,
# 0.00010287641209611671,
# 2.538758529906756e-05,
# 7.724530739908283e-06,
# 3.1585725297556283e-06]

# errs = [5.68721183e-01, 7.86765889e-04, 1.02873697e-04, 2.49677076e-05,
# 7.70534446e-06, 3.08258814e-06]

errs = [0.08530817745, 7.86765889e-04, 1.02873697e-04, 2.49677076e-05,
 7.70534446e-06, 3.08258814e-06] # dth0 = 0.1177

# errs = [3.58352729e-02, 5.71846597e-04, 9.42475142e-05, 2.42017454e-05,
#  7.74455331e-06, 3.14156581e-06]

# errs = [0.4, # bdm number for point 1
# 0.0009064410318617521,
# 0.00010652771828014701,
# 2.5159854973906635e-05,
# 7.621535781422774e-06,
# 3.0838259690420294e-06]

p0_r = array([1e-5]) # fit the rutherford functin
fit_r = curve_fit(r_model, angles_rad, rates, p0_r, errs, maxfev=500000)
c0_r = fit_r[0] # rutherford fit constant
print('C0_r: ', c0_r)

rutherford_discrete = r_model(angles_rad, c0_r) # 6 data points
rutherford = r_model(theta, c0_r) # continuous rutherford for convolution

f = list() # convolved model
f_upper = list() # convolved model upper band (beam profile error +dth0)
f_lower = list() # convolved model lower band (beam profile error -dth0)

f_discrete = list() # convolved rutherford function at measured angles
f_upper_disc = list() # convolved model upper band used for fitting
f_lower_disc = list() # convolved model lower band used for fitting

#----- continuous convolution -----#

for p in phi: # convolution of rutherford and triangular response function
    
    # p = p + offset # off center beam profile

    g = list()
    g_upper = list()
    g_lower = list()
    for th in theta: # - offset:
        g.append(calc_triangular(th, p, th0))
        g_upper.append(calc_triangular(th, p, th0 + dth0))
        g_lower.append(calc_triangular(th, p, th0 - dth0))

    res = dth * sum(g*rutherford)
    res_upper = dth * sum(g_upper*rutherford)
    res_lower = dth * sum(g_lower*rutherford)

    f.append(res)
    f_upper.append(res_upper)
    f_lower.append(res_lower)

#----- discrete convolution -----#

for p in angles_rad: # convolution only at discrete measurement angles
    
    # p = p + offset # off center beam profile

    g = list()
    g_upper = list()
    g_lower = list()
    for th in theta: # - offset:
        g.append(calc_triangular(th, p, th0))
        g_upper.append(calc_triangular(th, p, th0 + dth0))
        g_lower.append(calc_triangular(th, p, th0 - dth0))

    res = dth * sum(g*rutherford)
    res_upper = dth * sum(g_upper*rutherford)
    res_lower = dth * sum(g_lower*rutherford)

    f_discrete.append(res)
    f_upper_disc.append(res_upper)
    f_lower_disc.append(res_lower)

#----- fit convolved form constant -----#

def f_model(x, c0): # convolved rutherford model fit
    func = 0*x + c0*array(f_discrete)
    return func

def f_model_upper(x, c0): # convolved rutherford model fit
    func = 0*x + c0*array(f_upper_disc)
    return func

def f_model_lower(x, c0): # convolved rutherford model fit
    func = 0*x + c0*array(f_lower_disc)
    return func

p0_f = array([5e-4]) # use same initial scaling constant for each curve

fit_f = curve_fit(f_model, angles_rad, rates, p0_f, errs, maxfev=500000)
fit_f_upper = curve_fit(f_model_upper, angles_rad, rates, p0_f, errs, maxfev=500000)
fit_f_lower = curve_fit(f_model_lower, angles_rad, rates, p0_f, errs, maxfev=500000)

c0_f = fit_f[0] # constant obtained by fit
c0_f_upper = fit_f_upper[0]
c0_f_lower = fit_f_lower[0]

print('C0_f: ', c0_f)
print('C0_f_upper: ', c0_f_upper)
print('C0_f_lower: ', c0_f_lower)
print('PERCENT BEAM PROFILE: ', c0_f_upper/c0_f)

#----- angular uncertainty propagation -----#

angle_errs = list() # vertical errors due to angular uncertainty (1 degree)
th_err = 0.0174533 # radians (1 degree)

def find_closest(phi, p): # finds the value in phi closest to p
    minimum = 1000
    min_index = None
    for i in range(len(phi)):
        if abs(p - phi[i]) < minimum:
            minimum = p - phi[i]
            min_index = i
    return min_index

for p in angles_rad: # use the slope of the discrete fit to convert angular error

    p_o = find_closest(phi, p)
    p_l = find_closest(phi, p + th_err)

    # print('O: ', p, phi[p_o])
    # print('L: ', p-th_err, phi[p_l])
    # print('\n')

    err = c0_f*f[p_l] - c0_f*f[p_o] # rise over run times run = rise
    angle_errs.append(abs(err[0])) # err is an array of one element, so add the scalar first element

total_err = array(rate_errs) + array(angle_errs) # poisson (rate_errs) + angular (angle_errs)
print('TOTAL/val: ', total_err/rates)
print('ANGULAR/val: ', array(angle_errs)/rates)
print('POISSON/val: ', array(rate_errs)/rates)
print('PERCENT ANGULAR: ', array(angle_errs)/total_err) 
print('PERCENT POISSON: ', array(rate_errs)/total_err)
print('\n')

print('TOTAL: ', total_err)
print('ANGULAR: ', angle_errs)
print('POISSON: ', rate_errs)
print('\n')

#----- chi2 -----#

def chi_square(c, func):
    fit = c * array(func) # 6 fit points
    print('F', (rates - fit)**2)
    chi_sq = sum((rates - fit)**2/total_err**2) # rates is the 6 data points
    return chi_sq

def chi_square_r(c, func): # can fit directly to rutherford_discrete (c0_r already multiplied above)
    chi_sq = sum((rates - array(rutherford_discrete))**2/total_err**2) # rates is the 6 data points
    return chi_sq

chi2 = chi_square(c0_f, f_discrete) # perform chi2 on fit
chi2_r = chi_sq = sum((rates - array(rutherford_discrete))**2/total_err**2)

df = len(angles) - 1 # 1 fit parameter, 6 points

# convolution
print('DEGREES: ', df)
print('CHI2: ', chi2)
print('CHI2_PDF: ', chi2/df)
print('P(x2): ', 1 - sp.stats.chi2.cdf(chi2, df))
print('\n')

# unconvoluted rutherford
print('DEGREES: ', df)
print('CHI2_r: ', chi2_r)
print('CHI2_PDF_r: ', chi2_r/df)
print('P(x2): ', 1 - sp.stats.chi2.cdf(chi2_r, df))
print('\n')

#----- plot -----#

plt.scatter(angles_rad, rates, s=10, marker='.', color='black') # plot normalized rates
plt.errorbar(angles_rad, rates, errs, color='black', capsize=1.5, ls='none') # uncertainties

plt.plot(phi, c0_f*f, color='green') # plot convoluted fit
# plt.plot(phi, c0_f_upper*f_upper, color='red', linestyle='--') # upper band due to beam profile width error
# plt.plot(phi, c0_f_lower*f_lower, color='red', linestyle='--') # lower band due to beam profile width error

plt.plot(theta, rutherford, color='blue', linestyle='--') # plot unconvoluted rutherford fit

#----- plot parameters -----#

plt.yscale('log')
plt.xlim(0.1, 61 * 0.0174533)
plt.ylim(0, 0.7e1)

plt.show()
