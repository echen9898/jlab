import sys, os
import numpy as np
import scipy as sp
from scipy import *
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import math

#################################### CONSTANTS ####################################
# CONSTANTS
m_e = 9.10938356e-28     # mass of an electron [grams]
e = 4.80320451e-10       # charge of an electron [statcoulomb]
c = 2.99792458e10        # speed of light in a vacuum [cm/s]
em_ratio = 5.27280978e17 # electron charge to mass ratio [statcoulomb/gram]

# APPARATUS CONSTANTS
rho = 20.3 # path radius (cm, +/- 0.4)
drho = 0.4 # uncertainty in path radius (cm)
d = 0.180  # distance between velocity selectors (cm, +/- 0.003)
dd = 0.003 # uncertainty in voltage selector plate distance

# MAGNETIC FIELD
B = array(list(range(50, 140))) # 70 to 120 originally, 80 to 111 for residuals plot
B_data = array([80, 85, 90, 95, 100, 105, 110]) # magnetic fields tested [G]
B_path_var = array([100.0, 98.8, 100.2, 100.6, 101.0, 100.6, 100.9, 100.2, 100.8, 100.8, 100.4, 100.8]) # path magnetic field measurements
B_mean = mean(B_path_var)
B_std = std(B_path_var) # uncertainty in B fluctuation
B_err_rel = (np.ones(np.shape(B_data))*B_std/B_data) # quadrature B uncertainty

# CENTRAL VOLTAGE
V_c = array([2690, 3021, 3430, 3611, 4010, 4170, 4510]) * 0.0033356405 # velocity selector potentials in statVolt
V_c_errs = [0.13415442018201987, 
            0.12907190259404439, 
            0.12086974861576942, 
            0.11662153371238322, 
            0.1193046736935328, 
            0.14327468766968912, 
            0.1516384522385753] # voltage center uncertainties

# ELECTRIC FIELD
E_data = V_c/d 
E_err_rel = ((array(V_c_errs)*3.3356405)/V_c) # relative uncertainty on E
print(E_err_rel)
E_err = E_data*E_err_rel # uncertainty on E

# ENERGY LINE PARAMETERS
b_l = 0.5581562195398447
b_err = 0.28
m_l = 0.28628644445481133
m_err = 0.005

####################################### FUNCTIONAL FORMS #######################################
def constant_fit(x, b): # fit to a constant
    func = 0*x + b
    return func

def gaussian_fit(x, a, mu, std): # evaluate function for an x value
    func = a * np.exp(-(x-mu)**2/(2*std**2)) # single gaussian
    return func

def chi_square_constant(observed, b, sigma):
    chi_sq = 0
    for i in range(len(observed)):
        chi_sq += (observed[i] - b)**2/sigma**2
    return chi_sq

def chi_square_gaussian(observed, x, a, mu, std):
    chi_sq = 0
    for i in range(len(observed)):
        expected = gaussian_fit(x[i], a, mu, std)
        chi_sq += (observed[i] - expected)**2/observed[i]
    return chi_sq


####################################### UNPACK DATA #######################################
filenames = list()
for file in os.listdir():
    if file not in ['.DS_Store', 'process.py', 'beta_residual.nb']:
        filenames.append(file) 

data = dict() # {B : counts}
for name in filenames: # for each voltage

    with open(name) as file: # open the file
        rows = file.readlines()
        bins = list()
        for r in rows:
            try: # only take numerical data
                bins.append(int(r))
            except ValueError:
                pass
        data[int(name[:-4])] = bins[:-5] # cut off spe, and settings at the end of file

####################################### BETA CALCULATION #######################################

# Newtonian Arrays
beta_newt = list()
beta_newt_upper_rho = list() # upper band rho
beta_newt_lower_rho = list() # lower band rho

# Relativistic Arrays
beta_rel = list()
beta_rel_p = list() # beta relative only at the specific fields measured
beta_rel_upper_rho = list() # upper band rho
beta_rel_lower_rho = list() # lower band rho

# Data Arrays
beta_data = V_c/(d*B_data) # beta for data points

#----- Models -----
def predict_newt_b(e, rho, m, c, B, mode=None):
    if mode == 'upper_band_rho': rho += drho # add rho uncertainty
    if mode == 'lower_band_rho': rho -= drho # subtract rho uncertainty
    if mode == 'upper_band_b': B += B_std # add B uncertainty
    if mode == 'lower_band_b': B -= B_std # subtract B uncertainty
    beta = ((e*rho*B)/(m*c**2))
    return beta

def predict_rel_b(e, rho, m, c, B, mode=None):
    if mode == 'upper_band_rho': rho += drho # add rho uncertainty
    if mode == 'lower_band_rho': rho -= drho # subtract rho uncertainty
    if mode == 'upper_band_b': B += B_std # add B uncertainty
    if mode == 'lower_band_b': B -= B_std # subtract B uncertainty
    num = (e*rho*B)/(m*c**2)
    den = (sqrt(1 + num**2))
    beta = num/den
    return beta

#----- Populate arrays -----
for b in B:
    if b in B_data:
        beta_rel_p.append(predict_rel_b(e, rho, m_e, c, b)) # used for ratio calculation

    # newtonian
    beta_newt.append(predict_newt_b(e, rho, m_e, c, b))
    beta_newt_upper_rho.append(predict_newt_b(e, rho, m_e, c, b, 'upper_band_rho'))
    beta_newt_lower_rho.append(predict_newt_b(e, rho, m_e, c, b, 'lower_band_rho'))

    # relativistic
    beta_rel.append(predict_rel_b(e, rho, m_e, c, b))
    beta_rel_upper_rho.append(predict_rel_b(e, rho, m_e, c, b, 'upper_band_rho'))
    beta_rel_lower_rho.append(predict_rel_b(e, rho, m_e, c, b, 'lower_band_rho'))

#----- residual plot -----
norm_error = np.array(beta_rel_p)/beta_data # ratio of relative points to data points
ratio_resid_errs = [0.0121196, 0.0109088, 0.00968648, 0.00925626, 0.00837021, 0.00807286, 0.0074774] # uncertainty on the residual points

#----- fit residuals to a line -----
p0_resid = array([1.6]) # initialize parameters
fit = curve_fit(constant_fit, B_data, norm_error, p0_resid, ratio_resid_errs, maxfev=500000) # find the best fit with curve_fit

# ----- These are the best fit parameters returned -----
resid_line = fit[0]
print('RESID_LINE: ', resid_line)

chi2_resid = chi_square_constant(norm_error, resid_line, std(norm_error))
df = 6 # 7 data points, 1 fit parameter
print('CHi2_RESIDS: ', chi2_resid)
print('CHI2_PDF_RESIDS: ', chi2_resid/df)
print('P(chi2_resid): ', 1 - sp.stats.chi2.cdf(chi2_resid, df), '\n')

####################################### CHARGE MASS RATIO CALCULATION #######################################
em_newt = list() # in statcoulombs/gram
em_rel = list()
em_real = np.ones(np.shape(B)) * (em_ratio)

def predict_newt_em(E, B):
    # em = (beta*c**2)/(rho*B) # beta version
    em = (E*c**2)/(rho*B**2) # electric field version
    return em

def predict_rel_em(E, B):
    # em = (beta*c**2)/(B*rho*sqrt(1 - beta**2)) # beta version
    em = (E*c**2)/(rho*sqrt(1-(E/B)**2)*B**2) # electric field version
    return em

# for i in range(len(beta_data)):
#     beta = beta_data[i]
#     em_newt.append(predict_newt_em(e, rho, c, B_data[i], beta))
#     em_rel.append(predict_rel_em(e, rho, m_e, c, B_data[i], beta))

for i in range(len(B_data)):
    em_newt.append(predict_newt_em(E_data[i], B_data[i]))
    em_rel.append(predict_rel_em(E_data[i], B_data[i]))

#----- errors -----
em_rel_err = np.sqrt(E_err_rel**2 + B_err_rel**2) * em_rel # e/m relativistic error
em_newt_err = np.sqrt(E_err_rel**2 + 4*B_err_rel**2) * em_newt # e/m newt error

#----- fit values to a line -----
p0_rel = array([4.5e17])
p0_newt = array([3e17])
rel_fit = curve_fit(constant_fit, B_data, em_rel, p0_rel, em_rel_err, maxfev=500000) # find the best fit with curve_fit
newt_fit = curve_fit(constant_fit, B_data, em_newt, p0_newt, em_newt_err, maxfev=500000) # find the best fit with curve_fit

# rel_fit_max = curve_fit(constant_fit, B_data, em_rel + )
# ----- These are the best fit parameters returned -----
em_rel_line = rel_fit[0] # relativistic e/m fit
em_newt_line = newt_fit[0] # newtonian e/m fit

# ----- Chi Squared -----
chi2_rel = chi_square_constant(em_rel, em_rel_line, std(array(em_rel)))
chi2_newt = chi_square_constant(em_newt, em_newt_line, std(array(em_newt)))
df = 6 # 7 -1
print('E/M rel: ', em_rel_line)
print('E/M newt: ', em_newt_line)
print('CHI2 REL: ', chi2_rel/df)
print('CHI2 PDF NEWT: ', chi2_newt/df)
print('P(chi2_rel): ', 1 - sp.stats.chi2.cdf(chi2_rel, df)) # p-value relative curve
print('P(chi2_newt): ', 1 - sp.stats.chi2.cdf(chi2_newt, df), '\n') # p-value newtonian curve

# ----- Monte Carlo for linear fit -----
monte_carlo = True # turn the simulation on or off

if monte_carlo:
    iterations = 10000
    resids_rel = list() # distance of each raffled line away from calculated line
    for it in range(iterations):
        em_rel_val = list()
        for i in em_rel:
            em_rel_val.append(i)
        for i in range(len(em_rel)): # for every data point, raffle a random offset
            em_rel_val[i] += np.random.normal(0, em_rel_err[i])
        rel_err_fit = curve_fit(constant_fit, B_data, em_rel_val, p0_rel, maxfev=500000) # find the best fit with curve_fit
        resids_rel.append(abs(rel_err_fit[0] - em_rel_line)) # raffled - true

    print('ERR_IN_E/M_REL: ', std(array(resids_rel)))

    resids_newt = list() # distance of each raffled line away from calculated line
    for it in range(iterations):
        em_newt_val = list()
        for i in em_newt:
            em_newt_val.append(i)
        for i in range(len(em_newt)): # for every data point, raffle a random offset
            em_newt_val[i] += np.random.normal(0, em_newt_err[i])
        newt_err_fit = curve_fit(constant_fit, B_data, em_newt_val, p0_newt, maxfev=500000) # find the best fit with curve_fit
        resids_newt.append(abs(newt_err_fit[0] - em_newt_line)) # raffled - true

    print('ERR_IN_E/M_NEWT: ', std(array(resids_newt)))


####################################### K CALCULATION #######################################
K_newt = list()
K_rel = list()
K_data = list()
K_data_errs = list()
P = (B*e*rho)/c # convert B range to p range
P_data = (B_data*e*rho)/c

#----- find peak channel number for each B -----
# 80 -> [607, 680],
# 85 -> [691, 757], 
# 90 -> [780, 861], 
# 95 -> [843, 926], 
# 100 -> [943, 1052], 
# 105 -> [988, 1100], 
# 110 -> [1094, 1224]

fields = {
    80 : [[607, 680], [47, 642, 14]],
    85 : [[691, 757], [58, 723, 13]],
    90 : [[780, 861], [58, 822, 16]],
    95 : [[843, 926], [70, 880, 16]],
    100 : [[947, 1048], [75, 995, 15]],
    105 : [[1013, 1075], [90, 1045, 15]],
    110 : [[1121, 1195], [80, 1157, 18]]
}

def predict_K_newt(m, p):
    return (p**2)/(2*m)

def predict_K_rel(m, c, p):
    return sqrt((m**2*c**4) + (c**2*p**2)) - (m*c**2)

def predict_K_data(n, m, b):
    return m*n + b

#----- find K on data -----
means = list()
mean_errs = list()
for b in B_data:
    x = arange(fields[b][0][0], fields[b][0][1], 1)
    y = data[b][fields[b][0][0]:fields[b][0][1]]
    yerrs = sqrt(array(y)) # poisson random error on counts
    p0_K = array(fields[b][1]) # initialize parameters
    peak_fit = curve_fit(gaussian_fit, x, y, p0_K, yerrs, maxfev=500000) # find the best fit with curve_fit

    # ----- best fit parameters -----
    a, mu, std = peak_fit[0]
    std_m = std/sqrt(len(y))
    chi2 = chi_square_gaussian(y, x, a, mu, std)
    N = len(y) # number of observed counts
    df = N - len(peak_fit[0]) # degrees of freedom
    print('FIELD: ', b)
    print('MEAN: ', mu)
    print('STD: ', std)
    print('STD_M: ', std/sqrt(sum(y))) # standard deviation on mean
    print('CHI_SQ: ', chi2) # chi square
    print('DEGREES: ', df) # degrees of freedom
    print('CHI_SQ_PDF: ', chi2/df) # chi square pdf
    print('P(X2): ', 1 - sp.stats.chi2.cdf(chi2, df), '\n') # p-value

    #----- plotting option -----
    # if b == 110:
    #     plt.scatter(x, y, s=0.5, color='black')
    #     plt.plot(x, gaussian_fit(x, a, mu, std))

    means.append(mu) # mean channel numbers
    mean_errs.append(std_m) # standard error on the mean
    K_data.append(predict_K_data(mu, m_l, b_l)) # append energies in keV units
    K_data_errs.append(predict_K_data(mu+std_m, m_l+m_err, b_l+b_err)-predict_K_data(mu, m_l, b_l))

print(K_data_errs)
#----- predict models -----
for p in P:
    K_newt.append(predict_K_newt(m_e, p)*6.242e8) # convert to keV
    K_rel.append(predict_K_rel(m_e, c, p)*6.242e8) 


####################################### PLOTS #######################################
mode = 'beta' # beta, K, e/m, B

#----------- K vs p -----------#
if mode == 'K':
    # plt.scatter(list(range(607, 680)), data[80][607:680], s=0.5, color='black')
    plt.plot(P, K_newt, color='b')
    plt.plot(P, K_rel, color='r')
    plt.scatter(P_data, K_data, s=2, color='black')
    plt.errorbar(P_data, K_data, K_data_errs, color='black', capsize=1.5, ls='none')

#----------- β vs B -----------#
if mode == 'beta':

    # FINAL PLOT
    fig = plt.figure(1)
    frame1 = fig.add_axes((.1, .3, .8, .6))

    plt.plot(B, beta_rel_upper_rho, color='r', linestyle='--') # plot upper band rho
    plt.plot(B, beta_rel_lower_rho, color='r', linestyle='--', label='Relativistic') # plot lower band rho
    plt.plot(B, beta_newt_upper_rho, color='b', linestyle='--') # plot upper band rho
    plt.plot(B, beta_newt_lower_rho, color='b', linestyle='--', label='Newtonian') # plot lower band rho
    plt.scatter(B_data, beta_data, s=40, marker='.', color='black') # plot your data
    plt.errorbar(B_data, beta_data, V_c_errs/(B_data*d), color='black', capsize=1.5, ls='none')
    plt.xlim(70, 120)
    plt.errorbar(B_data, beta_data, V_c_errs/V_c, color='black', capsize=1.5, ls='none')
    # plt.legend()

    frame2 = fig.add_axes((.1, .1, .8, .2))
    plt.scatter(B_data, norm_error, s=40, marker='.', color='black') # plot residuals
    plt.errorbar(B_data, norm_error, ratio_resid_errs, color='black', capsize=1.5, ls='none')
    print(ratio_resid_errs)
    plt.plot(B, np.ones(np.shape(B))*resid_line, color='black', linestyle='--')
    plt.xlim(70, 120)
    plt.legend()

    # RELATIVISTIC LINE
    #plt.plot(B, beta_rel, color='red') # plot relativistic line
    # plt.plot(B, beta_rel_upper_rho, color='r', linestyle='--') # plot upper band rho
    # plt.plot(B, beta_rel_lower_rho, color='r', linestyle='--') # plot lower band rho

    # NEWTONIAN LINE
    #plt.plot(B, beta_newt, color='blue') # plot classical line
    # plt.plot(B, beta_newt_upper_rho, color='b', linestyle='--') # plot upper band rho
    # plt.plot(B, beta_newt_lower_rho, color='b', linestyle='--') # plot lower band rho

    # DATA POINTS
    # plt.scatter(B_data, beta_data, s=25, marker='.', color='black') # plot your data
    # plt.errorbar(B_data, beta_data, beta_sys, color='black', capsize=1.5, ls='none')

    # LABELLING
    # plt.title('\\beta vs. B', fontsize=19)
    # plt.xlabel('Magnetic Field (G)', fontsize=15)
    # plt.ylabel('\\beta', fontsize=15)
    print('NORM FLUCTUATION: ', sum(norm_error)/len(norm_error)) # average normalization error

#----------- e/m vs B -----------#
if mode == 'e/m':
    plt.plot(B, em_real, color='black', label='True value = ') # plot the true value

    plt.plot(B, np.ones(np.shape(B))*em_rel_line, color='red', label='Relativistic = ') # plot rel fit
    plt.plot(B, np.ones(np.shape(B))*(em_rel_line + std(array(resids_rel))), linewidth=1, color='red', linestyle='--') # upper error band
    plt.plot(B, np.ones(np.shape(B))*(em_rel_line - std(array(resids_rel))), linewidth=1, color='red', linestyle='--') # lower error band
    plt.scatter(B_data, em_rel, s=15, marker=',', color='red') # plot rel points
    plt.errorbar(B_data, em_rel, em_rel_err, color='black', capsize=1.5, ls='none') # plot rel errors

    plt.plot(B, np.ones(np.shape(B))*em_newt_line, color='blue', label='Newtonian = ') # plot newt fit
    plt.plot(B, np.ones(np.shape(B))*(em_newt_line + std(array(resids_newt))), linewidth=1, color='blue', linestyle='--') # upper error band
    plt.plot(B, np.ones(np.shape(B))*(em_newt_line - std(array(resids_newt))), linewidth=1, color='blue', linestyle='--') # lower error band
    plt.scatter(B_data, em_newt, s=15, marker=',', color='blue') # plot newt points
    plt.errorbar(B_data, em_newt, em_newt_err, color='black', capsize=1.5, ls='none') # plot newt errors

    plt.ylim((2.85e17, 5.9e17))
    # plt.legend()

#----------- B calibration -----------#
if mode == 'B':
    B_calibration = array([100.0, 98.8, 100.2, 100.6, 101.0, 100.6, 100.9, 100.2, 100.8, 100.8, 100.4, 100.8]) # magnetic field strengths along the path
    bins = np.arange(98, 102, 0.3)
    print('MEAN_B: ', np.mean(B_calibration))
    print('STD_B: ', np.std(B_calibration))
    plt.hist(B_calibration, bins)

plt.show()

