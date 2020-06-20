import numpy as np
import scipy as sp
from scipy import *
from scipy import stats
import math
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from scipy.optimize import leastsq
from scipy.optimize import curve_fit


###################################### EXTRACT DATA ######################################

name = 'BaCalibration.spe'

readout = list() # total readout - every line
counts = list() # skip settings and only take numerical data
bins = list()

with open(name) as file: # open the file
    rows = file.readlines()
    i = 0
    for r in rows:
        i += 1
        readout.append(r)
        if 12 < i < 2061: # bins are on lines 12 to 2061 only
            bins.append(i - 12)
            counts.append(int(r))

# ----- x and y values of your barium spectrum plot -----
x = array(bins[61:1400]) # discriminator at 61, noisy signal after 1500
y = array(counts[61:1400])


###################################### PLOT BARIUM SPECTRUM ######################################

# ----- plot the barium spectrum -----
# plt.scatter(x, np.absolute(y), s=1, color='black') # general plot
# plt.yscale('log')

# ----- label plot -----
# plt.title('Barium Spectrum', fontsize=25)
# plt.ylabel('Counts', fontsize=16)
# plt.xlabel('Channel Number', fontsize=16)
# plt.text(265, 95246, s='B', fontsize=15)
# plt.text(1040, 1299, s='C', fontsize=15)
# plt.text(1221, 645, s='D', fontsize=15)
# plt.text(1325, 84, s='E', fontsize=15)

###################################### PLOT EXP/GAUSS FIT TO PEAKS ######################################

# pick a specific peak to fit: 
# A: window = (85:130), offset = 400000, params_0 = [500000, 107, 8], center = 107.05628897807182, std = 9.403909385131344
# B: window = (263:294), offset = 50000, params_0 = [25000, 281, 8], center = 279.836315, std = 8.19760003
# C: window = (1040:1070), offset = 620, params_0 = [50, 1055, 5], center = 1056.46121102, std = 8.72888751
# D: window = (1213:1280), offset = 80, params_0 = [250, 1240, 10], center = 1240.01788842, std = 10.75859119
# E: window = (1305:1366), offset = 13, params_0 = [30, 1335, 10], center = 1337.7630451, std = 13.50652841

x_frame = np.array(bins[85:130]) # zoom in on a specific peak
y_frame = np.array(counts[85:130]) # bring peak lower to fit a gaussian accurately
y_frame_errs = sqrt(y_frame) # treat each bin count as a poisson random variable
counts = sum(y_frame) # total number of samples in the fit
offset = 0

def func(x, a, mu, std): # evaluate function for an x value
    func = a * np.exp(-(x-mu)**2/(2*std**2)) # single gaussian
    return func

params_0 = array([500000, 107, 8]) # initialize parameters

fit = curve_fit(func, x_frame, y_frame - offset, params_0, y_frame_errs, maxfev=500000) # find the best fit with curve_fit

# ----- calculate chi2  -----
def chi_square_gauss(observed, x_range, a, mu, std):
    chi_sq = 0
    for i in range(len(observed)):
        expected = func(x_range[i], a, mu, std) # expected gaussian fit point
        chi_sq += (observed[i] - expected)**2/observed[i]
    return chi_sq

def chi_square_linear(observed, x_range, m, b, errs):
    chi_sq = 0
    for i in range(len(observed)):
        expected = m*x_range[i] + b # expected linear fit point
        chi_sq += (observed[i] - expected)**2/errs[i]**2
    return chi_sq

# ----- display results -----
a, mu, std = fit[0]
chi2_g = chi_square_gauss(y_frame, x_frame, a, mu, std)
df_g = len(x_frame) - len(params_0)
print('MEAN: ', mu)
print('STD: ', std)
print('N: ', sum(y_frame))
print('STD_M: ', std/sum(y_frame))
print('CHI_SQ: ', chi2_g)
print('DEGREES: ', df_g)
print('CHI_SQ_PDF: ', chi2_g/df_g)
print('P(X2): ', 1 - sp.stats.chi2.cdf(chi2_g, df_g), '\n')

# ----- plot curve fit -----
plt.scatter(x, np.absolute(y) - offset, s=1, color='black') # plot specific zoomed in peak
plt.plot(x_frame, func(x_frame, a, mu, std), color='red') # plot the fit
plt.yscale('log')


###################################### PLOT FINAL ENERGY/CHANNEL LINE ######################################

channels = [107.05628897807182, 279.836315, 1056.46121102, 1240.01788842, 1337.7630451]
energies = [30.97, 81, 302.4, 356, 383.9]
uncertainties = np.array([9.403909385131344, 8.19760003, 8.72888751, 10.75859119, 13.50652841])
weights = 1/uncertainties

# ----- move endpoints through max uncertainty to get a limit on line -----
endpoints = array([107.05628897807182, 1337.7630451])
end_energies_u = array([30.97 - (9.403909385131344/sqrt(counts)), 383.9 + (13.50652841/sqrt(counts))])
end_energies_l = array([30.97 + (9.403909385131344/sqrt(counts)), 383.9 - (13.50652841/sqrt(counts))])
b_min, m_max = polyfit(endpoints, end_energies_u, 1)
b_max, m_min = polyfit(endpoints, end_energies_l, 1)
print(m_max, b_min)
print(m_min, b_max)

b, m = polyfit(channels, energies, 1, w=weights)
print(m, b)
chi2_l = chi_square_linear(energies, channels, m, b, uncertainties/sqrt(counts))
df_l = len(channels) - 2
print('LINEAR FIT: ', m, 'x + ', b)
print('CHI_SQ: ', chi2_l)
print('DEGREES: ', df_l)
print('CHI_SQ_PDF: ', chi2_l/df_l)
print('P(X2): ', 1 - sp.stats.chi2.cdf(chi2_l, df_l))

# # ----- plot 3 largest peak points, and its fit line -----
# plt.plot(x, m*x + b, color='green', label='Linear fit')
# plt.plot(x, m_max*x + b_min, color='red', linestyle='--', label='Max slope fit')
# plt.plot(x, m_min*x + b_max, color='blue', linestyle='--', label='Min slope fit')
# plt.errorbar(channels, energies, uncertainties/sqrt(counts), color='black', capsize=1.5, ls='none')
# plt.scatter(channels, energies, s=15, color='black', marker=',', label='Data')
# plt.legend()

# plt.title('Relationship Between Energy and MCA Channel Number', fontsize=15)
plt.xlabel('Channel Number', fontsize=15)
plt.ylabel('Energy (keV)', fontsize=15)
plt.show()





