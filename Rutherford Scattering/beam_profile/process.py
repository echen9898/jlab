import os
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy import *
from scipy import stats
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from numpy import vectorize


############################### COUNT RATE VS ANGLE ###############################

#----- get filenames -----#
files = list()
for file in os.listdir():
    if file not in ['.DS_Store', 'info.txt', 'process.py']:
        files.append(file) 
window = [1623, 1039]

#----- fill count rates -----#
count_rates = list() # [total counts/time elapsed] -> 120 secs at each angle
rate_errs = list() # error on each count_rate

files.sort(key=lambda x: int(x[:2])) # sort left to right numerically by angle

for name in files: 
    
    readout = list() # total readout - every line
    counts = list() 
    with open(name) as file: 
        rows = file.readlines()
        for r in rows:
            readout.append(r)
            try:
                counts.append(int(r))
            except ValueError:
                pass
    
    count_rates.append(sum(counts)/120) # counts/sec
    rate_errs.append(sqrt(sum(counts))/120) # adding variances is the same as adding counts bc std = sqrt(count)!

def gaussian(x, a, mu, std): # gaussian fit function
    func = a * np.exp(-(x-mu)**2/(2*std**2)) 
    return func

def linear(x, m, b): # linear fit function
    func = m*x + b
    return func

#----- fit a gaussian to data -----#
x = array([-6, -4, -2, 0, 2, 4, 6, 8, 10]) * 0.0174533 # x values discrete (radians)
x_continuous = arange(-10, 12, 0.001) * 0.0174533 # x values continuous (radians)

p0 = array([170, 2, 4]) # amplitude, mean, std
fit = curve_fit(gaussian, x, count_rates, p0, rate_errs, maxfev=500000) # find the best fit with curve_fit
a, mu, std = fit[0]
print('GAUSSIAN FIT (a, mu, std): ', a, mu, std)

#----- fit two lines to this gaussian -----#
x_l = arange(-5.5*0.0174533, mu, 0.0000001) # left side x value range (mu modified to make it look good)
x_r = arange(mu, 10*0.0174533, 0.0000001) # right side x value range (mu modified to make it look good)
x_r_ext = arange(mu, 40*0.0174533, 0.00000001)

p0_l = array([1.5, -600])
fit_left = curve_fit(linear, x_l, gaussian(x_l, a, mu, std), p0_l, maxfev=500000)

p0_r = array([-1.5, 600])
fit_right = curve_fit(linear, x_r, gaussian(x_r, a, mu, std), p0_r, maxfev=500000)

m_l, b_l = fit_left[0] # parameters of the left line
m_r, b_r = fit_right[0] # parameters of the right line

cov_l = fit_left[1] # covariance matrix left
cov_r = fit_right[1] # covariance matrix right

ave_slope = (abs(m_l) + abs(m_r))/2
ave_b = (abs(b_l) + abs(b_r))/2

m_l_err = sqrt(cov_l[0][0]) # uncertainty on m_l
m_r_err = sqrt(cov_r[0][0]) # uncertainty on m_r

ave_m_err = (m_l_err + m_r_err)/2

print('slope_l: ', m_l, ' y_int_l: ', b_l)
print('slope_r: ', m_r, ' y_int_r: ', b_r)
print('ave_slope: ', ave_slope)
print('ave_slope_err: ', ave_m_err)
print('\n')

line_left = linear(x_l, ave_slope, b_l) # plot using average slope
line_right = linear(x_r, -ave_slope, b_r)

line_left_err = linear(x_l, ave_slope+ave_m_err, b_l) # plot using average slope + error
line_right_err = linear(x_r, -ave_slope+ave_m_err, b_r)

#----- find intercepts -----#
negative = True # find the y coordinate corresponding to the left line x intercept
for i in line_left:
    if i > 0 and negative:
        int_y_l = i
        negative = False

positive = True # find the y coordinate corresponding to the right line x intercept
for i in line_right:
    if i < 0 and positive:
        int_y_r = i
        positive = False

x_int_l = x_l[line_left.tolist().index(int_y_l)]
x_int_r = x_r[line_right.tolist().index(int_y_r)]

negative = True # find the y coordinate corresponding to the error left line x intercept
for i in line_left_err:
    if i > 0 and negative:
        int_y_l_err = i
        negative = False

print(line_right)
print(line_right_err)

positive = True # find the y coordinate corresponding to the error right line x intercept
for i in line_right_err:
    if i < 0 and positive:
        int_y_r_err = i
        positive = False

x_int_l_err = x_l[line_left_err.tolist().index(int_y_l_err)]
x_int_r_err = x_r[line_right_err.tolist().index(int_y_r_err)]

line_left[:line_left.tolist().index(int_y_l)] = 0 # set left tail to zero
line_right[line_right.tolist().index(int_y_r):] = 0 # set right tail to zero

print('x_int_l: ', x_int_l)
print('x_int_r: ', x_int_r)
print('x_int_l_err: ', x_int_l_err)
print('x_int_r_err: ', x_int_r_err)
print('\n')

ave_half_width = ((mu - x_int_l) + (x_int_r - mu))/2 # average triangle half width
hw_err = (x_int_l_err + x_int_r_err)/2 # average of the triangle half width error

print('th0: ', ave_half_width)
print('th0_err: ', hw_err)
print('\n')

#----- calculate vertical angle uncertainty -----#
dth = 0.0174533 # 1/2 degree uncertainty in angle
angle_errs = list()
for i in [0, 1, 7, 8]:
    if i == 1:
        angle_errs[i] = errs[7]
    else:
        angle_errs[i] = errs[i]

def find_closest(x, target): 
    '''Finds the value in phi closest to p'''
    minimum = 1000
    min_index = None
    for i in range(len(x)):
        if abs(target - x[i]) < minimum:
            minimum = abs(target - x[i])
            min_index = i
    return min_index

g_fit = gaussian(x_continuous, a, mu, std)
for p in x: # use the slope of the discrete gaussian fit to convert angular error

    p_o = find_closest(x_continuous, p)
    p_l = find_closest(x_continuous, p - dth)
    p_r = find_closest(x_continuous, p + dth)

    print('O: ', p, phi[p_o])
    print('L: ', p-th_err, phi[p_l])
    print('\n')

    g0 = g_fit[p_o]
    err_l = abs(g_fit[p_l] - g0)
    err_r = abs(g_fit[p_r] - g0)
    err = (err_l + err_r)/2
    angle_errs.append(abs(err)) # err is an array of one element, so add the scalar first element


total_err = array(rate_errs) + array(angle_errs) # poisson + angular
print('ANGULAR: ', angle_errs)
print('PERCENT ANGULAR: ', array(angle_errs)/total_err)
print('POISSON: ', rate_errs)
print('PERCENT POISSON: ', array(rate_errs)/total_err)
print('\n')

#----- chi2 -----#
def chi_square(observed, x, m, b, errs):
    chi_sq = 0
    for i in range(len(observed)):
        expected = linear(x[i], m, b)
        chi_sq += (observed[i] - expected)**2/total_err[i]**2
    return chi_sq

chi2_l = chi_square(count_rates[:5], x[:5], m_l, b_l, total_err[:5]) # chi2 left line
chi2_r = chi_square(count_rates[5:], x[5:], m_r, b_r, total_err[5:]) # chi2 right line

df_l = 5-2 # left 5 points, 2 parameters in linear fit
df_r = 4-2 # right 4 points, 2 parameters in linear fit

print('DEGREES_l: ', df_l)
print('CHI2_l: ', chi2_l)
print('CHI2_PDF_l: ', chi2_l/df_l)
print('P(x2_l): ', 1 - sp.stats.chi2.cdf(chi2_l, df_l))
print('\n')

print('DEGREES_r: ', df_r)
print('CHI2_r: ', chi2_r)
print('CHI2_PDF_r: ', chi2_r/df_r)
print('P(x2_r): ', 1 - sp.stats.chi2.cdf(chi2_r, df_r))
print('\n')

def chi_square_g(observed, x, a, mu, std):
    chi_sq = 0
    for i in range(len(observed)):
        expected = gaussian(x[i], a, mu, std)
        chi_sq += (observed[i] - expected)**2/total_err[i]**2
    return chi_sq

chi2_g = chi_square_g(count_rates, x, a, mu, std) # chi2 gaussian
df_g = 9-3 # left 9 points, 3 parameters in gaussian fit

print('DEGREES_g: ', df_g)
print('CHI2_g: ', chi2_g)
print('CHI2_PDF_g: ', chi2_g/df_g)
print('P(x2_g): ', 1 - sp.stats.chi2.cdf(chi2_g, df_g))
print('\n')

#----- plots -----#
plt.scatter(x, count_rates, s=4, marker=',', color='black') # count rates
plt.errorbar(x, count_rates, total_err, color='black', capsize=1.5, ls='none') # uncertainty

plt.plot(x_continuous, gaussian(x_continuous, a, mu, std), color='green') # gaussian fit
plt.plot(x_l, line_left, color='red', linestyle='--') # left linear fit
plt.plot(x_r, line_right, color='red', linestyle='--') # right linear fit

#----- plot parameters -----#
plt.xlim(-.13, .20)
plt.ylim(-10, 250)
plt.show()



