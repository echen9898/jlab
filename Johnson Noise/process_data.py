import numpy as np
import scipy as sp
from scipy import *
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from math import *
from scipy.optimize import curve_fit
import data as d

plot_mode = 'k' # options: [gain, k, 0]

################################### CONSTANTS ###################################

C = 5.493e-11           # farads
k_true = 1.38064852e-23 # J/K
abs_0 = -273.15         # celsius
dT = 2                  # kelvin
dC = 0.075

################################### DATA ###################################

#---------- gain calibration ----------#
freqs = d.frequencies_1
v_in = array([d.v_in_1, d.v_in_2, d.v_in_3])
v_out = array([d.v_out_1, d.v_out_2, d.v_out_3])

v_in_ave = mean(v_in, axis=0)   # mean over the three trials
v_out_ave = mean(v_out, axis=0)
v_in_std = std(v_in, axis=0)    # standard deviation over the three trials
v_out_std = std(v_out, axis=0)

g2 = (v_out_ave/v_in_ave)**2    # pure gain curve

gain_curves = dict()            # adjusted gain curves for resists (r : [])
gain_curves_lower = dict()      # adjusted gain curves with lower bound on C

G_trapezoid = dict()            # gain integrals using trapezoidal (r : int)
G_simpson = dict()              # gain integrals using simpsons method (r : int)
G_trap_lower = dict()           # gain integrals with lower bound on C

G_err = dict()                  # percent error on trapezoidal value given C and numerical integration method variation
G_err_temp = dict()             # percent error on trapezoidal value given above and R fluctuation

#---------- temperatures ----------#
temps = d.temperatures          # temperature -> 'r' and 's' -> list

#---------- resistances ----------#
resists = d.new_resistances     # resistance -> 'r' and 's' -> list
old_resists = d.old_resistances # resistance -> 'r' and 's' -> list

temps[22] = {
    'r' : resists[99.8]['r'],
    's' : resists[99.8]['s']
}

V_s = dict()                    # shorted voltages {r : {'values' : [], 'mean' = int, 'std' = int}}
vs_temps = list()               # shorted voltages for temperature varying measurements
V_r = dict()                    # resistance voltages {r : {'values' : [], 'mean' = int, 'std' = int}}
V_r_old = dict()                # old resistances
dv2 = dict()                    # error in difference V2
dv2_temps = dict()              # error in difference V2 for temperature varying measurements

for r in resists:

    temp = array(resists[r]['r'])/1000 # volts
    V_r[r] = {
    'values' : temp,
    'mean' : mean(temp),
    'std' : std(temp),
    'dvr' : mean(temp)/sqrt(shape(temp)[0])
    }

    temp2 = array(resists[r]['s'])/1000
    V_s[r] = {
    'values' : temp2,
    'mean' : mean(temp2),
    'std' : std(temp2),
    }

for r in old_resists:

    temp3 = array(old_resists[r]['r'])/1000 # volts
    V_r_old[r] = {
    'values' : temp3,
    'mean' : mean(temp3),
    'std' : std(temp3),
    'dvr' : mean(temp3)/sqrt(shape(temp3)[0])
    }

# print(V_r[698.0]['mean'])
# print(V_r_old[698.0]['mean'])
# print(V_r[698.0]['mean']/V_r_old[698.0]['mean'])

for r in resists: # fill uncertainties in v2
    dv2[r] = sqrt((std(V_s[r]['values'])**2)**2/len(V_s[r]['values']) + (std((V_r[r]['values'])**2)**2/len(V_r[r]['values']))) # square root of variances

for t in temps: # fill vs for temperature varying measurement
    try:
        vs_temps += temps[t]['s']
    except KeyError:
        pass

for t in temps: # fill uncertainties in v2 for temperature varying measurement
    dv2_temps[t] = sqrt( (std(array(temps[t]['r'])/1000)**2)**2/len(temps[t]['r']) + ((std(array(vs_temps)/1000)**2)**2/len(vs_temps)) )

################################### GAIN CURVE ###################################

def calc_g_f(R, C=C): # adjusted gain curve
    result = g2/(1 + (2*pi*freqs*1000*C*R*1000)**2).tolist() # Hz, ohm
    return result

def calc_G_trapezoidal(g2_f): # gain integral using trapezoid method
    A = 0
    for i in range(len(g2_f) - 1):
        dX = freqs[i+1]*1000 - freqs[i]*1000 # ohm
        A += (dX/2)*(g2_f[i] + g2_f[i+1])
    return A

def calc_G_simpsons(g2_f): # gain integral using simpsons rule
    A = 0
    c = 0
    for i in range(len(g2_f)):

        if 10 <= freqs[i] <= 90: # simpsons
            factor = 5000/3 #step in Hz
            if freqs[i] == 10 or freqs[i] == 90: # first and last elements
                A += factor*g2_f[i]
            elif c%2 == 0: # even
                A += 2*factor*g2_f[i]
            elif c%2 != 0: # odd
                A += 4*factor*g2_f[i]
            c += 1
                  
        else: # use trapezoid for first part
            dX = freqs[i+1]*1000 - freqs[i]*1000
            A += (dX/2)*(g2_f[i] + g2_f[i+1])

    return A

for r in resists:
    gain_curves[r] = calc_g_f(r) # calculate gain curve
    gain_curves_lower[r] = calc_g_f(r, C+(C*dC)) # calculate gain curve with lower bound on C

    G_simpson[r] = calc_G_simpsons(gain_curves[r]) # simpsons method
    G_trapezoid[r] = calc_G_trapezoidal(gain_curves[r]) # trapezoid method
    G_trap_lower[r] = calc_G_trapezoidal(gain_curves_lower[r])

    G_err[r] =  (G_trapezoid[r] - G_simpson[r])/G_trapezoid[r] + (G_trapezoid[r] - G_trap_lower[r])/G_trapezoid[r] # error on trapezoidal method

    print(G_err[r])
    if r == 99.8:
        G_r_lower = calc_G_trapezoidal(calc_g_f(r - (r*0.002 + 0.1), C))
        G_err_temp[r] = G_err[r] + (G_trapezoid[r] - G_r_lower)/(G_trapezoid[r])

# print('trapezoid: ', G_trapezoid)
# print('simpsons: ', G_simpson)
# print('unc: ', G_err)

#----- plot -----#
if plot_mode == 'gain':
    print('R: ', 477.0)
    plt.plot(freqs, g2, color='black') # unadjusted gain curve
    plt.scatter(freqs, g2, marker='o', facecolors='none', color='black')
    plt.plot(freqs, gain_curves[477.0], color='red') # adjusted gain curve for a specific resistance
    plt.scatter(freqs, gain_curves[477.0], marker='o', facecolors='none', color='red')


################################### BOLTZMANN CONSTANT ###################################

# y axis values
k = list()
k_errs = list() 

# x axis values
r_discrete = array(list(resists.keys()))*1000 # Ohm
r_continuous = arange(-1000, 2000000, 1) # Ohm

volts = list() # visualize actual raw noise input
def calc_k(R):
    
    vr2 = mean(V_r[R]['values']**2)
    vs2 = mean(V_s[R]['values']**2)
    v2 = vr2 - vs2 # volts
    T = 295.15 # kelvin
    G = G_trapezoid[R] # Hz

    volts.append([v2, G])

    kR = v2/(4*T*G*R*1000) 

    dR = resists[R]['dr'][0]*(R*1000) + resists[R]['dr'][1]

    print('dv2: ', dv2[R])
    print('v2: ', dv2[R]/v2)
    print('G: ', G_err[R])
    print('R: ', dR/(R*1000))
    print('T: ', dT/T)
    print('')

    k_err = kR * sqrt( (dv2[R]/(v2))**2 + (G_err[R])**2 + (dR/(R*1000))**2  )

    return kR, k_err

    v_raw = v2/G 
    print(v_raw)

for r in resists:
    k.append(calc_k(r)[0]) # add values
    k_errs.append(calc_k(r)[1]) # add errors

for n in k_errs:
    print(n)
k_ave = mean(array(k)) # average of k values

print('kerrs: ,', k_errs)
print('k_ave: ', k_ave)

# ----- Monte Carlo for uncertainty on average -----
monte_carlo = True 
iterations = 10000
if monte_carlo:
    raffled_aves = list() # distance of each raffled line away from measured line
    for it in range(iterations):
        k_raffled = list() # raffled points
        for point in k: 
            k_raffled.append(point)
        for i in range(len(k)): # for each data point add a random offset
            k_raffled[i] += np.random.normal(0, k_errs[i])
        ave_raffled = mean(array(k_raffled))
        raffled_aves.append(abs(k_ave - ave_raffled))     

    dk_ave = std(array(raffled_aves))
    print('k_ave_unc: ', dk_ave)

print('percent unc k: ', dk_ave/k_ave)

#----- plot -----#
if plot_mode == 'k':

    blank = ones((len(r_continuous),)) # a blank constant line

    #----- data -----
    plt.scatter(r_discrete, k, s=5, color='black')
    plt.errorbar(r_discrete, k, k_errs, color='black', capsize=1.5, ls='none')

    #----- average -----
    plt.plot(r_continuous, blank*k_ave, color='red', label='kkkkkkkkkkkkkkkkkkkkk                   ') # k average
    plt.plot(r_continuous, blank*(k_ave+dk_ave), color='red', linestyle='--', label='k') # upper band
    plt.plot(r_continuous, blank*(k_ave-dk_ave), color='red', linestyle='--')

    #----- k curve -----
    plt.plot(r_continuous, blank*k_true, color='green', label='lllllllllllllllllllllllllll')

    plt.ylim(0.75e-23, 2.75e-23)  
    plt.xlim(0, 1000000)
    plt.legend() 


################################### ABSOLUTE ZERO ###################################

# y axis values
y_0 = list() # y axis values to be plotted against T values
y_0_errs = list() # y axis errors

# x axis values
t_discrete = array(list(temps.keys())) #+ 273.15 # kelvin
t_continuous = arange(-400, 300, 1)  # kelvin

def calc_0(T):
    
    vr2 = mean((array(temps[T]['r'])/1000)**2)
    vs2 = mean((array(vs_temps)/1000)**2)
    v2 = vr2 - vs2 # volts
    R = 99.8 * 1000 # ohms
    G = G_trapezoid[R/1000] # Hz

    y = v2/(4*R*G)

    dR_100 = 0.002*R + 0.1
    dT0 = k_true*dT

    # print('v2: ', dv2[99.8]/v2)
    # print('v2: ', dv2_temps[T]/v2)
    # print('G: ', G_err[R/1000])
    # print('R: ', dR_100/(R))
    # print('T: ', dT/T)

    dy = y * sqrt( (dv2_temps[T]/(v2))**2 + (G_err_temp[R/1000])**2 ) + dT0

    return y, dy

for t in temps:
    y_0.append(calc_0(t)[0]) # add values
    y_0_errs.append(calc_0(t)[1]) # add errors

print('y_0_errs: ', y_0_errs)

#----- linear fit -----
def linear_fit(x, m, b):
    func = m*x + b
    return func

def chi2_linear(observed, fit, sigma):
    chi_sq = 0
    for i in range(len(observed)):
        chi_sq += (observed[i] - fit[i])**2/sigma[i]**2
    return chi_sq

r_discrete = array(list(resists.keys()))*1000 # Ohm
r_continuous = arange(0, 1000000, 1) # Ohm

p0_y0 = array([8.06e-24, 1.6e-21]) # b for y = 0*x + b
y0_fit = curve_fit(linear_fit, t_discrete, y_0, p0_y0, y_0_errs, maxfev=500000) # best fit

m, b = y0_fit[0]
m_err = sqrt(diag(y0_fit[1])[0]) # square root of covariance matrix
b_err = sqrt(diag(y0_fit[1])[1]) # square root of covariance matrix

abs_0_measured = -b/m # celsius

chi2_y0 = chi2_linear(y_0, m*t_discrete + b, y_0_errs)
df = 7 # calculation: 9 - 2
print('linear fit: ', m, ', ', b)
print('fit error: ', m_err, ', ', b_err)
print('0 Centigrade: ', abs_0_measured) # x intercept
print('0 Centigrade True: ', abs_0)
print('degrees: ', df)
print('chi2: ', chi2_y0)
print('chi2_pdf: ', chi2_y0/df)
print('P(chi2_y0): ', 1 - sp.stats.chi2.cdf(chi2_y0, df)) # p-value

if plot_mode == '0':

    #----- data -----
    plt.scatter(t_discrete, y_0, s=3, color='black') # data
    plt.errorbar(t_discrete, y_0, y_0_errs, color='black', capsize=1.5, ls='none') # errors

    #----- k curve -----
    plt.plot(t_continuous, k_true*t_continuous - (k_true*abs_0), color='green', label='kkkkkkkkkkkkkkkkkkkkk                   ') # plot of true k
    # plt.ylim(0, 6e-21)
    plt.ylim(0, 6e-21)
    plt.xlim(-300, 160)

    #----- linear fit -----
    m_agreed = 1.4098e-23
    b_agreed = 236*m_agreed
    plt.plot(t_continuous, m_agreed*t_continuous + b_agreed, color='red', label='fit')

    plt.legend()

plt.show()

