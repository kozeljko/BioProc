from bioproc.proc_models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
    

"""
    TESTING
"""

# simulation parameters
t_end = 200 * 2
N = 1000 * 2


# model parameters
alpha1 = 34.73
alpha2 = 49.36
alpha3 = 32.73
alpha4 = 49.54
delta1 = 1.93
delta2 = 0.69
Kd = 10.44
n = 4.35
params_ff = (alpha1, alpha2, alpha3, alpha4, delta1, delta2, Kd, n)


# addressing params
alpha = 20
delta = 1
Kd = 80
n = 2

params_addr = (alpha, delta, Kd, n)

points = np.loadtxt('selected_points.txt')
params = points[0]
params_ff = list(params[:8])
params_addr = list(params[8:])


# Time
T = np.linspace(0, t_end, N)


# Input parameters
# X1, X2, X3, X4, X5, X6, X7, X8
params_input = [0, 100, 0, 100, 100, 100, 0, 100, [(100, 200)]]


# 8-bit LFSR (8654)

Y0 = np.array([0] * 41)

Y = odeint(eight_bit_lfsr_8654, Y0, T, args=(params_ff, params_input))


Y_reshaped = np.split(Y, Y.shape[1], 1)
"""
Q = Y_reshaped[2::4]

for q in Q:
    plt.plot(T, q)
plt.show()
"""
Q1 = Y_reshaped[2]
not_Q1 = Y_reshaped[3]
Q2 = Y_reshaped[6]
not_Q2 = Y_reshaped[7]
Q3 = Y_reshaped[10]
not_Q3 = Y_reshaped[11]
Q4 = Y_reshaped[14]
not_Q4 = Y_reshaped[15]
Q5 = Y_reshaped[18]
not_Q5 = Y_reshaped[19]
Q6 = Y_reshaped[22]
not_Q6 = Y_reshaped[23]
Q7 = Y_reshaped[26]
not_Q7 = Y_reshaped[27]
Q8 = Y_reshaped[30]
not_Q8 = Y_reshaped[31]


plt.style.use('dark_background')

plt.subplot(8, 1, 1)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='white', alpha=0.25)
plt.plot(T, Q1, label='q1', color='tab:blue')
plt.legend()
plt.subplot(8, 1, 2)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='white', alpha=0.25)
plt.plot(T, Q2, label='q2', color='tab:orange')
plt.legend()
plt.subplot(8, 1, 3)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='white', alpha=0.25)
plt.plot(T, Q3, label='q3', color='tab:green')
plt.legend()
plt.subplot(8, 1, 4)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='white', alpha=0.25)
plt.plot(T, Q4, label='q4', color='tab:red')
plt.legend()
plt.subplot(8, 1, 5)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='white', alpha=0.25)
plt.plot(T, Q5, label='q5', color='tab:blue')
plt.legend()
plt.subplot(8, 1, 6)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='white', alpha=0.25)
plt.plot(T, Q6, label='q6', color='tab:orange')
plt.legend()
plt.subplot(8, 1, 7)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='white', alpha=0.25)
plt.plot(T, Q7, label='q7', color='tab:green')
plt.legend()
plt.subplot(8, 1, 8)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='white', alpha=0.25)
plt.plot(T, Q8, label='q8', color='tab:red')
plt.legend()



plt.savefig('figs\\lfsr8.pdf')
plt.show()
