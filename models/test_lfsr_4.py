from bioproc.proc_models import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
    

"""
    TESTING
"""

# simulation parameters
t_end = 300 * 2
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
# X1, X2, X3, X4
params_input = [0, 100, 0, 100, [(100, 200)]]


# Uncomment the model you want to test


# # 4-bit SR
# # a1, not_a1, q1, not_q1, a2, not_a2, q2, not_q2, a3, not_a3, q3, not_q3, a4, not_a4, q4, not_q4
# Y0 = np.array([0] * 16)

# Y0[0] = Y0[2] = 1 # a1
# Y0[4] = Y0[6] = 1 # a2
# Y0[8] = Y0[10] = 0 # a3
# Y0[12] = Y0[14] = 0 # a4
# Y = odeint(four_bit_sr, Y0, T, args=(params_ff, ))



# # 4-bit LFSR (34)
# # a1, not_a1, q1, not_q1, a2, not_a2, q2, not_q2, a3, not_a3, q3, not_q3, a4, not_a4, q4, not_q4, d1_in, d2_in, d3_in, d4_in, xor34
# Y0 = np.array([0] * 21)

# Y = odeint(four_bit_lfsr_34, Y0, T, args=(params_ff, params_input))



# # 4-bit LFSR (234)
# # a1, not_a1, q1, not_q1, a2, not_a2, q2, not_q2, a3, not_a3, q3, not_q3, a4, not_a4, q4, not_q4, d1_in, d2_in, d3_in, d4_in, xor34, not_xor34, xor234
# Y0 = np.array([0] * 23)
# Y = odeint(four_bit_lfsr_234, Y0, T, args=(params_ff, params_input))



# 4-bit LFSR Advanced (234)
# a1, not_a1, q1, not_q1, a2, not_a2, q2, not_q2, a3, not_a3, q3, not_q3, a4, not_a4, q4, not_q4, d1_in, d2_in, d3_in, d4_in,
# xor34, not_xor34, xor234, q3_or_q4, nq3_or_nq4, q2_or_x34, nq2_or_nx34
Y0 = np.array([0] * 27)
Y = odeint(four_bit_lfsr_234_adv, Y0, T, args=(params_ff, params_input))



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


# Plot offset
off = int(186 * N / t_end)
T = T[off:]
Q1 = Q1[off:]
Q2 = Q2[off:]
Q3 = Q3[off:]
Q4 = Q4[off:]

plt.rcParams['figure.figsize'] = 8, 5

#plt.style.use('dark_background')

plt.subplot(4, 1, 1)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='black', alpha=0.25)
plt.plot(T, Q1, label='q1', color='tab:blue')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='black', alpha=0.25)
plt.plot(T, Q2, label='q2', color='tab:orange')
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='black', alpha=0.25)
plt.plot(T, Q3, label='q3', color='tab:green')
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='black', alpha=0.25)
plt.plot(T, Q4, label='q4', color='tab:red')
plt.legend()

plt.subplots_adjust(0.06, 0.05, 0.94, 0.95, hspace=0.28)


# plt.plot(T, Q1, label='q1')
# plt.plot(T, Q2, label='q2')
# plt.plot(T, Q3, '--', label='q3')
# plt.plot(T, Q4, '--', label='q4')

# plt.plot(T, not_Q1, label='not q1')
# plt.plot(T, not_Q2, label='not q2')

# plt.plot(T, get_clock(T),  '--', linewidth=2, label="CLK", color='black', alpha=0.25)
# plt.legend()


plt.savefig('figs\\lfsr4.pdf')
plt.show()
