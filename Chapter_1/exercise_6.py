from math import pi
import numpy as np

N = 100                                     # number of inputs
u_0 = np.linspace(0, 1, N).reshape((-1,1))  # no noise inputs
y_0 = np.tan(u_0 * 0.9 * pi / 2)            # no noise generated data

# generat K matrix
n_max = 20

K = u_0
for k in range(1, n_max):
    np.concatenate((K, u_0**k), axis=1)

theta_LS = np.linalg.inv(K.T @ K) @ K.T @ y_0
print(theta_LS)
w_mat = np.identity(N)

