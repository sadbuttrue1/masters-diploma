import numpy as np

gamma = 2.234e-14
f = lambda phi: phi * np.abs(phi)
g = lambda theta: np.sin(2 * theta)

phi = np.linspace(-30e3, 30e3, num=10)
theta = np.linspace(-100, 100, num=100)
L = gamma * np.outer(f(phi), g(np.deg2rad(theta)))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(phi, theta)
surf = ax.plot_surface(X, Y, L.T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()

l = 1.1569 + 2 * 0.5909
m = 100
I = m * l ** 2 / 12
phi = 10e3

import os


def plot_to_file(x, y, variable_name, function_name):
    plt.figure()
    plt.plot(x, y, label='{}({})'.format(function_name, variable_name))
    plt.grid()
    plt.legend()
    plt.ylabel('{}({})'.format(function_name, variable_name))
    plt.xlabel(variable_name)
    dir = 'images/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    plt.savefig('images/{}.png'.format(function_name))


from scipy.integrate import ode


def dd_theta(t, q, phi, I, gamma, f, g):
    theta = q[0]
    dtheta = q[1]
    ddtheta = gamma * f(phi) * g(theta) / I
    return np.append(dtheta, ddtheta)


q_0 = [0.0, 0.1]
t_0 = 0.

solver = ode(dd_theta).set_integrator('dopri5', nsteps=1)
solver.set_initial_value(q_0, t_0).set_f_params(phi, I, gamma, f, g)
sol_t = []
sol_q = []
tk = 1000
while solver.t < tk:
    solver.integrate(tk, step=True)
    sol_t.append(solver.t)
    sol_q.append(solver.y)
sol_t = np.array(sol_t)
sol_q = np.array(sol_q)
sol_theta = sol_q[:, 0]
sol_dtheta = sol_q[:, 1]

plot_to_file(sol_t, sol_theta, 't', 'theta')
plot_to_file(sol_t, sol_dtheta, 't', 'dtheta')
