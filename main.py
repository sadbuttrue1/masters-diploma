from sympy import *
import time
from msm import *
from scipy.integrate import ode
from matplotlib import pyplot as plt
import os


def plot_to_file(x, y, variable_name, function_name):
    plt.figure()
    plt.plot(x, y, label='{}({})'.format(function_name, variable_name))
    plt.grid()
    plt.legend()
    plt.ylabel('{}({})'.format(function_name, variable_name))
    plt.xlabel(variable_name)
    dir = 'images/task1'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    plt.savefig('images/task1/{}.png'.format(function_name))


start = time.time()

init_printing()

phi, theta = symbols('phi theta')
R1, R2a, R2b, R2c = symbols('R1 R2a R2b R2c')
l, d = symbols('l d')
t = symbols('t')
J = symbols('J')
A = Matrix([[0, -d]])
L2 = Matrix([[l * cos(theta(t)), l * sin(theta(t))]])
L1 = -L2
Ra = L1 - A
Rb = -A
Rc = L2 - A
ra = Ra.norm(ord=2)
rb = Rb.norm(ord=2)
rc = Rc.norm(ord=2)
rba = Matrix(Matrix(
    [
        [cos(theta(t)), -sin(theta(t))],
        [sin(theta(t)), cos(theta(t))]
    ]
).dot(Matrix([[-l, 0]])))
rbb = [0, 0]
rbc = Matrix(Matrix(
    [
        [cos(theta(t)), -sin(theta(t))],
        [sin(theta(t)), cos(theta(t))]
    ]
).dot(Matrix([[l, 0]])))
Phi = Matrix([-phi, phi, phi, phi])
R = Matrix([R1, R2a, R2b, R2c])
rA, rB, rC = symbols('rA rB rC')
D = Matrix(
    [
        [0, rA, rB, rC],
        [rA, 0, l, 2 * l],
        [rB, l, 0, l],
        [rC, 2 * l, l, 0]
    ]
)
print('Done with preps at {} seconds'.format(time.time() - start))

q = find_charges(R, D, 4, Phi)
print(q)
print('Done with charges vector at {} seconds'.format(time.time() - start))
q10 = (q.row(0))
q2a = (q.row(1))
q2b = (q.row(2))
q2c = (q.row(3))
F2a = - (k_c * q10 * q2a) / rA ** 3 * Ra
F2b = - (k_c * q10 * q2b) / rB ** 3 * Rb
F2c = - (k_c * q10 * q2c) / rC ** 3 * Rc
print('Done with forces at {} seconds'.format(time.time() - start))

Q1 = diff(rba, theta(t)).dot(F2a) + diff(rbc, theta(t)).dot(F2c)
print(Q1)
Q1 = Q1.subs(rA, ra).subs(rB, rb).subs(rC, rc)
print(Q1)
T = J * (diff(theta(t), t) ** 2) / 2
left = diff(diff(T, diff(theta(t), t)), t) - diff(t, theta(t))
print('Done with Lagrange equation at {} seconds'.format(time.time() - start))

phi_ = 20000
R2a_ = .59
R2c_ = R2a_
R2b_ = .65
R1_ = .5
l_ = 1.5
d_ = 15
J_ = 1000
left = left.subs(phi, phi_).subs(R2a, R2a_).subs(R2b, R2b_).subs(R2c, R2c_).subs(R1, R1_).subs(l, l_).subs(J, J_).subs(
    d, d_)
Q1 = Q1.subs(phi, phi_).subs(R2a, R2a_).subs(R2b, R2b_).subs(R2c, R2c_).subs(R1, R1_).subs(l, l_).subs(J, J_).subs(
    d, d_)
print('Done with substitution at {} seconds'.format(time.time() - start))

# second_derivatives = solve(Eq(left, Q1), diff(theta(t), t, t))
# print('Solved at {} seconds'.format(time.time() - start))
# print(second_derivatives)
ddtheta_tt = lambdify((theta(t)), Q1 / J_)


def dq(t, q):
    theta = q[0]
    dtheta = q[1]
    ddtheta = ddtheta_tt(theta)

    return np.append(dtheta, ddtheta)


q_0 = [1., 0.]
t_0 = 0.
solver = ode(dq).set_integrator('dopri5', nsteps=50000)
solver.set_initial_value(q_0, t_0)

sol_t = []
sol_q = []
tk = 100
while solver.t < tk:
    print(solver.t)
    solver.integrate(tk, step=True)
    sol_t.append(solver.t)
    sol_q.append(solver.y)

sol_t = np.array(sol_t)
sol_q = np.array(sol_q)
sol_theta = sol_q[:, 0]
sol_dtheta = sol_q[:, 1]
plot_to_file(sol_t, sol_theta, 't', 'theta')

plot_to_file(sol_t, sol_dtheta, 't', 'dtheta')

# gamma = 2.234e-14
# f = lambda phi: phi * np.abs(phi)
# g = lambda theta: np.sin(2 * theta)
#
# phi = np.linspace(-30e3, 30e3, num=10)
# theta = np.linspace(-100, 100, num=100)
# L = gamma * np.outer(f(phi), g(np.deg2rad(theta)))
#
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y = np.meshgrid(phi, theta)
# surf = ax.plot_surface(X, Y, L.T, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# plt.show()
#
# l = 1.1569 + 2 * 0.5909
# m = 100
# I = m * l ** 2 / 12
# phi = 10e3
#
# import os
#
#
# def plot_to_file(x, y, variable_name, function_name):
#     plt.figure()
#     plt.plot(x, y, label='{}({})'.format(function_name, variable_name))
#     plt.grid()
#     plt.legend()
#     plt.ylabel('{}({})'.format(function_name, variable_name))
#     plt.xlabel(variable_name)
#     dir = 'images/'
#     if not os.path.isdir(dir):
#         os.makedirs(dir)
#     plt.savefig('images/{}.png'.format(function_name))
#
#
# from scipy.integrate import ode
#
#
# def dd_theta(t, q, phi, I, gamma, f, g):
#     theta = q[0]
#     dtheta = q[1]
#     ddtheta = gamma * f(phi) * g(theta) / I
#     return np.append(dtheta, ddtheta)
#
#
# q_0 = [0.0, 0.1]
# t_0 = 0.
#
# solver = ode(dd_theta).set_integrator('dopri5', nsteps=1)
# solver.set_initial_value(q_0, t_0).set_f_params(phi, I, gamma, f, g)
# sol_t = []
# sol_q = []
# tk = 1000
# while solver.t < tk:
#     solver.integrate(tk, step=True)
#     sol_t.append(solver.t)
#     sol_q.append(solver.y)
# sol_t = np.array(sol_t)
# sol_q = np.array(sol_q)
# sol_theta = sol_q[:, 0]
# sol_dtheta = sol_q[:, 1]
#
# plot_to_file(sol_t, sol_theta, 't', 'theta')
# plot_to_file(sol_t, sol_dtheta, 't', 'dtheta')

print('Elapsed {} seconds'.format(time.time() - start))
