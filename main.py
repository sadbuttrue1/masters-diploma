import numpy as np

gamma = 2.234e-14
f = lambda phi: phi * np.abs(phi)
g = lambda theta: np.sin(2 * theta)

phi = np.linspace(-30, 30, num=10)
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
