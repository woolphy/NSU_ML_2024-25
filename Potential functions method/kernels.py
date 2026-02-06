import numpy as np

class Kernels():
    def Newton(self, r, a = 1):
        return 1.0 / (r + a)

    def Tri_cube(self, r):
        return (1 - r**3)**3

    def Gaussian(self, r):
        return 1/(2*np.pi)**(1/2)*np.exp(-1/2*r**2)

    def Epanechnikov(self, r):
        return 3/4*(1-r**2)