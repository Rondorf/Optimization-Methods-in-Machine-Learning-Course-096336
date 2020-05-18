
import numpy as np
from scipy.linalg import svdvals


class OracleLS:
    '''
        objective function is 1/2 * ||Ax-b||^2
        - A: (m, d)
        - b: (m,)
    '''
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def get_value(self, x):
        """Least squares objective."""
        return 0.5 * (np.linalg.norm(self.A.dot(x) - self.b) ** 2)

    def get_grad(self, x):
        """Gradient of least squares objective at x."""
        return self.A.T.dot(self.A.dot(x) - self.b)

    def get_analytic_solution(self):
        return np.linalg.inv(self.A.T.dot(self.A)).dot(self.A.T).dot(self.b)

    def get_lipschitzness(self, r):
        '''
            returns L - Lipschitzness, which only exists for constrained problem
        :param r: l2 constraint radius
        :return:
        '''
        sigma_max = svdvals(self.A)[0]
        return r * (sigma_max ** 2) + sigma_max * np.linalg.norm(self.b)

    def get_smoothness(self):
        sigma_max = svdvals(self.A)[0]
        return sigma_max ** 2

    def get_strong_convexity(self):
        '''
            if not strongly convex, returns 0
        :return:
        '''
        sigma_min = svdvals(self.A)[-1]
        return sigma_min ** 2