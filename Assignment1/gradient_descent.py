
import numpy as np
import time


def gradient_descent(init, grad_func, step_size='diminishing', tolerance=1e-5,
                     max_steps=100, proj=lambda x: x, *args, **kwargs):
    """ (projected) gradient descent.

    Inputs:
        init: starting point
        grad_func: function mapping points to gradients
        step_size: int for constant stepsize or 'Diminishing' string for R/(L*sqrt(t)).
        tolerance: gradient norm stopping criterion
        max_steps: bound total num of steps to prevent infinite loop
        proj (optional): function mapping points to points
        **kwargs:   if step_size == 'diminishing' -- includes 'diam' (R) and 'Lipschitz' (L)

    Returns:
        List of all the algorithm's iterates.
    """
    xs = [init]
    if step_size == 'diminishing':
        assert 'diam' in kwargs and 'Lipschitz' in kwargs, 'diameter/Lipschitz parameter are missing.'
        R = kwargs['diam']
        L = kwargs['Lipschitz']

    for t in range(max_steps):
        grad = grad_func(xs[-1], *args)     # get grad
        if step_size == 'diminishing':
            eta = R / (L * np.sqrt(t+1))
        else:
            eta = step_size
        xs.append(proj(xs[-1] - eta * grad))

        if np.linalg.norm(grad) <= tolerance:
            break
    return xs


def accelerated_gradient_descent(init, grad_func, step_size, max_steps=100,
                                 proj=lambda x: x, *args):
    """ (projected) accelerated gradient descent.

    Inputs:
        init: starting point
        grad_func: function mapping points to gradients
        step_size: int for constant stepsize
        max_steps: bound total num of steps to prevent infinite loop
        proj (optional): function mapping points to points
        **kwargs:   if step_size == 'diminishing' -- includes 'diam' (R) and 'Lipschitz' (L)

    Returns:
        List of all points computed by algorithm.
    """

    xs = [init]
    ys = [init]
    lambdas = [0, 1]   # lambda for step size computation - lambda_0 = 0, lambda_1 = 1.

    for t in range(max_steps):
        grad = grad_func(xs[-1], *args)     # get grad
        ys.append(proj(xs[-1] - step_size * grad))

        # compute update parameters - lambda and gamma
        lambdas.append((1 + np.sqrt(1 + 4 * lambdas[-1] ** 2)) / 2)
        gamma = (1 - lambdas[-2]) / lambdas[-1]

        xs.append((1 - gamma) * ys[-1] + gamma * ys[-2])

    return ys

