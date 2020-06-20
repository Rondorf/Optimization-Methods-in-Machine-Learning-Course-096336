
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


def stochastic_gradient_descent(init, so, batch_size, m, num_steps, alpha, proj=lambda x: x):
    """ (projected) stochastic gradient descent.

        Inputs:
            init: starting point
            so: stochastic oracle -- gradient estimator
            batch_size: mini-batch size for SGD
            m: number of functions in finite-sum
            num_steps: total num of outer-loop steps (in each inner step make m/batch_size steps)
            proj (optional): function mapping points to points

        Returns:
            List of all the algorithm's iterates.
        """
    xs = [init]
    y = [xs[-1]]

    for t in range(num_steps):
        eta = 2 / (alpha * (t+1))
        for update in range(int(np.ceil(m / batch_size))):
            y.append(proj(y[-1] - eta * so(y[-1])))
        T = len(y)
        xs.append(np.sum([(2 * (t+1) / (T * (T+1))) * iterate for (t, iterate) in enumerate(y)], axis=0))
        # xs.append(y[-1])
        y = [y[-1]]
    return xs


def svrg(init, alpha, beta, m, stochastic_grad_func, grad_func, max_steps=100, proj=lambda x: x):
    k = int(np.ceil(20 * (beta / alpha)))
    eta = 1 / (10 * beta)

    ys = [init]
    for s in range(max_steps):
        xs = [ys[-1]]
        full_grad = (1 / m) * grad_func(ys[-1])
        for t in range(k):
            # draw random gradient estimate idx
            idx = np.random.randint(m)
            xs.append(proj(xs[-1] -
                      eta * (stochastic_grad_func(xs[-1], idx) -
                             stochastic_grad_func(ys[-1], idx) + full_grad)))
        ys.append(np.mean(np.vstack(xs), axis=0))
    return ys