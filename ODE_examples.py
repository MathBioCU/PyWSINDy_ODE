import numpy as np
from scipy.integrate import solve_ivp

def lorenz(x, sigma, beta, rho):
    a = sigma*(x[1] - x[0])
    b = x[0]*(rho - x[2]) - x[1]
    c = x[0]*x[1] - beta*x[2]
    return np.array([a, b, c])


def simODE(x0, t_span, t_eval, tol_ode, ode_name, params, noise_ratio):
    if ode_name == 'Linear':
        A = params[0]
        def rhs(t, x): return A.dot(x)
        weights = []
        for i in range(len(A[0])):
            weights.append(np.insert(np.identity(
                len(A[0])), 2, np.array((A[i, :])), axis=1))
    elif ode_name == 'Logistic_Growth':
        pow = 2  # params[0]
        def rhs(t, x): return x - x**pow
        weights = [np.array([[1, 1],    [pow, -1]])]
    elif ode_name == 'Duffing':
        mu = params[0]
        alpha = params[1]
        beta = params[2]
        def rhs(t, x): return np.array([x[1], -mu*x[1] - alpha*x[0] - beta*x[0]**3])
        weights = [np.reshape(np.array([0, 1, 1]), (1, 3)), np.array(
            [[1, 0, -alpha], [0, 1, -mu], [3, 0, -beta]])]
    elif ode_name == 'Lotka_Volterra':
        alpha = params[0]
        beta = params[1]
        delta = params[2]
        gamma = params[3]
        def rhs(t, x): return np.array([alpha*x[0] - beta*x[0]*x[1], delta*x[0]*x[1] - gamma*x[1]])
        weights = [np.array([[1, 0, alpha], [1, 1, -beta]]),
                   np.array([[0, 1, -gamma], [1, 1, delta]])]
    elif ode_name == 'Van_der_Pol':
        mu = params[0]
        def rhs(t, x): return np.array([x[1], mu*x[1] - mu*x[0]**2*x[1] - x[0]])
        weights = [np.reshape(np.array([0, 1, 1]), (1, 3)), np.array(
            [[1, 0, -1], [0, 1, mu], [2, 1, -mu]])]
    elif ode_name == 'Lorenz':
        sigma = params[0]
        beta = params[1]
        rho = params[2]
        def rhs(t, x): return lorenz(x, sigma, beta, rho)
        weights = [np.array([[0, 1, 0, sigma], [1, 0, 0, -sigma]]), np.array([[1, 0, 0, rho], [1, 0, 1, -1], [0, 1, 0, -1]]), np.array([[1, 1, 0, 1], [0, 0, 1, -beta]])]

    sol = solve_ivp(fun=rhs, t_eval=t_eval, t_span=t_span, y0=x0, rtol=tol_ode)
    

    x = sol.y.T
    xobs = addNoise(x, noise_ratio)
    return weights, sol.t, xobs, rhs


def addNoise(x, noise_ratio):
    signal_power = np.sqrt(np.mean(x**2))
    sigma = noise_ratio*signal_power
    noise = np.random.normal(0, sigma, x.shape)
    xobs = x + noise
    return xobs
