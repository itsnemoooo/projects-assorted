import numpy as np

x1 = np.array([1., 1.])
x2 = np.array([2., 1.])

def V(w, x):
    return np.dot(w, x)

def expected_update(w, pi, mu, trace_parameter, discount, lr=0.1):
    reward_tp1 = 0
    gamma = discount
    alpha = lr

    expected_V = pi * V(w, x1) + (1-pi) * V(w, x2)

    e1 = x1
    e2 = x2

    g_t_lambda = gamma * (1 - trace_parameter) * expected_V * (1 / (1 - gamma * trace_parameter))

    td_error_1 = g_t_lambda - V(w, x1)
    td_error_2 = g_t_lambda - V(w, x2)

    dw_x1 = alpha * td_error_1 * e1
    dw_x2 = alpha * td_error_2 * e2

    w_update = (mu) * dw_x1 + (1-mu) * dw_x2

    return w_update

def generate_ws(w, pi, mu, l, g):
    """Apply the expected update 1000 times"""
    ws = [w]
    for _ in range(1000):
        w = w + expected_update(w, pi, mu, l, g, lr=0.1)
        ws.append(w)
    return np.array(ws)
