# -*- coding: utf-8 -*-
"""
Template for to homework exercise 5, question 1
Fundamentals of CS for EE students 2023
"""

import numpy as np
# from matplotlib import pyplot as plt


def euler(f, a: float, b: float, initial_cond: float, N=None, h=None):
    # raise exceptions if needed
    if h is None and N is None:
        raise Exception
    if N is None:
        N = int((b - a)/h)
    if h is None:
        h = (b-a)/N
    if (b-a)/N != h:
        raise Exception
    # a > b
    i_to_N = np.arange(a,b+h,h)
    arr = np.zeros(N+1)

    for i in range(len(i_to_N)):
        if i < 1:
            arr[i] = initial_cond  # y(a) = a
        else:
            arr[i] = arr[i-1] + (h * f((a+(h * (i-1))), arr[i-1]))  # y(n-1) = y(n-2) + h(f( t(n-2), y(n-2))

    return i_to_N, arr


if __name__ == '__main__':
    f = lambda t, y: y - t**2 + 1
    t, y = euler(f, 0, 2, 0.5, N=10)
    y_exact = lambda t: (t + 1)**2 - 0.5 * np.exp(t)
    sol_exact = y_exact(t)

    print('t_i \t \t approx \t \t exact \t \t err')
    for t_i, y_i, ye_i in zip(t, y, sol_exact):
        print(f'{t_i:.2f}\t \t {y_i:.2f}\t \t{ye_i:.2f}\t \t {ye_i-y_i:.2f}')

    # fig1 = plot_funcs(t, y, y_exact=y_exact)
    # plt.show() # uncomment to see plot, but add comment back before submitting
    # fig2 = plot_error(t, y, y_exact)
    # plt.show() # uncomment to see plot, but add comment back before submitting

    f = lambda t, y: np.cos(2*t) + np.sin(3*t)
    t, y = euler(f, 0, 1, 1, h=0.25)
    y_exact = lambda t: 1/2 * np.sin(2*t) - 1/3 * np.cos(3*t) + 4/3

    sol_exact = y_exact(t)

    print('t_i \t \t approx \t \t exact \t \t err')
    for t_i, y_i, ye_i in zip(t, y, sol_exact):
        print(f'{t_i:.2f}\t \t {y_i:.2f}\t \t{ye_i:.2f}\t \t {ye_i-y_i:.2f}')

    # fig3 = plot_funcs(t, y, y_exact=y_exact)
    # plt.show() # uncomment to see plot, but add comment back before submitting
    # fig4 = plot_error(t, y, y_exact)
    # plt.show() # uncomment to see plot, but add comment back before submitting
