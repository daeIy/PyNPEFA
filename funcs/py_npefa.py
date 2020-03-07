# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 22:34:56 2020

@author: Gabriel Daely
https://github.com/daeIy
"""

import numpy as np
from scipy import signal
from cvxopt import matrix
from spectrum.burg import _arburg2
import matplotlib.pyplot as plt
from l1tf_lm import l1tf_lm
from l1tf import l1tf

def py_npefa(y,x):
    """
    Generating integrated predicition error filter analysis curve.

    Parameters
    ----------
    y : numpy.ndarray or pandas.Series
        1-D array of original curve data.
    x : numpy.ndarray
        1-D array of date/depth data.

    Returns
    -------
    ipfy : dict
        Containing original data (ipfy['OG']), long term INPEFA (ipfy['1']),
        mid term INPEFA (ipfy['2']), short term INPEFA (ipfy['3']), and
        shorter term INPEFA (ipfy['4']),

    """
    y = y.to_numpy()

    # Set maximum regularization parameter
    lambdamax = l1tf_lm(y)
    z = {}
    z['0'] = matrix(y)

    # l1 trend filtering
    for i in range(1,10):
        z['{0}'.format(i)] = l1tf(z['{0}'.format(i-1)],10**(-10+i)*lambdamax)

    # Set trend filter for long, mid, short, and shorter term
    fy = {}
    fy['1'] = z['0']-(z['1']+z['2']+z['3']+z['4']+z['5']+z['6']+z['7']+z['8']+z['9'])/9.0
    fy['2'] = z['0']-(z['1']+z['2']+z['3']+z['4']+z['5']+z['6']+z['7'])/7.0
    fy['3'] = z['0']-(z['1']+z['2']+z['3']+z['4']+z['5'])/5.0
    fy['4'] = z['0']-(z['1']+z['2']+z['3'])/3.0

    # Compute Burg Filter, Prediction Error, and Integrated Prediction Error
    ipfy = {}
    ipfy['OG'] = z['0']
    for j in range(1,5):
        # Burg's AR coeff
        bffy = _arburg2(fy['{0}'.format(j)],32)[0].real
        # PEFA
        pffy = signal.convolve(fy['{0}'.format(j)],
                               np.reshape(bffy,(len(bffy), 1)),
                               mode='same')
        # INPEFA
        iipfy = np.cumsum(pffy[::-1])[::-1]
        # Normalized to -1 <= INPEFA <= 1
        ipfy['{0}'.format(j)] = iipfy / max(abs(iipfy))

    plt.subplot(151)
    plt.plot(ipfy['OG'],-x) # Original signal

    plt.subplot(152)
    plt.plot(ipfy['1'],-x) # Long term INPEFA

    plt.subplot(153)
    plt.plot(ipfy['2'],-x) # Mid term INPEFA

    plt.subplot(154)
    plt.plot(ipfy['3'],-x) # Short term INPEFA

    plt.subplot(155)
    plt.plot(ipfy['4'],-x) # Shorter term INPEFA

    plt.show()

    return ipfy
