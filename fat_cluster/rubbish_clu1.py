''' ??Rubbish file??

Produce graphs about the dependency of the optimal layer size with the
network size gamma, the network size beta, and the compression factor sigma.

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

file_name = __file__[0:__file__.rfind('.')]
def opt2(beta, gamma, sigma, event_rate=10, debug=False):
    '''Return optimal widths, variety improvement and quotient between
    internal and relay cost.'''
    r_b = beta * tx_rg + r_a
    s = gamma * tx_rg # Side of the square area that detects the event
    t = 0.03 * tx_rg # Increment for variations
    w1 = w0 / sigma
    A = np.pi * (r_b**2 - r_a**2)
    # Feasible positions for the lower left corner of the event
    Af = np.pi * ((r_b - s)**2 - r_a**2) 
    layers = np.array([6], dtype=int)
    # It contains the internal (intracluster) and external (relay) costs of
    # the optimal solution for each layer.
    con  = np.ones((len(layers), 2)) * 9e9
    # Initial power consumption for the 
    init = np.ones((len(layers), 2)) * 9e9
    # Optimal widths for each number of layers
    div = [[] for i in xrange(len(layers))]
    for i, nl in enumerate(layers):
        h2 = np.ones(nl) * (r_b - r_a) / nl
        for j in xrange(999): # iteration number
            r = np.cumsum(h2) + r_a
            n = event_rate * 2 * r * (h2 + s)**2 / Af
            P = np.zeros((nl, 2))
            for k in xrange(nl):
                P[k, 0] = w0 * event_rate * s**2 * h2[k] / A  / tx_rg
                P[k, 1] = w1 * sum(n[k:]) / 5 / delta / r[k] / tx_rg 
            mx = P.sum(1).argmax()
            mn = P.sum(1).argmin()
            if j == 0:
                init[i,:] = P[mx,:]
            if debug: 
                print h2, '\n', 
                print np.hstack((P, P.sum(1).reshape((-1, 1)))) 
                print  "--number of events" 
                print sum(n[1:])
            if (con[i].sum() - P.sum(1).max()) / con[i].sum() < 1e-5:
                break
            con[i, :] = P[mx, :]
            div[i] = h2.copy()
            if h2[mn] - t < hmin: 
                break
            h2[mn] += t
            h2[mx] -= t       
        else:
            raise Exception('We should not reach this point')
    i = con.sum(1).argmin()
    width = (r_b - r_a) / layers[i] / tx_rg
    imp = (init.sum(1)[i] - con.sum(1)[i]) / init.sum(1)[i]
    in_ra = init[i, 1] / init[i].sum() # initial ratio
    fi_ra = con[i, 1] / con[i].sum()# Final ra
    return width, imp, (in_ra, fi_ra)
                    
r_a = 10.
w0 = 100.0 # Cost of transmitting one packet 
tx_rg = 1
rho = 8 # Normalized node density
delta = rho / np.pi / tx_rg**2 # Unnormalized node density.
hmin = max(tx_rg, np.sqrt(5.0/delta))
beta = 10
gamma =  4
sigma = 10

opt2(beta, gamma, sigma, event_rate=10, debug=True)

