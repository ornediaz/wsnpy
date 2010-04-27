'''
This script creates graphs that compute how the cost varies with the number
of tiers and computes the optimal clustersize division.

This algorithm is pointless. When the most energy consuming tier cannot be
reduced any more, there is no point in reducing the energy consumption of
other tiers. We are only interested in the biggest energy consumption.

'''
import numpy as ny
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
#from scipy.integrate import dblquad
r_a = 10.
r_b = 100.
s = 10. # Side of the square area that detects the event
A = ny.pi * (r_b**2 - r_a**2)
# Feasible positions for the lower left corner of the evento
Af = ny.pi * ((r_b - s)**2 - r_a**2) 


w1 = 100.0 # Cost of transmitting one packet 
w2 = 1.0 # Cost of transmitting one compressed packet from many packets
tx_rg = 1
rho = 8
delta = rho / ny.pi / tx_rg**2
hmin = max(tx_rg, ny.sqrt(5.0/delta))
nlayers = range(5,int((r_b-r_a)/hmin),15)

Gamma = 10 # Number of events per unit time




t = 0.1 # Increment for variations
deb = 0

def a(h):
    r = ny.cumsum(h) + r_a
    n = Gamma * 2 * r * (h + s)**2 / Af
    if deb:
        print "--number of events", sum(n[1:])
    P = ny.zeros(nl)
    for i in xrange(nl):
        P[i] = (w1 * Gamma * s**2 * h[i] / A  
                + w2 * sum(n[i:]) / 5 / delta / r[i] ) / tx_rg
    return P
evol = []
plt.close('all')
plt.ioff()
#fig = plt.figure(1, figsize=(15, 8 ))

n_tests = 300
rec = ny.zeros((n_tests,len(nlayers)))
fig = plt.figure(1)
for k, w2 in enumerate([1,20,50]):
    for j, nl in enumerate(nlayers):

        hin = (r_b - r_a) / nl # Initial guess
        assert hin > hmin, "Too many layers"
        h = ny.ones(nl) * hin
        p2 = 9e9
        evol = []
        for i in xrange(n_tests):
            P = a(h)
            if deb:
                print 20 * '*'
                print 'Iteration %d' %i
                print h
                print P
            p1 = p2
            p2 = P.max()
            rec[i,j] = p2
            #if (p1 - p2) / p1 < 1e-5:
            #    break
            # Compute the new 
            imx = P.argmax()
            imn = (P + 9e9 * (h < t + hmin)).argmin()
            h[imn] += t
            h[imx] -= t       
        #ax = fig.add_subplot(ny.ceil(len(nlayers)/3.),3,j + 1)
        #ax.plot(ny.arange(len(evol)) + 1, evol)
        #ax.set_title("%d layers" %nl)

    ax = fig.add_subplot(1,3,k + 1)
    ax.plot(rec)
    if k == 0:
        ax.legend([r'$l = %d$' %i for i in nlayers])
    ax.axis([0,n_tests,0,max(rec[0,:]) * 1.1])
    ax.set_title('w2 = %1.f' %w2)
    
fig.savefig('a.eps')
os.system('ps2pdf a.eps a.pdf')
plt.show()


#def show_distance():
#    '''Compute the average distance to the cluster size.'''
#    def g(l):
#        u = dblquad(lambda y, x: ny.sqrt(x**2 + y**2), 
#            0.0, l/2, lambda x: 0.0, lambda x: l)
#        return u[0] / (l**2 /2) / l
#    for x in xrange(2,40):
#        print g(x)
#
#    def f(r):
#        u = dblquad(
#            lambda y, x: 1.0,
#            0.0, r,
#            lambda x: 0.0,
#            lambda x: ny.sqrt( r**2 - x**2),
#            )
#        return (u[0] * 4 / ny.pi) ** 0.5 /r
#    print 'Second test'
#    for x in xrange(2,6):
#        print f(x)
