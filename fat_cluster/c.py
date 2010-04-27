'''
This file can be deleted. It is simply a copy of a.py to play with.
'''
import numpy as ny
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#from scipy.integrate import dblquad
ra = 10.
rb = 100.
s = 10. # Side of the square area that detects the event
A = ny.pi * (rb**2 - ra**2)
# Feasible positions for the lower left corner of the evento
Af = ny.pi * ((rb - s)**2 - ra**2) 


w1 = 100.0 # Cost of transmitting one packet 
w2 = 1.0 # Cost of transmitting one compressed packet from many packets
rtx = 1
rho = 8
delta = rho / ny.pi / rtx**2
hmin = max(rtx, ny.sqrt(5.0/delta))
nlayers = range(5,int((rb-ra)/hmin),15)

Gamma = 10 # Number of events per unit time




t = 0.1 # Increment for variations
deb = 1

def a(h):
    r = ny.cumsum(h) + ra
    n = Gamma * 2 * r * (h + s)**2 / Af
    if deb:
        print "--number of events", sum(n[1:])
    P = ny.zeros(nl)
    for i in xrange(nl):
        P[i] = (w1 * Gamma * s**2 * h[i] / A  
                + w2 * sum(n[i:]) / 5 / delta / r[i] ) / rtx
    return P
evol = []
plt.close('all')
plt.ioff()
#fig = plt.figure(1, figsize=(15, 8 ))

ntests = 10
rec = list()
fig = plt.figure(1)
w2 = 50
nl = 20
    
        # Initialize distance
hin = (rb - ra) / nl
assert hin > hmin, "Too many layers"
h = ny.ones(nl) * hin
hv = ny.zeros((nl,ntests))
p2 = 9e9
evol = []
for i in xrange(ntests):
    P = a(h)
    hv[:,i] = h
    if deb:
        print 20 * '*'
        print 'Iteration %d' %i
        print h
        print P
    p1 = p2
    p2 = P.max()
    rec.append(p2)
    #if (p1 - p2) / p1 < 1e-5:
    #    break
    # Compute the new 
    imx = P.argmax()
    imn = (P + 9e9 * (h < t + hmin)).argmin()
    h[imn] += t
    h[imx] -= t
plt.plot(rec)
#ax = fig.add_subplot(ny.ceil(len(nlayers)/3.),3,j + 1)
#ax.plot(ny.arange(len(evol)) + 1, evol)
#ax.set_title("%d layers" %nl)

    
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
