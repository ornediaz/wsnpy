''' Produce plots of optimal unequal cluster size distribution

In order to plot the results type:
   python clu1.py 0

In order to generate new results and plot them type:
   python clu1.py 1

Produce graphs about the dependency of the
optimal layer size with the network size gamma, the network size beta, and
the compression factor sigma.

'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import shutil
import subprocess
import platform
from matplotlib.font_manager import FontProperties

fn = __file__[0:__file__.rfind('.')]
compute = bool(int(sys.argv[1])) # Compute new results?
plt.rc('font', size=10)

def save(n):
    '''Adjust the fonts in the current figure and save.'''
    font2 = FontProperties(size=8,family='Times New Roman') # For legend
    for a in f.axes:
        for x in ['x','y']:
            # Set tick label fonts
            getattr(a,'set_%sticks' %x)(getattr(a,'get_%sticks'
                %x)()[0:-1:2])
            # lv = getattr(a,'get_%sticklabels' %x)()
            # lv = lv[0:-1:2]
            for l in getattr(a,'get_%sticklabels' %x)():
                l.set_fontproperties(font2)
            # Set axis label fonts
            l = getattr(a, '%saxis' %x).label.set_fontproperties(font2)
        # Set font of legends    
        l = a.get_legend()
        if l is not None:
            for t in l.get_texts():
                t.set_fontsize(7)
    f.savefig('%s_%d.pdf' %(fn, n))

def opt1(beta, gamma, sigma, event_rate=10, debug=False):
    '''Return optimal widths and consumption.

    Using a simple heuristic, approximate the optimal number of layer that
    should exist in the hierarchical FAT method.  This heuristic tests all
    the possible layers.

    This function is not very useful.

    '''
    r_b = beta * tx_rg + r_a
    s = gamma * tx_rg # Side of the square area that detects the event
    t = 0.1 * tx_rg # Increment for variations
    w1 = w0 / sigma
    A = np.pi * (r_b**2 - r_a**2)
    # Feasible positions for the lower left corner of the evento
    Af = np.pi * ((r_b - s)**2 - r_a**2) 
    layers = np.arange(2, (r_b - r_a) / hmin, dtype=int)
    consump = np.ones(len(layers)) * 9e9
    div = [[] for i in xrange(len(layers))]
    for j, nl in enumerate(layers):
        h2 = np.ones(nl) * (r_b - r_a) / nl
        while True:
            r = np.cumsum(h2) + r_a
            n = event_rate * 2 * r * (h2 + s)**2 / Af
            P = np.zeros(nl)
            for i in xrange(nl):
                P[i] = (w0 * event_rate * s**2 * h2[i] / A  + w1 *
                        sum(n[i:]) / 5 / delta / r[i]) / tx_rg
            if debug: print h2, '\n', P, "\n--number of events", sum(n[1:])
            if (consump[j] - P.max()) / consump[j] < 1e-5: break
            consump[j] = P.max()
            div[j] = h2.copy()
            imn = P.argmin()
            if h2[imn] - t < hmin: break
            h2[imn] += t
            h2[P.argmax()] -= t
    return layers, div, consump

def opt2(beta, gamma, sigma, event_rate=10, debug=False):
    '''Return optimal widths, variety improvement and quotient between
    internal and relay cost.

    This is the main computational function of the script, but does not plot
    anything.

    '''
    r_b = beta * tx_rg + r_a
    s = gamma * tx_rg # Side of the square area that detects the event
    t = 0.03 * tx_rg # Increment for variations
    w1 = w0 / sigma
    A = np.pi * (r_b**2 - r_a**2)
    # Feasible positions for the lower left corner of the event
    Af = np.pi * ((r_b - s)**2 - r_a**2) 
    layers = np.arange(2, (r_b - r_a) / hmin, dtype=int)
    # It contains the internal (intracluster) and external (relay) costs of
    # the optimal solution for each layer.
    best  = np.ones((len(layers), 2)) * 9e9
    # Power consumption before changing the uniform size distribution. 
    init = np.ones((len(layers), 2)) * 9e9
    # Best size distribution we find.
    div = [[] for i in xrange(len(layers))]
    for i, nl in enumerate(layers):
        h2 = np.ones(nl) * (r_b - r_a) / nl # Initial uniform distribution.
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
                print h2, '\n', P, "\n--number of events", sum(n[1:])
            if (best[i].sum() - P.sum(1).max()) / best[i].sum() < 1e-5:
                break
            best[i, :] = P[mx, :]
            div[i] = h2.copy()
            if h2[mn] - t < hmin: # One layer has become too thin
                break
            h2[mn] += t # Extend the least energy consuming ring
            h2[mx] -= t # Reduce the most energy consuming ring      
        else:
            raise Exception('We should not reach this point')
    i = best.sum(1).argmin()
    width = (r_b - r_a) / layers[i] / tx_rg
    imp = (init.sum(1)[i] - best.sum(1)[i]) / init.sum(1)[i]
    in_ra = init[i, 1] / init[i].sum() # initial ratio
    fi_ra = best[i, 1] / best[i].sum()# Final ra
    return width, imp * 100, in_ra, fi_ra

def pl(n, x, lb, tit): 
    z = np.load('%s_1_%d.npy' %(fn, n)).T
    # f.subplots_adjust(bottom=0.1, 
    for i in xrange(3): 
        ax = f.add_subplot(3, 3, 3 * i + n + 1)
        yv = ((z[0],), (z[1],), (z[2], z[3]))[i]
        for y, s in zip(yv, ('-', '--')):
            ax.plot(x, y, s)
        if i == 2:
            ax.set_xlabel(lb)
        if n == 0:
            ax.set_ylabel((r'mean($h_k/r_{tx}$)', 
                r'Variety improvement $\nu$ (%)', 
                r'external to total ratio $\varepsilon$')[i])
        if i == 0:
            ax.set_title(tit)
    ax.legend(('before', 'after'), loc=(4, 1, 1)[n])

r_a = 10.
w0 = 100.0 # Cost of transmitting one packet 
tx_rg = 1
rho = 8 # Normalized node density
delta = rho / np.pi / tx_rg**2 # Unnormalized node density.
hmin = max(tx_rg, np.sqrt(5.0/delta))


plt.ioff()
plt.close('all')

# Power as a function of the number of tiers, when fixing beta, gamma and
# sigma.
beta = 400.
gamma = 3.
sigma = 20.
if compute:
    layers, div, consump = opt1(beta, gamma, sigma)
    np.savez('%s_0.npz' %fn, layers=layers, consump=consump)
else:
    d = np.load('%s_0.npz' %fn)   
    layers = d['layers']
    consump = d['consump']

f = plt.figure(0)
ax = f.add_subplot(111)
hmean = beta * tx_rg / layers
ax.stem(hmean, consump)
ax.set_xlabel(r'Average layer width')
ax.set_ylabel(r'Power consumption')
# save(0)

sam = 50
f = plt.figure(1, figsize=(5,5))

beta_v = np.linspace(10, 160, sam)
gamma = 4. #Event size
sigma = 10. # Compression factor
if compute:
    np.save('%s_1_%d.npy' %(fn, 0), [opt2(b, gamma, sigma) for b in beta_v])
pl(0, beta_v, r'Network size $\beta$', 
        r'$\gamma=%1.f;\sigma=%1.f$' %(gamma, sigma))

beta = 120.
gamma_v = np.linspace(0.2, 30, sam)
sigma = 15. # Compression factor
if compute:
    np.save('%s_1_%d.npy' %(fn, 1), [opt2(beta, g, sigma) for g in gamma_v])
pl(1, gamma_v, r'normalized event size $\gamma$', 
    r'$\beta=%1.f;\sigma=%1.f$' %(beta,sigma))

beta = 80. # Network size
gamma = 4. # Event size
sigma_v = np.linspace(4., 40, sam)
if compute:
    np.save('%s_1_%d.npy' %(fn,2), [opt2(beta, gamma, s) for s in sigma_v])
pl(2, sigma_v, r'Compression factor $\sigma$',
    r'$\beta=%1.f;\gamma=%1.f$' %(beta, gamma))

save(1)
#plt.show()
orig='clu1_1.pdf'
dest='../../latex/pic_clu1_1.pdf'
shutil.copy(orig, dest)
if platform.system() == 'Linux':
    if platform.system() == 'Windows':
        subprocess.Popen(['acrord32', orig])
    else:
        subprocess.Popen(['xpdf', orig])
